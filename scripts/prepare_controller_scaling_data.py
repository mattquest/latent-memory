#!/usr/bin/env python3
"""Freeze fresh controller-scaling questions from existing local MuSiQue inputs.

CPU only. Selection reads benchmark inputs and identity ledgers, never predictions
or metrics. Gold annotations and support sizes remain scoring-only metadata.
The existing global corpus is verified and copied byte-for-byte.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.datasets import sha256_file, write_jsonl
from scripts.prepare_adaptive_musique import (
    ARCHIVE_SHA256, MEMBER, build_corpus, normalize_questions, structural_hops,
)

CORPUS_SHA256 = "4e8ad63e12ab37e7fabec0158241d890a96171e006d226a7f22b6d00d594ce4a"
MEMBER_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
PRIOR_INPUTS = {
    "data/musique_dev_sample.jsonl": "f4edc244a51c7c566dbffc26d22149b177c27f345e34e195745d567a4d201432",
    "data/adaptive_musique_dev.jsonl": "a7627e89d17ee6933b6b810406784dbc9e115264432dcf4df34b3fbce09f87a3",
    "data/adaptive_musique_test.jsonl": "e09e7d8e3aa87cd81bb05d055013c9e9ab6e97ed7688a249ba6dd4563b53124f",
}
AUDIT = "reports/2026-09-11/development-overlap-audit.json"
NORMALIZATION = "Unicode NFKC, casefold, split/join Unicode whitespace; preserve punctuation and articles"


def normalized_question(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Question text must be a nonempty string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity(row, source):
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError(f"Invalid identity in {source}")
    return {"id": row["id"], "question": row["question"],
            "normalized_question": normalized_question(row["question"]), "source": source}


def identities_from_audit(audit):
    """Use only explicit examples; never inspect scores, generated text, or flags."""
    if not isinstance(audit.get("observed_model_pilots"), list) or not isinstance(audit.get("cohorts"), dict):
        raise ValueError("Prior audit lacks pilot/cohort identity ledgers")
    result = []
    for pilot in audit["observed_model_pilots"]:
        result.extend(identity(row, "pilot:" + pilot["source"]) for row in pilot["examples"])
    for name, cohort in sorted(audit["cohorts"].items()):
        result.extend(identity(row, "cohort:" + name) for row in cohort["examples"])
    return result


def select_questions(rows, exclusions, seed, dev_per_hop=4, test_per_hop=32):
    if min(dev_per_hop, test_per_hop) < 1:
        raise ValueError("Per-hop counts must be positive")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source question IDs")
    by_id, by_question = defaultdict(set), defaultdict(set)
    for row in exclusions:
        by_id[row["id"]].add(row["source"])
        by_question[normalized_question(row["question"])].add(row["source"])
    eligible, blocked = defaultdict(list), []
    for row in rows:
        norm = normalized_question(row["question"])
        reasons = sorted(by_id[row["id"]] | by_question[norm])
        if reasons:
            blocked.append({"id": row["id"], "question": row["question"], "normalized_question": norm,
                            "id_match_sources": sorted(by_id[row["id"]]),
                            "normalized_question_match_sources": sorted(by_question[norm])})
        else:
            eligible[structural_hops(row)].append(row)
    chosen, used_questions = {"dev": [], "test": []}, set()
    for hop in (2, 3, 4):
        ordered = sorted(eligible[hop], key=lambda row: (hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"]))
        distinct = []
        for row in ordered:
            norm = normalized_question(row["question"])
            if norm in used_questions:
                continue
            distinct.append(row)
            used_questions.add(norm)
            if len(distinct) == dev_per_hop + test_per_hop:
                break
        if len(distinct) != dev_per_hop + test_per_hop:
            raise ValueError(f"Insufficient distinct eligible {hop}-hop questions")
        chosen["dev"].extend(distinct[:dev_per_hop])
        chosen["test"].extend(distinct[dev_per_hop:])
    return chosen, {str(hop): len(eligible[hop]) for hop in (2, 3, 4)}, sorted(blocked, key=lambda row: row["id"])


def verify_corpus(corpus_bytes, rebuilt):
    parsed = [json.loads(line) for line in corpus_bytes.splitlines() if line.strip()]
    if any(set(row) != {"id", "title", "text"} for row in parsed):
        raise ValueError("Retrieval corpus contains forbidden metadata")
    if parsed != rebuilt:
        raise ValueError("Existing corpus does not match the official archive")
    if len({row["id"] for row in parsed}) != len(parsed):
        raise ValueError("Duplicate corpus IDs")
    return parsed


def support_sizes(questions, corpus, tokenizer, per_document_cap=300):
    """Scoring-only sizes; do not use these to choose questions or prompt models."""
    documents = {row["id"]: row for row in corpus}
    result = []
    for question in questions:
        sizes = []
        for doc_id in question["supporting_context_ids"]:
            document = documents[doc_id]
            fragment = f"\n<document>\n{document['title']}\n{document['text']}\n</document>\n"
            count = len(tokenizer.encode(fragment, add_special_tokens=False).ids)
            sizes.append({"id": doc_id, "full_fragment_tokens": count,
                          "capped_fragment_tokens": min(per_document_cap, count)})
        result.append({"example_id": question["id"], "split": question["split"], "category": question["category"],
                       "support_documents": sizes, "support_document_count": len(sizes),
                       "full_support_fragment_tokens": sum(row["full_fragment_tokens"] for row in sizes),
                       "capped_support_fragment_tokens": sum(row["capped_fragment_tokens"] for row in sizes)})
    return result


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def prepare(root, output_dir, seed=20260921, dev_per_hop=4, test_per_hop=32,
            tokenizer_json=None):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; frozen inputs are never overwritten")
    archive = root / "data/raw/musique.zip"
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Official MuSiQue archive checksum mismatch")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.getinfo(MEMBER).file_size > 128 * 1024**2:
            raise ValueError("Archive member exceeds preparation size bound")
        member_bytes = zipped.read(MEMBER)
    if hashlib.sha256(member_bytes).hexdigest() != MEMBER_SHA256:
        raise ValueError("Official answerable development member checksum mismatch")
    rows = [json.loads(line) for line in member_bytes.splitlines() if line.strip()]
    if not all(row.get("answerable") is True for row in rows):
        raise ValueError("Expected answerable questions only")
    source_bytes, exclusions = {}, []
    for name, digest in PRIOR_INPUTS.items():
        raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Prior normalized input checksum mismatch: {name}")
        source_bytes[name] = raw
        exclusions.extend(identity(json.loads(line), name) for line in raw.splitlines() if line.strip())
    source_bytes[AUDIT] = (root / AUDIT).read_bytes()
    audit = json.loads(source_bytes[AUDIT])
    for name, digest in PRIOR_INPUTS.items():
        if audit["data_sha256"].get(name) != digest:
            raise ValueError(f"Prior audit/input receipt mismatch: {name}")
    exclusions.extend(identities_from_audit(audit))
    source_map_by_id = {row["id"]: row for row in rows}
    for excluded in exclusions:
        if excluded["id"] in source_map_by_id and normalized_question(source_map_by_id[excluded["id"]]["question"]) != excluded["normalized_question"]:
            raise ValueError("Previously seen source ID has mismatched question text")
    # Structural selection finishes before token-size diagnostics are computed.
    selected, eligible, blocked = select_questions(rows, exclusions, seed, dev_per_hop, test_per_hop)
    corpus_path = root / "data/adaptive_musique_corpus.jsonl"
    corpus_bytes = corpus_path.read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != CORPUS_SHA256:
        raise ValueError("Existing global corpus checksum mismatch")
    rebuilt, provenance, source_map = build_corpus(rows)
    corpus = verify_corpus(corpus_bytes, rebuilt)
    corpus_receipt = {"path": "corpus.jsonl", "sha256": CORPUS_SHA256}
    questions = {split: normalize_questions(values, split, source_map, corpus_receipt, seed)
                 for split, values in selected.items()}
    for split, values in questions.items():
        for row in values:
            row["metadata"]["protocol"] = "controller_scaling_fresh_questions_v1"
    token_stats = None
    sizes = None
    if tokenizer_json:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from tokenizers import Tokenizer
        tokenizer_json = Path(tokenizer_json)
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        sizes = support_sizes(questions["dev"] + questions["test"], corpus, tokenizer)
        token_stats = {"tokenizer_json_sha256": sha256_file(tokenizer_json),
                       "add_special_tokens": False, "per_document_cap": 300,
                       "fragment_template": "\\n<document>\\n{title}\\n{text}\\n</document>\\n",
                       "used_for_selection": False, "used_in_prompts": False,
                       "excludes": "System, question, controller, answer, and chat overhead"}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".controller-scaling-", dir=output_dir.parent))
    try:
        outputs = []

        def put(name, raw):
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            receipt = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            outputs.append(receipt)
            return receipt

        def put_rows(name, values):
            receipt = write_jsonl(staging / name, values)
            receipt["path"] = name
            outputs.append(receipt)
            return receipt

        def archive_source(name, raw, source):
            receipt = put(name, gzip.compress(raw, mtime=0))
            receipt.update(compression="gzip", source=source, source_bytes=len(raw),
                           source_sha256=hashlib.sha256(raw).hexdigest())

        put("corpus.jsonl", corpus_bytes)
        for split, values in questions.items():
            put_rows(f"{split}.jsonl", values)
        put_rows("corpus-provenance.jsonl", provenance)
        put_rows("excluded-identities.jsonl", sorted(exclusions, key=lambda row: (row["source"], row["id"], row["question"])))
        put_rows("excluded-source-questions.jsonl", blocked)
        if sizes is not None:
            put_rows("support-sizes.scoring-only.jsonl", sizes)
        archive_source("provenance/musique_ans_v1.0_dev.jsonl.gz", member_bytes, MEMBER)
        for name, raw in source_bytes.items():
            archive_source("provenance/" + name + ".gz", raw, name)
        source_scripts = [Path(__file__), ROOT / "scripts/prepare_adaptive_musique.py", ROOT / "eval/datasets.py"]
        for source in source_scripts:
            put("provenance/source/" + str(source.relative_to(ROOT)), source.read_bytes())
        license_text = ("# MuSiQue data license\n\nTrivedi et al. (2022), MuSiQue. "
                        "Source: https://github.com/StonyBrookNLP/musique . "
                        "License: CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/ .\n\n"
                        "Changes: exact title/text deduplication, content-hash paragraph IDs, "
                        "local structurally balanced question splits, prior-question exclusions, "
                        "normalized JSONL, and separate scoring/provenance metadata. "
                        "This is a local subset of the public answerable development split, "
                        "not the publisher's hidden test set or full-Wikipedia benchmark.\n")
        put("DATA_LICENSE.md", license_text.encode())
        manifest = {"schema_version": 1, "protocol": "controller-scaling-data-v1", "seed": seed,
                    "generator_sha256": sha256_file(Path(__file__)),
                    "selection_rule": "For hop groups 2, 3, 4: order by SHA256(seed:source_id), exclude previous IDs and normalized questions, suppress normalized duplicates, take dev first then test. No outcomes or token sizes used.",
                    "question_normalization": NORMALIZATION, "dev_per_hop": dev_per_hop,
                    "test_per_hop": test_per_hop, "eligible_source_rows_by_hop": eligible,
                    "source": {"archive_sha256": ARCHIVE_SHA256, "member": MEMBER, "member_sha256": MEMBER_SHA256,
                               "source_rows": len(rows), "source_revision": "922ac98f19a201998dbdae6d7f2887a5258dbdeb",
                               "url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view"},
                    "corpus": {"path": "corpus.jsonl", "source_path": "data/adaptive_musique_corpus.jsonl",
                               "sha256": CORPUS_SHA256, "documents": len(corpus), "bytes": len(corpus_bytes),
                               "fields": ["id", "title", "text"], "byte_identical_to_prior_corpus": True},
                    "exclusions": {"identity_ledger_rows": len(exclusions), "unique_seen_ids": len({row["id"] for row in exclusions}),
                                   "unique_seen_normalized_questions": len({row["normalized_question"] for row in exclusions}),
                                   "excluded_musique_source_rows": len(blocked),
                                   "prior_mu_input_rows": {name: len(raw.splitlines()) for name, raw in source_bytes.items() if name in PRIOR_INPUTS},
                                   "audit_sha256": hashlib.sha256(source_bytes[AUDIT]).hexdigest()},
                    "selection": {split: {"count": len(values), "question_ids": [row["id"] for row in values],
                                          "hop_counts": dict(sorted(Counter(str(row["metadata"]["hop_count"]) for row in values).items()))}
                                  for split, values in questions.items()},
                    "support_size_encoding": token_stats,
                    "validation": {"prior_inputs_match_pinned_hashes_and_audit": True,
                                   "dev_test_and_seen_ids_normalized_questions_disjoint": True,
                                   "corpus_matches_official_archive_and_prior_bytes": True,
                                   "retrieval_records_only_id_title_text": True,
                                   "support_mapping_complete_and_annotations_agree": True,
                                   "no_model_outputs_or_metrics_used_for_selection": True},
                    "limitations": ["Local splits of public development questions; public pretraining exposure is unknown.",
                                    "Question disjointness does not imply disjoint component facts, source articles, or templates.",
                                    "Prior exclusion coverage is bounded by preserved input/identity ledgers, not unrecorded human inspection.",
                                    "Gold answers/support IDs/hop counts and support sizes are scoring/provenance only; controllers may read only questions and corpus content.",
                                    "This corpus supplies 2-4-hop questions; size alone does not establish long productive retrieval trajectories or necessary-information overflow."],
                    "outputs": outputs}
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        # Another creator cannot replace an existing nonempty frozen directory.
        if output_dir.exists():
            raise ValueError("Output directory appeared during preparation")
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/controller-scaling"))
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--dev-per-hop", type=int, default=4)
    parser.add_argument("--test-per-hop", type=int, default=32)
    parser.add_argument("--tokenizer-json", type=Path, default=Path("models/Qwen3-8B/tokenizer.json"))
    args = parser.parse_args()
    manifest = prepare(args.root, args.output_dir, args.seed, args.dev_per_hop, args.test_per_hop, args.tokenizer_json)
    print(json.dumps({"counts": {key: value["count"] for key, value in manifest["selection"].items()},
                      "excluded_source_rows": manifest["exclusions"]["excluded_musique_source_rows"],
                      "corpus": manifest["corpus"], "validation": manifest["validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
