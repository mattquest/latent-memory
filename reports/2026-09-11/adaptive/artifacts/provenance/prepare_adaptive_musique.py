#!/usr/bin/env python3
"""Build disjoint global-corpus MuSiQue retrieval questions from a local archive.

CPU only; no network, model weights, generated answers, or run-result reads.
Corpus records expose only content hashes, titles, and text. Gold support and
source-occurrence mappings are separate scoring/provenance artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.datasets import sha256_file, write_jsonl

ARCHIVE_SHA256 = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
MEMBER = "data/musique_ans_v1.0_dev.jsonl"
SCRIPT_VERSION = 1


def paragraph_id(title, text):
    raw = json.dumps([title, text], ensure_ascii=False, separators=(",", ":")).encode()
    return "musique-paragraph-" + hashlib.sha256(raw).hexdigest()


def structural_hops(row):
    hops = len(row["question_decomposition"])
    match = re.match(r"([234])hop", row["id"])
    if match is None or int(match.group(1)) != hops:
        raise ValueError(f"Unrecognized/inconsistent structural hop count: {row['id']}")
    return hops


def build_corpus(rows):
    documents, occurrences, source_map = {}, defaultdict(list), {}
    for row in rows:
        for paragraph in row["paragraphs"]:
            title, text = paragraph["title"], paragraph["paragraph_text"]
            doc_id = paragraph_id(title, text)
            document = {"id": doc_id, "title": title, "text": text}
            if doc_id in documents and documents[doc_id] != document:
                raise ValueError("Paragraph content hash collision")
            documents[doc_id] = document
            source_key = (row["id"], paragraph["idx"])
            if source_key in source_map:
                raise ValueError(f"Duplicate paragraph source key: {source_key}")
            source_map[source_key] = doc_id
            occurrences[doc_id].append({"question_id": row["id"], "paragraph_idx": paragraph["idx"]})
    corpus = [documents[key] for key in sorted(documents)]
    provenance = [{"id": key, "source_occurrences": sorted(occurrences[key], key=lambda value: (value["question_id"], value["paragraph_idx"]))}
                  for key in sorted(documents)]
    return corpus, provenance, source_map


def select_questions(rows, excluded_rows, seed, test_per_hop=32, dev_per_hop=2):
    excluded_ids = {row["id"] for row in excluded_rows}
    excluded_questions = {row["question"] for row in excluded_rows}
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source question IDs")
    by_hop = defaultdict(list)
    for row in rows:
        if row["id"] not in excluded_ids and row["question"] not in excluded_questions:
            by_hop[structural_hops(row)].append(row)
    chosen, used_questions = {"dev": [], "test": []}, set()
    for hops in (2, 3, 4):
        ordered = sorted(by_hop[hops], key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        distinct = []
        for row in ordered:
            if row["question"] in used_questions:
                continue
            distinct.append(row)
            used_questions.add(row["question"])
            if len(distinct) == test_per_hop + dev_per_hop:
                break
        if len(distinct) < test_per_hop + dev_per_hop:
            raise ValueError(f"Insufficient eligible {hops}-hop questions")
        chosen["dev"].extend(distinct[:dev_per_hop])
        chosen["test"].extend(distinct[dev_per_hop:])
    return chosen, {str(hop): len(values) for hop, values in sorted(by_hop.items())}


def normalize_questions(rows, split, source_map, corpus_receipt, seed):
    normalized = []
    for row in rows:
        hop_count = structural_hops(row)
        annotated = {p["idx"] for p in row["paragraphs"] if p["is_supporting"]}
        decomposed = {step["paragraph_support_idx"] for step in row["question_decomposition"]}
        if annotated != decomposed:
            raise ValueError(f"Support annotations disagree: {row['id']}")
        supports = sorted({source_map[(row["id"], idx)] for idx in annotated})
        answers = list(dict.fromkeys([row["answer"], *row.get("answer_aliases", [])]))
        normalized.append({"schema_version": 1, "id": row["id"], "dataset": "musique_global",
                           "split": split, "category": f"{hop_count}hop", "question": row["question"],
                           "answers": answers, "evidence": [], "supporting_context_ids": supports,
                           "corpus_path": Path(corpus_receipt["path"]).name,
                           "metadata": {"source_id": row["id"], "source_structural_category": row["id"].split("__", 1)[0],
                                        "hop_count": hop_count, "seed": seed,
                                        "protocol": "global_answerable_dev_corpus_retrieval",
                                        "corpus_path": Path(corpus_receipt["path"]).name,
                                        "corpus_sha256": corpus_receipt["sha256"],
                                        "license": "CC-BY-4.0", "source": "https://github.com/StonyBrookNLP/musique"}})
    return normalized


def prepare(archive, excluded, output_dir, seed=20260913, test_per_hop=32, dev_per_hop=2, tokenizer_json=None):
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("Official MuSiQue v1.0 archive checksum mismatch")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.getinfo(MEMBER).file_size > 128 * 1024**2:
            raise ValueError("Development member exceeds bounded preparation size")
        raw = zipped.read(MEMBER)
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not all(row.get("answerable") is True for row in rows):
        raise ValueError("Expected answerable development rows only")
    excluded_rows = [json.loads(line) for line in excluded.read_text().splitlines() if line.strip()]
    corpus, provenance, source_map = build_corpus(rows)
    selected, eligible_counts = select_questions(rows, excluded_rows, seed, test_per_hop, dev_per_hop)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    corpus_receipt = write_jsonl(output_dir / "adaptive_musique_corpus.jsonl", corpus)
    outputs.append(corpus_receipt)
    outputs.append(write_jsonl(output_dir / "adaptive_musique_corpus_provenance.jsonl", provenance))
    questions = {}
    for split in ("dev", "test"):
        questions[split] = normalize_questions(selected[split], split, source_map, corpus_receipt, seed)
        outputs.append(write_jsonl(output_dir / f"adaptive_musique_{split}.jsonl", questions[split]))
    all_selected = questions["dev"] + questions["test"]
    selected_ids = {row["id"] for row in all_selected}
    excluded_ids = {row["id"] for row in excluded_rows}
    corpus_ids = {row["id"] for row in corpus}
    assert not selected_ids & excluded_ids
    assert len(selected_ids) == len(all_selected)
    assert len({row["question"] for row in all_selected}) == len(all_selected)
    assert all(set(row) == {"id", "title", "text"} for row in corpus)
    assert all(set(row["supporting_context_ids"]) <= corpus_ids for row in all_selected)
    title_counts = Counter(row["title"] for row in corpus)
    occurrence_counts = {row["id"]: len(row["source_occurrences"]) for row in provenance}
    selected_support_ids = {doc_id for row in all_selected for doc_id in row["supporting_context_ids"]}
    token_stats = {"characters_title_plus_newline_plus_text": sum(len(row["title"] + "\n" + row["text"]) for row in corpus),
                   "whitespace_words_title_plus_text": sum(len((row["title"] + "\n" + row["text"]).split()) for row in corpus)}
    if tokenizer_json:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        token_stats.update(tokenizer_json_sha256=sha256_file(tokenizer_json),
                           tokenizer_path=str(tokenizer_json), add_special_tokens=False,
                           exact_tokens_title_plus_newline_plus_text=sum(len(tokenizer.encode(row["title"] + "\n" + row["text"], add_special_tokens=False).ids) for row in corpus))
    manifest = {"schema_version": 1, "generator_version": SCRIPT_VERSION, "generator_sha256": sha256_file(Path(__file__)),
                "seed": seed, "selection_rule": "Per structural hop count: SHA-256(seed:source_id), first dev_per_hop for dev then test_per_hop for test; no model outcomes. Exclude previous IDs and exact question texts; selected question texts are distinct.",
                "test_per_hop": test_per_hop, "dev_per_hop": dev_per_hop,
                "source": {"archive": str(archive), "archive_sha256": ARCHIVE_SHA256, "member": MEMBER,
                           "member_sha256": hashlib.sha256(raw).hexdigest(), "source_rows": len(rows),
                           "source_hop_counts": dict(sorted(Counter(str(structural_hops(row)) for row in rows).items())),
                           "download_script_revision": "922ac98f19a201998dbdae6d7f2887a5258dbdeb",
                           "url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view"},
                "excluded": {"path": str(excluded), "sha256": sha256_file(excluded), "question_ids": sorted(excluded_ids),
                             "count": len(excluded_ids)},
                "eligible_by_hop": eligible_counts, "corpus_unique_paragraphs": len(corpus),
                "corpus_source_occurrences": len(source_map), "token_counts": token_stats,
                "corpus_statistics": {"exact_duplicate_occurrences_removed": len(source_map) - len(corpus),
                                      "unique_titles": len(title_counts),
                                      "titles_with_multiple_distinct_texts": sum(count > 1 for count in title_counts.values()),
                                      "selected_unique_support_paragraphs": len(selected_support_ids),
                                      "selected_support_paragraphs_with_duplicate_source_occurrences": sum(occurrence_counts[doc_id] > 1 for doc_id in selected_support_ids)},
                "selection": {split: {"count": len(values), "question_ids": [row["id"] for row in values],
                                       "hop_counts": dict(sorted(Counter(str(row["metadata"]["hop_count"]) for row in values).items()))}
                              for split, values in questions.items()},
                "validation": {"support_mapping_complete": True, "support_annotations_agree": True,
                               "dev_test_ids_and_question_text_disjoint": True, "prior_main_ids_and_question_text_excluded": True,
                               "corpus_fields_only_id_title_text": True},
                "license": {"name": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/",
                            "attribution": "Trivedi et al. (2022), MuSiQue", "homepage": "https://github.com/StonyBrookNLP/musique",
                            "modifications": "Global corpus from all supplied answerable development paragraphs, exact title/text deduplication, content-hash IDs, and disjoint balanced question subsets. Not an official full-Wikipedia benchmark."},
                "limitations": ["Development and test here are local subsets of the public answerable development split.",
                                "Questions can share component facts, source articles, or public pretraining exposure despite disjoint IDs/text.",
                                "Distinct text variants under the same title remain separate documents; only exact title-plus-text duplicates are removed.",
                                "Corpus includes all supplied dev contexts, including support content, but no support labels, answers, question decomposition, or source question IDs in retrievable records.",
                                "Support IDs and hop counts are scoring/provenance only; never show them to a retriever or query controller."],
                "outputs": outputs}
    (output_dir / "adaptive_musique.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/musique.zip"))
    parser.add_argument("--exclude", type=Path, default=Path("data/musique_dev_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--test-per-hop", type=int, default=32)
    parser.add_argument("--dev-per-hop", type=int, default=2)
    parser.add_argument("--tokenizer-json", type=Path)
    args = parser.parse_args()
    if args.test_per_hop < 1 or args.dev_per_hop < 1:
        parser.error("Both per-hop sample counts must be positive")
    manifest = prepare(args.archive, args.exclude, args.output_dir, args.seed, args.test_per_hop,
                       args.dev_per_hop, args.tokenizer_json)
    print(json.dumps({"corpus_unique_paragraphs": manifest["corpus_unique_paragraphs"],
                      "source_rows": manifest["source"]["source_rows"],
                      "selection_counts": {split: row["hop_counts"] for split, row in manifest["selection"].items()},
                      "token_counts": manifest["token_counts"], "validation": manifest["validation"]}))


if __name__ == "__main__":
    main()
