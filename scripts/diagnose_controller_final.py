#!/usr/bin/env python3
"""Posthoc paired final-synthesis diagnostic on verified saved pilot evidence.

No retrieval or controller is rerun. The ablation removes only the system's
SEARCH/ANSWER instruction sentence. Every original greedy final must first
reproduce its saved token IDs. These are synthesis diagnostics, not measured
end-to-end pipeline improvements or independent test results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.adaptive_experiments import PROLOGUE, SYSTEM, append_record, document_text, final_text, write_json
from eval.real_experiments import answer_scores, clean_prediction
from scripts.build_controller_scaling_report import inspect_matrix, read_matrix
from scripts.build_adaptive_report import safe_source
from scripts.report_paths import validate_report_output

VERSION = "controller-final-system-ablation-v1"
SEARCH_RULE = "A search decision must be exactly ANSWER or SEARCH: followed by one short query. "
LIMITATION = ("Posthoc development diagnostic: fixed previously discovered evidence, no retrieval/controller rerun. "
              "Only one system instruction sentence changes. Timings cover final synthesis, not the full pipeline; "
              "original controller thoughts and answer payloads are not supplied to the model.")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ids_sha(ids):
    return sha(json.dumps(ids).encode())


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def final_only_system():
    if SYSTEM.count(SEARCH_RULE) != 1:
        raise ValueError("The frozen original system must contain the exact search-rule sentence once")
    return SYSTEM.replace(SEARCH_RULE, "")


def reconstruct(backend, row, corpus, config):
    """Rebuild the original segmented token stream, checking every saved round."""
    tokens, evidence, seen, truncated = backend.encode(PROLOGUE), [], [], []
    for event in row["trace"]:
        if event["event"] != "retrieval":
            continue
        for identifier in event["new_document_ids"]:
            if identifier in seen or identifier not in corpus:
                raise ValueError("Saved evidence repeats an ID or references a missing corpus document")
            raw = backend.encode(document_text(corpus[identifier]))
            fragment = raw[:config["max_document_tokens"]]
            if not fragment:
                raise ValueError("An admitted document reconstructed to an empty token fragment")
            if len(fragment) < len(raw):
                truncated.append(identifier)
            tokens.extend(fragment)
            evidence.extend(fragment)
            seen.append(identifier)
        if (event["accumulated_document_ids"] != seen or event["evidence_tokens"] != len(evidence) or
                event["evidence_token_ids_sha256"] != ids_sha(tokens)):
            raise ValueError("Reconstructed evidence differs from the saved round hash, count, or order")
    if (seen != row["evidence_ids"] or len(seen) != row["counts"]["retrieved_documents"] or
            len(evidence) != row["counts"]["retrieved_evidence_tokens"] or
            truncated != row["delivered_truncated_document_ids"]):
        raise ValueError("Reconstructed final evidence differs from the saved document/token/truncation ledger")
    suffix = backend.encode(final_text(row["question"]))
    original = tokens + suffix
    ablated_prologue = PROLOGUE.replace(SYSTEM, final_only_system(), 1)
    ablated = backend.encode(ablated_prologue) + evidence + suffix
    if len(original) != row["counts"]["final_prefill_tokens"]:
        raise ValueError("Reconstructed original final input length differs from the saved full-text prefill count")
    if max(len(original), len(ablated)) + config["max_answer_tokens"] > config["max_model_context_tokens"]:
        raise ValueError("Diagnostic final reservation exceeds the original context cap")
    return {"original": original, "final_only_system": ablated}, {
        "evidence_ids": seen, "evidence_tokens": len(evidence),
        "evidence_only_token_ids_sha256": ids_sha(evidence),
        "original_evidence_prefix_sha256": ids_sha(tokens),
        "final_suffix_token_ids_sha256": ids_sha(suffix)}


def verify_checkpoint(job, root=ROOT):
    """Verify every checkpoint file against its local pinned download receipt."""
    directory = safe_source(root, job["model_path"])
    receipt = directory / "download-manifest.json"
    if not receipt.exists():
        receipt = directory.parent / "model-manifest.json"
    manifest = json.loads(receipt.read_text())
    if manifest.get("revision") != job["model_revision"] or manifest.get("weights_dtype") != "bfloat16":
        raise ValueError("Model download receipt does not match the pinned revision and BF16 precision")
    verified, weights = [], set()
    for entry in manifest["files"]:
        name = entry.get("path", entry.get("name"))
        path = directory / name
        if path.resolve().parent != directory.resolve() or path.is_symlink():
            raise ValueError("Checkpoint receipt contains an unsafe path or symlink")
        if path.stat().st_size != entry.get("size_bytes", entry.get("bytes")):
            raise ValueError(f"Checkpoint file size changed: {name}")
        digest = file_sha(path)
        expected = entry.get("sha256")
        if expected and digest != expected:
            raise ValueError(f"Checkpoint SHA-256 changed: {name}")
        if name.endswith(".safetensors"):
            if not expected:
                raise ValueError("Every model-weight shard needs a pinned SHA-256")
            weights.add(name)
        verified.append({"path": name, "bytes": path.stat().st_size, "sha256": digest,
                         "matched_download_sha256": bool(expected)})
    if not weights or weights != {path.name for path in directory.glob("*.safetensors")}:
        raise ValueError("Model weight files differ from the pinned shard inventory")
    return {"receipt": str(receipt.relative_to(root)), "receipt_sha256": file_sha(receipt),
            "model_revision": job["model_revision"], "files": verified,
            "note": "Files lacking historical download hashes have their current SHA recorded; original-prompt token replay is mandatory."}


def claim_output(output, source_directories):
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Diagnostic output must not be a symlink")
    validate_report_output(output, *source_directories)
    # A guard may already have created its own resource receipt directory. Use
    # a child output directory for this script, which must itself be absent.
    if output.exists():
        raise ValueError("Use a fresh diagnostic output directory; existing outputs are never resumed or overwritten")
    output.mkdir(parents=True, exist_ok=False)


def generate_final(backend, tokens, budget):
    backend.clear_cache()
    backend.synchronize()
    started = time.perf_counter()
    output = backend.decode(None, tokens, max_tokens=budget)
    backend.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    raw = backend.decode_tokens(output)
    return {"prompt_token_ids": tokens, "prompt_token_ids_sha256": ids_sha(tokens),
            "output_token_ids": output, "raw_output": raw, "prediction": clean_prediction(raw),
            "final_synthesis_ms": elapsed, "decode": dict(backend.last_decode_stats),
            "work": {"prefill_tokens": len(tokens), "generated_tokens": len(output),
                     "sampled_tokens_including_eos": backend.last_decode_stats["sampled_tokens"],
                     "context_reservation_tokens": len(tokens) + budget}, "memory": backend.memory_stats()}


def run_diagnostic(matrix_path, model_label, output, memory_limit_gib, root=ROOT):
    matrix_path = Path(matrix_path).resolve()
    matrix = read_matrix(matrix_path, root)
    if (matrix["cohort"] != "development" or matrix["shared_config"]["limit"] != 12 or
            matrix["shared_config"]["arms"] != ["iterative_text"]):
        raise ValueError("This diagnostic is restricted to the complete 12-question iterative-text development pilot")
    inspections, _productivity, issues = inspect_matrix(matrix, root)
    if issues:
        raise ValueError("The entire four-job pilot must be complete and source/input verified: " + json.dumps(issues))
    jobs = [job for job in matrix["jobs"] if job["model_label"] == model_label]
    if len(jobs) != 2:
        raise ValueError("Select one of the two pinned model labels in the matrix")
    first = jobs[0]
    selected = [(job, row) for job in jobs for row in inspections[job["id"]]["records"]]
    if len(selected) != 24 or any(row["status"] != "ok" for _, row in selected):
        raise ValueError("Expected exactly 24 successful unique source conditions for the selected model")
    source_directories = [safe_source(root, job["run_dir"]) for job in matrix["jobs"]]
    source_directories += [safe_source(root, matrix[key]).parent for key in ("dataset", "corpus")]
    source_directories += [safe_source(root, first["model_path"])]
    claim_output(output, source_directories)
    output = Path(output)
    source_names = ("scripts/diagnose_controller_final.py", "scripts/build_controller_scaling_report.py",
                    "scripts/build_adaptive_report.py", "scripts/report_paths.py", "engine/backends_mlx.py",
                    "eval/adaptive_experiments.py", "eval/real_experiments.py", "eval/retrieval.py")
    manifest = {"schema_version": VERSION, "scope": LIMITATION, "model_label": model_label,
        "matrix": str(matrix_path), "matrix_sha256": file_sha(matrix_path),
        "questions_sha256": matrix["questions_sha256"], "corpus_sha256": matrix["corpus_sha256"],
        "source_sha256": {name: file_sha(root / name) for name in source_names},
        "source_runs": {job["id"]: {name: file_sha(safe_source(root, job["run_dir"]) / name)
                                   for name in ("manifest.json", "results.jsonl", "summary.json")}
                        for job in matrix["jobs"]},
        "original_system": SYSTEM, "final_only_system": final_only_system(), "removed_sentence": SEARCH_RULE,
        "system_sha256": {"original": sha(SYSTEM.encode()), "final_only_system": sha(final_only_system().encode())},
        "checkpoint": verify_checkpoint(first, root), "memory_limit_gib": memory_limit_gib,
        "planned_source_conditions": 24, "planned_final_generations": 48,
        "order": "matrix job order, source question order; original replay before its system ablation",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(output / "manifest.json", manifest)
    from engine.backends_mlx import MLXBackend
    started = time.perf_counter()
    backend = MLXBackend(safe_source(root, first["model_path"]), revision=first["model_revision"],
        max_context=matrix["shared_config"]["max_model_context_tokens"], memory_limit_gb=memory_limit_gib)
    manifest["model_load_ms"] = (time.perf_counter() - started) * 1000
    manifest["backend"] = backend.metadata()
    for job in jobs:
        expected = inspections[job["id"]]["manifest"]["identity"]["backend"]
        if backend.metadata()["backend_identity"] != expected["backend_identity"]:
            raise ValueError("Loaded backend identity differs from the source pilot")
    write_json(output / "manifest.json", manifest)
    corpus = {doc["id"]: doc for doc in (json.loads(line) for line in
        inspections[first["id"]]["verified_inputs"]["corpus"]["raw"].splitlines() if line.strip())}
    rows, generations, status = [], [], "complete"
    started = time.perf_counter()
    try:
        for job, source in selected:
            prompts, evidence = reconstruct(backend, source, corpus, matrix["shared_config"])
            pair = {"schema_version": VERSION, "job_id": job["id"], "source_result_id": source["id"],
                "example_id": source["example_id"], "model_label": model_label,
                "controller_reasoning_budget": job["max_reasoning_tokens"], "question": source["question"],
                "answers_for_scoring_only": source["answers"], "evidence": evidence, "conditions": {}}
            for condition in ("original", "final_only_system"):
                result = generate_final(backend, prompts[condition], matrix["shared_config"]["max_answer_tokens"])
                generated = {"job_id": job["id"], "source_result_id": source["id"],
                             "example_id": source["example_id"], "condition": condition, **result}
                append_record(output / "generations.jsonl", generated)
                generations.append(generated)
                if condition == "original" and (result["output_token_ids"] != source["output_token_ids"] or
                                                  result["raw_output"] != source["raw_output"]):
                    write_json(output / "replay-failure.json", {**pair, "actual": result,
                        "expected_output_token_ids": source["output_token_ids"]})
                    raise ValueError("Original final token replay diverged; no paired ablation accepted")
                result["scores"] = answer_scores(result["prediction"], source["answers"])
                pair["conditions"][condition] = result
            pair["original_token_replay_exact"] = True
            pair["score_difference"] = {key: pair["conditions"]["final_only_system"]["scores"][key] -
                pair["conditions"]["original"]["scores"][key] for key in ("exact_match", "f1")}
            append_record(output / "results.jsonl", pair)
            rows.append(pair)
            print(json.dumps({"completed_pairs": len(rows), "planned_pairs": 24, "job": job["id"],
                              "example_id": source["example_id"], "score_difference": pair["score_difference"]}), flush=True)
    except BaseException:
        status = "failed_or_interrupted"
        raise
    finally:
        groups = defaultdict(list)
        for row in rows:
            groups[row["job_id"]].append(row)
        metrics = [{"job_id": job_id, "n_paired": len(values), "condition": condition,
                    "exact_match": statistics.mean(row["conditions"][condition]["scores"]["exact_match"] for row in values),
                    "f1": statistics.mean(row["conditions"][condition]["scores"]["f1"] for row in values),
                    "median_final_synthesis_ms": statistics.median(row["conditions"][condition]["final_synthesis_ms"] for row in values)}
                   for job_id, values in groups.items() for condition in ("original", "final_only_system")]
        write_json(output / "summary.json", {"schema_version": VERSION, "scope": LIMITATION,
            "status": status, "successful_pairs": len(rows), "planned_pairs": 24,
            "verified_original_replays": len(rows), "final_generations_in_successful_pairs": len(rows) * 2,
            "completed_final_generations": len(generations),
            "elapsed_seconds_excluding_load": time.perf_counter() - started, "metrics": metrics})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-limit-gib", type=float, default=32)
    args = parser.parse_args()
    return run_diagnostic(args.matrix, args.model_label, args.output_dir, args.memory_limit_gib)


if __name__ == "__main__":
    raise SystemExit(main())
