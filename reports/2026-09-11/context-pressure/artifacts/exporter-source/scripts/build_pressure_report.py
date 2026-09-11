#!/usr/bin/env python3
"""Export verified matched-trace context-pressure artifacts without inference.

Generation sources and inputs are read-only. Partial or unverifiable runs retain
raw receipts and diagnostic CSVs, while headline tables remain withheld.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.context_pressure import ARMS, PressureConfig, aggregate, report_tables
from eval.real_experiments import answer_scores, clean_prediction
from scripts.build_adaptive_report import archive, csv_rows, load_json, safe_source, sha, snapshot
from scripts.report_paths import validate_report_output

VERSION = "pressure-artifact-report-v1"


def expected_id(protocol, example_id, arm, config):
    return sha(json.dumps([protocol, example_id, arm, config["evidence_budget"], config["summary_budget"]]).encode())[:24]


def verify_answer(row, question):
    if row.get("question") != question.get("question") or row.get("answers") != question.get("answers"):
        return "question_or_answers_differ_from_pinned_dataset"
    if not isinstance(row.get("prediction"), str):
        return "missing_prediction"
    if not isinstance(row.get("raw_output"), str) or clean_prediction(row["raw_output"]) != row["prediction"]:
        return "raw_output_prediction_mismatch"
    scores = answer_scores(row["prediction"], question["answers"])
    try:
        if any(not math.isclose(row["scores"][key], expected, abs_tol=1e-12, rel_tol=0)
               for key, expected in scores.items()):
            return "scores_differ_from_pinned_answers"
    except (KeyError, TypeError):
        return "invalid_scores"
    return None


def check_record(row):
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        return "missing_record_id"
    if row.get("status") == "error":
        return None
    if row.get("status") != "ok":
        return "unknown_record_status"
    try:
        assert isinstance(row["example_id"], str) and row["arm"] in ARMS
        assert isinstance(row["retained_document_ids"], list)
        assert isinstance(row["model_calls"], list) and row["model_calls"]
        assert isinstance(row["trace_overflow"], bool)
        assert row["scores"]["exact_match"] in (0, 1) and 0 <= row["scores"]["f1"] <= 1
        assert all(isinstance(row["scores"][key], (int, float)) and math.isfinite(row["scores"][key]) for key in ("exact_match", "f1"))
        assert all(isinstance(call, dict) and all(isinstance(call[key], (int, float)) and math.isfinite(call[key]) and call[key] >= 0
                   for key in ("evidence_input_tokens", "total_context_reservation_tokens")) for call in row["model_calls"])
        numeric = [row["replay_cold_ms"], row["source_controller_ms"], row["source_retrieval_ms"],
                   row["retained_evidence_tokens"], row["cumulative_source_tokens"],
                   row["replay_plus_source_controller_and_retrieval_ms"], *row["component_ms"].values()]
        assert all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in numeric)
        row["retained_original_support_id_coverage"]
        row["retained_evidence_token_ids_sha256"]
        row["source_trace_sha256"]
        row["source_result_id"]
    except (KeyError, TypeError, AttributeError, AssertionError):
        return "invalid_success_record_schema"
    return None


def read_archived(run_dir, manifest, name, expected_sha):
    entries = [entry for entry in manifest.get("archives", []) if entry.get("artifact") == name]
    if len(entries) != 1:
        raise ValueError(f"Exactly one archived {name} receipt is required")
    entry = entries[0]
    packed = (run_dir / name).read_bytes()
    if sha(packed) != entry["artifact_sha256"]:
        raise ValueError(f"Packed archive checksum mismatch: {name}")
    raw = gzip.decompress(packed)
    if sha(raw) != entry["source_sha256"] or sha(raw) != expected_sha:
        raise ValueError(f"Uncompressed archive checksum mismatch: {name}")
    return raw


def inspect_run(run_dir, root=ROOT):
    manifest, runner_summary = load_json(run_dir / "manifest.json"), load_json(run_dir / "summary.json")
    identity = manifest.get("identity", {})
    config = identity.get("config", {})
    rows, raw_results, issues = snapshot(run_dir / "results.jsonl")
    verified, source_files = {}, []
    for field, filename, hash_field in (("dataset", "questions.jsonl.gz", "dataset_sha256"),
                                        ("corpus", "corpus.jsonl.gz", "corpus_sha256"),
                                        ("source_results", "source-results.jsonl.gz", "source_results_sha256")):
        try:
            raw = read_archived(run_dir, manifest, filename, identity[hash_field])
            verified[field] = {"raw": raw, "sha256": sha(raw), "source": run_dir / filename,
                               "original_basename": Path(config[field]).name if field in {"dataset", "corpus"} else "results.jsonl"}
        except (KeyError, OSError, ValueError, EOFError) as error:
            issues.append({"kind": "archive_verification_failed", "input": field, "error": str(error)})
    for name, expected in identity.get("source_sha256", {}).items():
        try:
            source = safe_source(root, name)
            raw = source.read_bytes()
            if sha(raw) != expected:
                raise ValueError("Executable source differs from the pressure run identity")
            source_files.append({"path": source, "relative": str(source.relative_to(root.resolve())), "raw": raw})
        except (OSError, ValueError) as error:
            issues.append({"kind": "source_verification_failed", "source": name, "error": str(error)})
    if not identity.get("source_sha256"):
        issues.append({"kind": "missing_executable_source_hashes"})
    questions = {}
    if "dataset" in verified:
        items = [json.loads(line) for line in verified["dataset"]["raw"].splitlines() if line.strip()]
        questions = {row["id"]: row for row in items}
        if len(items) != len(questions) or len(items) != config.get("expected_questions"):
            issues.append({"kind": "question_count_or_uniqueness_mismatch"})
    source_rows = {}
    if "source_results" in verified:
        source_rows = {row["id"]: row for row in (json.loads(line) for line in verified["source_results"]["raw"].splitlines() if line.strip())}
    expected = {expected_id(identity.get("protocol"), key, arm, config) for key in questions for arm in ARMS}
    valid, latest = [], {}
    for number, row in enumerate(rows, 1):
        problem = check_record(row)
        if problem:
            issues.append({"kind": problem, "line": number})
            continue
        valid.append(row)
        latest[row["id"]] = row
        if row["status"] != "ok":
            continue
        if row["example_id"] not in questions:
            issues.append({"kind": "unknown_result_question", "id": row["id"]})
            continue
        question = questions[row["example_id"]]
        problem = verify_answer(row, question)
        if problem:
            issues.append({"kind": problem, "id": row["id"]})
        if row["id"] != expected_id(identity.get("protocol"), row["example_id"], row["arm"], config):
            issues.append({"kind": "result_identity_mismatch", "id": row["id"]})
        source = source_rows.get(row["source_result_id"], {})
        if (source.get("status") != "ok" or source.get("example_id") != row["example_id"]
                or source.get("case") != {"arm": "iterative_text", "max_rounds": config["source_rounds"]}
                or sha(json.dumps(source.get("trace"), sort_keys=True).encode()) != row["source_trace_sha256"]):
            issues.append({"kind": "source_controller_trace_mismatch", "id": row["id"]})
        else:
            if row["cumulative_source_tokens"] != source["counts"]["retrieved_evidence_tokens"]:
                issues.append({"kind": "source_token_count_mismatch", "id": row["id"]})
            for output_key, source_key in (("source_controller_ms", "controller_ms"), ("source_retrieval_ms", "retrieval_ms")):
                if row[output_key] != source["component_ms"].get(source_key, 0):
                    issues.append({"kind": "source_timing_component_mismatch", "id": row["id"]})
        if row["trace_overflow"] != (row["cumulative_source_tokens"] > config["evidence_budget"]):
            issues.append({"kind": "overflow_subset_mismatch", "id": row["id"]})
        if (row["retained_evidence_tokens"] > config["evidence_budget"] or
                any(call["evidence_input_tokens"] > config["evidence_budget"] or
                    call["total_context_reservation_tokens"] > config["max_model_context"] for call in row["model_calls"])):
            issues.append({"kind": "evidence_or_context_budget_violation", "id": row["id"]})
        expected_total = row["replay_cold_ms"] + row["source_controller_ms"] + row["source_retrieval_ms"]
        if not math.isclose(row["replay_plus_source_controller_and_retrieval_ms"], expected_total, rel_tol=1e-12, abs_tol=1e-9):
            issues.append({"kind": "trace_accounting_sum_mismatch", "id": row["id"]})
    references, controller_overflow = [], {}
    for source in source_rows.values():
        if source.get("example_id") not in questions or source.get("status") != "ok":
            continue
        case = source.get("case", {})
        if case == {"arm": "iterative_text", "max_rounds": config.get("source_rounds")}:
            controller_overflow[source["example_id"]] = source["counts"]["retrieved_evidence_tokens"] > config["evidence_budget"]
        elif case != {"arm": "basic_rag", "max_rounds": 1}:
            continue
        problem = verify_answer(source, questions[source["example_id"]])
        if problem:
            issues.append({"kind": "source_reference_" + problem, "id": source["id"]})
        references.append({key: source[key] for key in ("id", "example_id", "case", "scores", "cold_end_to_end_ms", "evidence_ids")})
    for arm in ("basic_rag", "iterative_text"):
        ids = [row["example_id"] for row in references if row["case"]["arm"] == arm]
        if len(ids) != len(questions) or set(ids) != set(questions):
            issues.append({"kind": "incomplete_source_references", "arm": arm})
    for row in references:
        row["controller_trace_overflow"] = controller_overflow.get(row["example_id"], False)
    variants = defaultdict(list)
    for row in latest.values():
        if row["status"] == "ok" and row["arm"] != "rolling_summary":
            variants[row["example_id"]].append(row)
    for key, group in variants.items():
        if len({json.dumps([row["retained_document_ids"], row["retained_evidence_token_ids_sha256"]]) for row in group}) != 1:
            issues.append({"kind": "text_kv_retained_evidence_mismatch", "example_id": key})
    ok = {key for key, row in latest.items() if row["status"] == "ok"}
    if set(latest) - expected:
        issues.append({"kind": "unexpected_result_ids", "count": len(set(latest) - expected)})
    unresolved = sum(row["status"] != "ok" for row in latest.values())
    complete = bool(expected) and ok == expected and not unresolved and not issues and runner_summary.get("status") == "complete" and runner_summary.get("stop_reason") == "complete"
    return {"status": "complete" if complete else "INCOMPLETE_OR_UNVERIFIED", "manifest": manifest,
            "runner_summary": runner_summary, "records": valid, "raw_results": raw_results, "verified_inputs": verified,
            "verified_sources": source_files, "references": references, "issues": issues,
            "planned_runs": len(expected), "successful_runs": len(ok), "missing_runs": len(expected - ok),
            "unresolved_errors": unresolved, "duplicate_record_ids": len(valid) - len(latest)}


def build(run_dir, output, root=ROOT):
    validate_report_output(output, run_dir)
    output.mkdir(parents=True, exist_ok=True)
    inspected = inspect_run(run_dir, root)
    config_values = dict(inspected["manifest"].get("identity", {}).get("config", {}))
    for key in ("dataset", "corpus", "source_dir"):
        config_values.setdefault(key, "")
    config = PressureConfig(**config_values, output_dir=str(run_dir))
    summary = aggregate(inspected["records"], inspected["references"], config,
                        inspected["runner_summary"].get("stop_reason", "missing"), inspected["planned_runs"])
    summary.update(generation_status=summary["status"], status=inspected["status"], report_version=VERSION,
                   report_status=inspected["status"], verification_issues=inspected["issues"],
                   completion={key: inspected[key] for key in ("planned_runs", "successful_runs", "missing_runs", "unresolved_errors", "duplicate_record_ids")})
    receipts = []
    for name in ("results.jsonl", "manifest.json", "summary.json", "checkpoint.json", "resources.jsonl", "guard-status.json", "process.log"):
        path = run_dir / name
        if path.exists():
            raw = inspected["raw_results"] if name == "results.jsonl" else path.read_bytes()
            receipts.append(archive(output, Path("run") / (name + ".gz"), raw, path, compress=True))
    for field, item in inspected["verified_inputs"].items():
        extras = {"original_basename": item["original_basename"]} if field in {"dataset", "corpus"} else {}
        receipts.append(archive(output, Path("inputs") / (field + ".jsonl.gz"), item["raw"], item["source"], compress=True,
                                verified_against_generation_manifest=True, **extras))
    for source in inspected["verified_sources"]:
        receipts.append(archive(output, Path("source") / source["relative"], source["raw"], source["path"], verified_against_generation_manifest=True))
    for relative in ("data/adaptive_musique.manifest.json", "docs/adaptive-data.md", "docs/adaptive-protocol.md"):
        path = root / relative
        if path.exists():
            receipts.append(archive(output, Path("provenance") / path.name, path.read_bytes(), path))
    for relative in ("scripts/build_pressure_report.py", "scripts/build_adaptive_report.py", "scripts/report_paths.py"):
        path = ROOT / relative
        receipts.append(archive(output, Path("exporter-source") / relative, path.read_bytes(), path))
    license_text = ("# MuSiQue data license and replay\n\nTrivedi et al. (2022), MuSiQue: https://github.com/StonyBrookNLP/musique\n\n"
                    "Data are CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . This applies to benchmark data separately from project code. "
                    "Modifications include exact title/text deduplication, content-hash IDs and local disjoint question subsets. "
                    "Decompress verified dataset/corpus input archives and restore original_basename under data/; receipts preserve exact uncompressed hashes. "
                    "No model weights are included. Source adaptive traces and measured reference costs are archived separately from pressure replay results.\n")
    (output / "DATA_LICENSE.md").write_text(license_text)
    csv_rows(output / "metrics.csv", summary["metrics"])
    csv_rows(output / "pairs.csv", summary["paired"])
    csv_rows(output / "references.csv", summary["reference_metrics"])
    complete = inspected["status"] == "complete"
    text = report_tables(summary) if complete else ("# Context pressure — incomplete or unverified\n\nHeadline tables are withheld. Partial metrics and exact raw receipts remain available.\n\n" + json.dumps(inspected["issues"], indent=2))
    (output / "tables.md").write_text(text)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "artifact-manifest.json").write_text(json.dumps({"report_version": VERSION, "report_source_sha256": sha(Path(__file__).read_bytes()), "artifacts": receipts}, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/context-pressure"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/2026-09-11/context-pressure"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = build(args.run_dir, args.output_dir)
    print(json.dumps({"status": result["report_status"], "output_dir": str(args.output_dir), "completion": result["completion"], "issues": result["verification_issues"]}))
    return 2 if args.require_complete and result["report_status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
