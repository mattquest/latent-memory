#!/usr/bin/env python3
"""CPU-only, source-verified export of adaptive experiment artifacts.

No inference is invoked. A complete report requires every expected question/arm
ID, no unresolved errors or torn records, a completed runner summary, and exact
input/source checksums. Incomplete ledgers remain archived but do not produce
headline accuracy tables or figures.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from eval.adaptive_experiments import aggregate, report_tables
from eval.real_experiments import answer_scores
from scripts.report_paths import validate_report_output


VERSION = "adaptive-artifact-report-v1"
LABELS = {"basic_rag": "Basic BM25", "expanded_rag": "Query expansion",
          "reranked_rag": "BM25 + Qwen3 reranker", "iterative_text": "Iterative full text",
          "incremental_kv_bf16": "Exact incremental KV", "relay_kv_bf16": "Independent KV · BF16",
          "relay_kv_int8": "Independent KV · int8"}
COLORS = {"basic_rag": "#374151", "expanded_rag": "#d97706", "reranked_rag": "#be185d",
          "iterative_text": "#2563eb", "incremental_kv_bf16": "#059669",
          "relay_kv_bf16": "#7c3aed", "relay_kv_int8": "#b45309"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def expected_case_id(protocol, example_id, case, max_reasoning_tokens=0,
                     controller_temperature=0.0, controller_top_p=1.0, controller_top_k=0):
    """Mirror the manifest-pinned ID recipe, preserving historical no-thinking IDs."""
    payload = [protocol, example_id, case]
    if protocol == "adaptive-global-bm25-controller-v2":
        payload.append({"max_reasoning_tokens": max_reasoning_tokens,
                        "controller_temperature": float(controller_temperature),
                        "controller_top_p": float(controller_top_p), "controller_top_k": int(controller_top_k)})
    elif max_reasoning_tokens > 0:
        payload.append({"max_reasoning_tokens": max_reasoning_tokens})
    return sha(json.dumps(payload, sort_keys=True).encode())[:24]


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot(path):
    raw = path.read_bytes() if path.exists() else b""
    rows, issues = [], []
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            issues.append({"kind": "partial_trailing_record" if index == len(lines) - 1 and not line.endswith(b"\n")
                           else "invalid_json_record", "line": index + 1})
    return rows, raw, issues


def safe_source(root, name):
    path = Path(name)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact source is outside the repository: {name}")
    return path


def record_issue(row):
    """Reject malformed metric rows before they enter the shared aggregator."""
    if not isinstance(row, dict) or not isinstance(row.get("id"), str):
        return "missing_record_id"
    if row.get("status") == "error":
        return None
    if row.get("status") != "ok":
        return "unknown_record_status"
    try:
        for key in ("example_id", "case", "scores", "question", "answers", "prediction", "component_ms", "counts", "executed_retrieval_rounds",
                    "invalid_actions", "generation_stop", "support_annotation_coverage", "stop_reason"):
            row[key]
        assert isinstance(row["example_id"], str)
        assert isinstance(row["question"], str) and isinstance(row["prediction"], str) and isinstance(row["answers"], list)
        assert isinstance(row["case"]["arm"], str) and isinstance(row["case"]["max_rounds"], int)
        values = [row["cold_end_to_end_ms"], *row["component_ms"].values(), *row["counts"].values(),
                  row["scores"]["exact_match"], row["scores"]["f1"]]
        assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        assert row["scores"]["exact_match"] in (0, 1) and 0 <= row["scores"]["f1"] <= 1
        assert row["cold_end_to_end_ms"] >= 0
    except (KeyError, TypeError, AttributeError, AssertionError):
        return "invalid_success_record_schema"
    return None


def inspect_run(run_dir, root=REPOSITORY):
    manifest = load_json(run_dir / "manifest.json")
    runner_summary = load_json(run_dir / "summary.json")
    rows, raw, issues = snapshot(run_dir / "results.jsonl")
    identity = manifest.get("identity", {})
    config = identity.get("config", {})
    latest, valid_rows = {}, []
    for index, row in enumerate(rows):
        invalid = record_issue(row)
        if invalid:
            issues.append({"kind": invalid, "line": index + 1})
        else:
            valid_rows.append(row)
            latest[row["id"]] = row
    verified = {}
    for field, digest_field in (("dataset", "questions_sha256"), ("corpus", "corpus_sha256")):
        try:
            source = safe_source(root, config[field])
            content = source.read_bytes()
            if sha(content) != identity.get(digest_field):
                raise ValueError("SHA-256 differs from the executed manifest")
            verified[field] = {"path": source, "raw": content, "sha256": sha(content)}
        except (KeyError, OSError, ValueError) as exc:
            issues.append({"kind": "input_verification_failed", "input": field, "error": str(exc)})
    source_files = []
    for name, expected in identity.get("source_sha256", {}).items():
        try:
            source = safe_source(root, name)
            content = source.read_bytes()
            if sha(content) != expected:
                raise ValueError("Executable source differs from the executed manifest")
            source_files.append({"relative": str(source.relative_to(root.resolve())), "path": source, "raw": content, "sha256": expected})
        except (OSError, ValueError) as exc:
            issues.append({"kind": "source_verification_failed", "source": name, "error": str(exc)})
    if not identity.get("source_sha256"):
        issues.append({"kind": "missing_executable_source_hashes"})
    expected_ids, pinned_questions = set(), {}
    if "dataset" in verified:
        questions = [json.loads(line) for line in verified["dataset"]["raw"].splitlines() if line.strip()]
        questions = questions[:config.get("limit", len(questions))]
        pinned_questions = {question["id"]: question for question in questions}
        for question in questions:
            for arm in config.get("arms", []):
                rounds = (1,) if arm in {"basic_rag", "expanded_rag", "reranked_rag"} else config.get("rounds", [5])
                for budget in rounds:
                    case = {"arm": arm, "max_rounds": budget}
                    expected_ids.add(expected_case_id(identity.get("protocol"), question["id"], case,
                        config.get("max_reasoning_tokens", 0), config.get("controller_temperature", 0.0),
                        config.get("controller_top_p", 1.0), config.get("controller_top_k", 0)))
    ok_ids = {key for key, row in latest.items() if row.get("status") == "ok"}
    for row in latest.values():
        if row.get("status") == "ok":
            row_id = expected_case_id(identity.get("protocol"), row["example_id"], row["case"],
                config.get("max_reasoning_tokens", 0), config.get("controller_temperature", 0.0),
                config.get("controller_top_p", 1.0), config.get("controller_top_k", 0))
            if row_id != row["id"]:
                issues.append({"kind": "result_identity_mismatch", "id": row["id"]})
            question = pinned_questions.get(row["example_id"])
            if question is not None:
                if row["question"] != question.get("question") or row["answers"] != question.get("answers"):
                    issues.append({"kind": "result_question_or_answers_mismatch", "id": row["id"]})
                expected_scores = answer_scores(row["prediction"], question["answers"])
                if any(abs(row["scores"][key] - value) > 1e-12 for key, value in expected_scores.items()):
                    issues.append({"kind": "result_score_mismatch", "id": row["id"], "recomputed_scores": expected_scores})
    unexpected = set(latest) - expected_ids
    if unexpected:
        issues.append({"kind": "unexpected_result_ids", "count": len(unexpected)})
    unresolved = sum(row.get("status") != "ok" for row in latest.values())
    complete = (bool(expected_ids) and ok_ids == expected_ids and not unresolved and not issues
                and runner_summary.get("stop_reason") == "complete")
    return {"manifest": manifest, "runner_summary": runner_summary, "records": valid_rows,
            "raw_results": raw, "verified_inputs": verified, "verified_sources": source_files,
            "status": "complete" if complete else "INCOMPLETE_OR_UNVERIFIED",
            "planned_unique_runs": len(expected_ids), "successful_unique_runs": len(ok_ids),
            "missing_unique_runs": len(expected_ids - ok_ids), "unresolved_errors": unresolved,
            "historical_error_records": sum(row.get("status") != "ok" for row in valid_rows),
            "duplicate_record_ids": len(valid_rows) - len(latest), "issues": issues}


def archive(output, relative, raw, source, compress=False, **extra):
    path = output / "artifacts" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = gzip.compress(raw, mtime=0) if compress else raw
    path.write_bytes(content)
    return {"artifact": str(path.relative_to(output)), "source": str(source),
            "source_sha256": sha(raw), "source_bytes": len(raw), "artifact_sha256": sha(content),
            "artifact_bytes": len(content), "compression": "gzip" if compress else None, **extra}


def csv_rows(path, rows):
    flattened = []
    for row in rows:
        result = {}
        for key, value in row.items():
            if isinstance(value, dict) and key in {"mean_counts", "mean_component_ms"}:
                result.update({f"{key}_{nested}": item for nested, item in value.items()})
            else:
                result[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        flattened.append(result)
    keys = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flattened)


def figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    full = [row for row in summary["metrics"] if row["category"] == "all"]
    fig, axis = plt.subplots(figsize=(12, 6))
    for row in full:
        accuracy = row["exact_match"]
        low, high = row["exact_match_ci95"]
        label = f"{LABELS.get(row['arm'], row['arm'])} · rounds≤{row['max_rounds']} · n={row['n']}"
        axis.errorbar(row["cold_p50_ms"] / 1000, accuracy,
                      yerr=[[max(0, accuracy - low)], [max(0, high - accuracy)]],
                      fmt="o", markersize=8, capsize=4, color=COLORS.get(row["arm"]), label=label)
    axis.set(xlabel="Cold end-to-end query latency · p50 (seconds)", ylabel="Exact match",
             title="Global-corpus retrieval: answer accuracy and total online cost", ylim=(-.03, 1.03))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=.18)
    axis.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, fontsize=9)
    fig.text(.08, .01, "Whiskers: nominal 95% Wilson intervals on the full fixed subset; questions may share facts. No model loading or offline index cost is included.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    path = directory / "accuracy_vs_cold_latency.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths = [str(path.relative_to(output))]
    recovery = [row for row in summary["paired_vs_basic"] if row["subset"] == "basic_rag_exact_match_failures"]
    fig, axis = plt.subplots(figsize=(11, 5.5))
    if recovery:
        y = list(range(len(recovery)))
        axis.barh(y, [row["enhanced_exact_match"] for row in recovery],
                  color=[COLORS.get(row["arm"], "#64748b") for row in recovery])
        labels = [f"{LABELS.get(row['arm'], row['arm'])} · rounds≤{row['max_rounds']}" for row in recovery]
        axis.set_yticks(y, labels)
        for index, row in enumerate(recovery):
            axis.text(min(.99, row["enhanced_exact_match"] + .015), index,
                      f"{row['enhanced_correct']} / {row['n_paired']} paired failures", va="center", fontsize=9)
        axis.invert_yaxis()
    else:
        axis.text(.5, .5, "No paired basic-RAG failures are available in this run.", ha="center", transform=axis.transAxes)
    axis.set(xlim=(0, 1.05), xlabel="Fraction recovered among basic-RAG exact-match failures",
             title="Conditional recovery: the denominator is basic RAG's failed questions")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis="x", alpha=.15)
    fig.text(.04, .01, "Posthoc conditional subset. These bars have no population confidence intervals and must be read alongside the full-test comparison.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    path = directory / "basic_failure_recovery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths + [str(path.relative_to(output))]


def build(run_dir, output, root=REPOSITORY, no_plots=False):
    validate_report_output(output, run_dir)
    output.mkdir(parents=True, exist_ok=True)
    inspection = inspect_run(run_dir, root)
    records = inspection["records"]
    config = inspection["manifest"].get("identity", {}).get("config", {})
    summary = aggregate(records, config.get("seed", 42))
    summary["limitations"] = [note.replace("Single-model, greedy", "One fixed generator plus an optional reranker; greedy")
                              for note in summary["limitations"]]
    for comparison in summary["paired_vs_basic"]:
        if comparison["subset"] == "basic_rag_exact_match_failures":
            for key in comparison:
                if "ci95" in key:
                    comparison[key] = None
            comparison["uncertainty_note"] = "Posthoc basic-failure subset; population confidence intervals are withheld."
    summary.update(report_version=VERSION, report_status=inspection["status"],
                   stop_reason=inspection["runner_summary"].get("stop_reason", "missing"),
                   completion={key: inspection[key] for key in ("planned_unique_runs", "successful_unique_runs",
                       "missing_unique_runs", "unresolved_errors", "historical_error_records", "duplicate_record_ids")},
                   verification_issues=inspection["issues"])
    receipts = []
    for name in ("results.jsonl", "manifest.json", "summary.json", "checkpoint.json", "resources.jsonl", "guard-status.json", "process.log"):
        path = run_dir / name
        if path.exists():
            raw = inspection["raw_results"] if name == "results.jsonl" else path.read_bytes()
            receipts.append(archive(output, Path("run") / (name + ".gz"), raw, path, compress=True))
    for key, source in inspection["verified_inputs"].items():
        receipts.append(archive(output, Path("inputs") / (key + ".jsonl.gz"), source["raw"], source["path"],
                                compress=True, verified_against_generation_manifest=True, original_basename=source["path"].name))
    for source in inspection["verified_sources"]:
        receipts.append(archive(output, Path("source") / source["relative"], source["raw"], source["path"],
                                verified_against_generation_manifest=True))
    provenance_path = root / "data/adaptive_musique.manifest.json"
    if provenance_path.exists():
        provenance = load_json(provenance_path)
        receipts.append(archive(output, Path("provenance") / provenance_path.name, provenance_path.read_bytes(), provenance_path))
        for entry in provenance.get("outputs", []):
            if "_dev.jsonl" not in entry["path"] and "corpus_provenance" not in entry["path"]:
                continue
            source = safe_source(root, entry["path"])
            if not source.exists() or sha(source.read_bytes()) != entry["sha256"]:
                summary["verification_issues"].append({"kind": "dev_provenance_hash_mismatch", "source": str(source)})
                summary["report_status"] = "INCOMPLETE_OR_UNVERIFIED"
                continue
            receipts.append(archive(output, Path("provenance") / (source.name + ".gz"), source.read_bytes(), source,
                                    compress=True, verified_against_data_manifest=True, original_basename=source.name))
    else:
        summary["verification_issues"].append({"kind": "missing_adaptive_data_manifest"})
        summary["report_status"] = "INCOMPLETE_OR_UNVERIFIED"
    model_paths = set((root / "models").glob("*manifest*.json"))
    model_paths.update((root / "models").glob("*/*manifest*.json"))
    for source in sorted(model_paths):
        relative = source.relative_to(root / "models")
        receipts.append(archive(output, Path("model-provenance") / relative, source.read_bytes(), source))
    for name in ("docs/adaptive-protocol.md", "docs/adaptive-data.md", "docs/reranker-model-manifest.json",
                 "docs/reranker-provenance.md", "scripts/prepare_adaptive_musique.py", "scripts/download_reranker.py"):
        source = root / name
        if source.exists():
            receipts.append(archive(output, Path("provenance") / source.name, source.read_bytes(), source))
    own_source = Path(__file__).resolve()
    receipts.append(archive(output, Path("source") / "scripts/build_adaptive_report.py", own_source.read_bytes(), own_source))
    helper_source = own_source.with_name("report_paths.py")
    receipts.append(archive(output, Path("source") / "scripts/report_paths.py", helper_source.read_bytes(), helper_source))
    license_text = ("# MuSiQue attribution and modifications\n\n"
                    "Trivedi et al. (2022), MuSiQue. Dataset: https://github.com/StonyBrookNLP/musique\n\n"
                    "Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0): "
                    "https://creativecommons.org/licenses/by/4.0/\n\n"
                    "This experiment builds a global corpus from supplied answerable-development paragraphs, "
                    "deduplicates exact title/text pairs, assigns content-hash IDs, and selects disjoint balanced "
                    "local development/test question subsets. These are modifications, not an official full-Wikipedia "
                    "MuSiQue evaluation. Source hashes, selection rules, exclusions and corpus mappings are archived. "
                    "Model weights are not redistributed. Basic-failure recovery is posthoc conditional analysis; "
                    "it is not a population estimate of all hard questions.\n")
    (output / "DATA_LICENSE.md").write_text(license_text)
    csv_rows(output / "metrics.csv", summary["metrics"])
    pairs = [{"comparison_family": "vs_basic", **row} for row in summary["paired_vs_basic"]]
    pairs += [{"comparison_family": "cache_methods", **row} for row in summary["paired_core_comparisons"]]
    csv_rows(output / "pairs.csv", pairs)
    complete = summary["report_status"] == "complete"
    summary["figures"] = figures(summary, output) if complete and not no_plots else []
    if complete:
        text = report_tables(summary) + "\n## Figures\n\n" + "\n\n".join(f"![{Path(path).stem}]({path})" for path in summary["figures"])
    else:
        text = ("# Adaptive evaluation — incomplete or unverified\n\n"
                "Headline accuracy figures are withheld. Raw ledgers and partial metrics remain available for diagnosis.\n\n" +
                "```json\n" + json.dumps(summary["completion"], indent=2) + "\n```\n\n" +
                "Verification issues: " + json.dumps(summary["verification_issues"]) + "\n")
    (output / "tables.md").write_text(text)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "artifact-manifest.json").write_text(json.dumps({"report_version": VERSION,
        "report_source_sha256": sha(own_source.read_bytes()), "artifacts": receipts}, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/adaptive-test"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/2026-09-11/adaptive"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = build(args.run_dir, args.output_dir, no_plots=args.no_plots)
    print(json.dumps({"status": summary["report_status"], "output_dir": str(args.output_dir),
                      "successful_unique_runs": summary["successful_unique_runs"], "issues": summary["verification_issues"]}))
    return 2 if args.require_complete and summary["report_status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
