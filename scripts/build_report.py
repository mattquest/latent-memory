#!/usr/bin/env python3
"""Build an auditable CPU-only report from saved experiment/judge artifacts.

Never imports the generation harness or model runtime, changes run files, or
regenerates predictions. Incomplete jobs remain in CSV/receipts, but are excluded
from headline tables and figures unless --include-incomplete is explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_paths import validate_report_output

REPORT_VERSION = "artifact-report-v1"
CASE_KEYS = ("experiment", "arm", "hops", "control", "bridge_ratio", "precision", "update_policy")
COLORS = {"direct_text/bf16": "#303d49", "text/bf16": "#cc6b36", "latent/bf16": "#197c80", "latent/int8": "#6852a3", "latent/int4": "#b34775"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_") or "artifact"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def read_jsonl_snapshot(path):
    """One read produces both analysis and archive; tolerate only torn last line."""
    if not path.exists():
        return [], b"", []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, issues = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            kind = "partial_trailing_line" if index == len(lines) - 1 and not line.endswith(b"\n") else "malformed_json"
            issues.append({"kind": kind, "line": index + 1, "error": str(exc)})
    return rows, raw, issues


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.mean(values) if values else None


def percentile(values, quantile):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def wilson(values):
    if not values:
        return [None, None]
    n, p, z = len(values), mean(values), 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0, center - radius), min(1, center + radius)]


def paired_interval(values, seed=42):
    # A degenerate empirical bootstrap is not a meaningful certainty statement.
    if len(values) < 2 or len(set(values)) == 1:
        return [None, None]
    rng = random.Random(seed)
    means = [mean(rng.choices(values, k=len(values))) for _ in range(999)]
    return [percentile(means, .025), percentile(means, .975)]


def variant(case):
    return f"{case.get('arm', 'unknown')}/{case.get('precision', 'bf16')}"


def is_beam(row):
    return row.get("dataset") == "beam" or row.get("metric_role") == "lexical_overlap_only_not_behavior_accuracy"


def archive_bytes(output, relative, raw, source, compressed=False):
    destination = output / "artifacts" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw
    destination.write_bytes(payload)
    return {"source": str(source), "artifact": str(destination.relative_to(output)),
            "source_bytes": len(raw), "source_sha256": digest(raw),
            "artifact_bytes": len(payload), "artifact_sha256": digest(payload),
            "compression": "gzip" if compressed else None}


def discover_jobs(run_root, matrix):
    expected = {}
    for entry in matrix.get("jobs", []):
        expected[entry["name"]] = {**matrix.get("defaults", {}), **entry.get("config", {})}
    if (run_root / "results.jsonl").exists():
        return [(run_root.name, run_root, expected.get(run_root.name, {}))]
    names = set(expected)
    if run_root.exists():
        names.update(p.name for p in run_root.iterdir() if p.is_dir() and
                     ((p / "results.jsonl").exists() or (p / "manifest.json").exists()))
    return [(name, run_root / name, expected.get(name, {})) for name in sorted(names)]


def collect_jobs(run_root, matrix, output):
    jobs, records, receipts, issues = [], [], [], []
    for name, directory, expected_config in discover_jobs(run_root, matrix):
        manifest = read_json(directory / "manifest.json")
        summary = read_json(directory / "summary.json")
        checkpoint = read_json(directory / "checkpoint.json")
        rows, raw, jsonl_issues = read_jsonl_snapshot(directory / "results.jsonl")
        latest = {}
        for row in rows:
            if "id" not in row:
                jsonl_issues.append({"kind": "missing_record_id"})
                continue
            latest[row["id"]] = row
        ok = [r for r in latest.values() if r.get("status") == "ok"]
        errors = [r for r in latest.values() if r.get("status") != "ok"]
        planned = summary.get("planned_runs", checkpoint.get("planned"))
        complete = (summary.get("stop_reason") == "complete" and planned is not None
                    and len(ok) == planned and not errors and not jsonl_issues)
        if complete:
            status = "complete"
        elif errors or any(i["kind"] == "malformed_json" for i in jsonl_issues):
            status = "errors"
        elif rows or manifest:
            status = "incomplete"
        else:
            status = "not_started"
        config = manifest.get("identity", {}).get("config", expected_config)
        job = {"job": name, "status": status, "planned_runs": planned,
               "successful_unique_runs": len(ok), "failed_latest_runs": len(errors),
               "historical_error_records": sum(r.get("status") != "ok" for r in rows),
               "raw_records": len(rows), "duplicate_record_ids": len(rows) - len(latest),
               "partial_or_malformed_lines": jsonl_issues,
               "stop_reason": summary.get("stop_reason"),
               "dataset": config.get("dataset"), "config": config,
               "generation_identity": manifest.get("identity"),
               "generation_machine": {key: manifest.get("machine", {}).get(key)
                                      for key in ("git_head", "code_sha256")},
               "source_results_sha256": digest(raw) if raw else None}
        jobs.append(job)
        for row in ok:
            records.append({**row, "report_job": name, "report_job_status": status})
        issues.extend({"job": name, **issue} for issue in jsonl_issues)
        if raw:
            receipts.append(archive_bytes(output, Path("runs") / name / "results.jsonl.gz", raw,
                                          directory / "results.jsonl", compressed=True))
        for filename in ("manifest.json", "summary.json", "checkpoint.json"):
            path = directory / filename
            if path.exists():
                receipts.append(archive_bytes(output, Path("runs") / name / filename, path.read_bytes(), path))
    return jobs, records, receipts, issues


def resolve_recorded_source(repository, relative, expected_sha256, git_heads):
    """Return only bytes matching the executed hash, never a nearby revision."""
    repository = repository.resolve()
    path = Path(relative)
    if (path.is_absolute() or not path.parts or ".." in path.parts or "\\" in relative
            or not (repository / path).resolve().is_relative_to(repository)):
        raise ValueError("Executable source must be a safe repository-relative path")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Executable source requires a recorded SHA-256")
    current = repository / path
    if current.is_file():
        raw = current.read_bytes()
        if digest(raw) == expected_sha256:
            return raw, {"resolution": "matching_working_file", "source": str(current)}
    attempts = []
    for revision in sorted(set(git_heads), key=str):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            attempts.append({"git_head": revision, "error": "invalid recorded commit hash"})
            continue
        result = subprocess.run(["git", "show", f"{revision}:{path.as_posix()}"],
                                cwd=repository, capture_output=True, check=False)
        if result.returncode:
            attempts.append({"git_head": revision, "error": "recorded Git blob unavailable"})
        elif digest(result.stdout) != expected_sha256:
            attempts.append({"git_head": revision, "error": "recorded Git blob SHA-256 mismatch",
                             "actual_sha256": digest(result.stdout)})
        else:
            return result.stdout, {"resolution": "matching_recorded_git_blob", "git_head": revision,
                                   "source": f"git:{revision}:{path.as_posix()}"}
    raise ValueError("Exact executed source bytes unavailable: " + json.dumps(attempts, sort_keys=True))


def collect_executable_artifacts(jobs, output, repository=None):
    """Deduplicate executed source sets across jobs and recorded Git heads."""
    repository = Path(repository or Path(__file__).resolve().parents[1])
    groups, receipts, issues, versions = {}, [], [], []
    for job in jobs:
        identity = job.get("generation_identity") or {}
        machine = job.get("generation_machine") or {}
        identity_hashes = identity.get("source_code_sha256") or {}
        machine_hashes = machine.get("code_sha256") or {}
        if not identity and not machine_hashes and job.get("status") == "not_started":
            continue  # Unstarted jobs already fail the generation completion gate.
        if not identity_hashes and not machine_hashes:
            issues.append({"kind": "missing_executable_source_hashes", "job": job["job"]})
            continue
        if (not isinstance(identity_hashes, dict) or not isinstance(machine_hashes, dict) or
                any(identity_hashes[key] != machine_hashes[key] for key in set(identity_hashes) & set(machine_hashes))):
            issues.append({"kind": "conflicting_executable_source_hashes", "job": job["job"]})
            continue
        hashes = {**machine_hashes, **identity_hashes}
        version = digest(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())
        group = groups.setdefault(version, {"source_version_sha256": version, "files": hashes, "jobs": [], "git_heads": []})
        group["jobs"].append(job["job"])
        if machine.get("git_head"):
            group["git_heads"].append(machine["git_head"])
    for version, group in sorted(groups.items()):
        details = {"source_version_sha256": version, "used_by_jobs": sorted(group["jobs"]),
                   "recorded_git_heads": sorted(set(group["git_heads"]), key=str), "files": [], "complete": True}
        for relative, expected in sorted(group["files"].items()):
            try:
                raw, resolution = resolve_recorded_source(repository, relative, expected, group["git_heads"])
            except (OSError, TypeError, ValueError) as error:
                issues.append({"kind": "executed_source_unavailable", "source": relative,
                               "expected_sha256": expected, "source_version_sha256": version,
                               "jobs": details["used_by_jobs"], "error": str(error)})
                details["complete"] = False
                continue
            receipt = archive_bytes(output, Path("source") / version / relative, raw, resolution["source"])
            receipt.update(verified_against_generation_manifest=True, original_source_path=relative,
                           source_version_sha256=version, used_by_jobs=details["used_by_jobs"],
                           recorded_git_heads=details["recorded_git_heads"], resolution=resolution["resolution"])
            if resolution.get("git_head"):
                receipt["resolved_git_head"] = resolution["git_head"]
            receipts.append(receipt)
            details["files"].append({"path": relative, "sha256": expected, "artifact": receipt["artifact"],
                                     "resolution": resolution["resolution"], "resolved_git_head": resolution.get("git_head")})
        versions.append(details)
    return receipts, issues, versions


def metric_groups(records):
    groups = defaultdict(list)
    for row in records:
        case = row["case"]
        base = (row["report_job"], row.get("dataset", "unknown"),
                tuple(case.get(k) for k in CASE_KEYS))
        groups[base + ("all", None)].append(row)
        if case["experiment"] == "E2" and row.get("example_metadata", {}).get("chain_length") is not None:
            groups[base + ("chain_length", row["example_metadata"]["chain_length"])].append(row)
        if case["experiment"] == "E6":
            groups[base + ("category", row.get("category", "unknown"))].append(row)
    metrics = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        job, dataset, case_values, stratification, stratum = key
        case = dict(zip(CASE_KEYS, case_values))
        by_example = defaultdict(list)
        for row in rows:
            by_example[row["example_id"]].append(row)
        primary = [min(value, key=lambda r: r["case"].get("repeat", 0)) for value in by_example.values()]
        em = [r["scores"]["exact_match"] for r in primary]
        ci = [None, None] if is_beam(primary[0]) else wilson(em)
        result = {"job": job, "job_status": rows[0]["report_job_status"], "dataset": dataset,
                  **case, "variant": variant(case), "stratification": stratification,
                  "stratum": stratum, "n_examples": len(primary), "n_runs": len(rows),
                  "example_cohort_sha256": digest(json.dumps(sorted(by_example), separators=(",", ":")).encode()),
                  "metric_role": "BEAM lexical proxy, not benchmark accuracy" if is_beam(primary[0]) else "QA exact match",
                  "exact_match": mean(em), "exact_match_ci95_low": ci[0], "exact_match_ci95_high": ci[1],
                  "f1": mean(r["scores"]["f1"] for r in primary),
                  "warm_latency_p50_ms": percentile([r["latency_ms"] for r in rows], .5),
                  "warm_latency_p95_ms": percentile([r["latency_ms"] for r in rows], .95),
                  "cache_precompute_union_p50_ms": percentile([r.get("cold_ingest_ms", 0) for r in rows], .5),
                  "retrieval_mean_ms": mean(r.get("retrieval_ms") for r in rows),
                  "schedule_selection_mean_ms": mean(r.get("schedule_selection_ms") for r in rows),
                  "mean_executed_evidence_hops": mean(r.get("evidence_hops_executed") for r in primary),
                  "fraction_document_truncation": mean(bool(set(r.get("truncated_document_ids", [])) &
                                                            set(r.get("evidence_ids", []))) for r in primary),
                  "fraction_any_candidate_truncation": mean(bool(r.get("truncated_document_ids")) for r in primary),
                  "fraction_output_cap_reached": mean(r.get("generation_stop") == "max_tokens" for r in primary),
                  "mean_cache_bytes": mean(r.get("cache", {}).get("nbytes") for r in rows),
                  "mean_cache_bytes_per_token": mean(r["cache"]["nbytes"] / r["cache"]["seq_len"]
                                                     for r in rows if r.get("cache", {}).get("seq_len", 0)),
                  "mean_support_recall": mean(r.get("support_recall") for r in primary)}
        count_keys = set().union(*(r.get("counts", {}) for r in rows))
        result.update({f"mean_{name}": mean(r.get("counts", {}).get(name) for r in rows)
                       for name in sorted(count_keys)})
        metrics.append(result)
    return metrics


def paired_groups(records):
    grouped = defaultdict(dict)
    for row in records:
        if row["case"].get("repeat", 0) != 0 or is_beam(row):
            continue
        c = row["case"]
        exp = c["experiment"]
        # E4 varies recomputation; E5 varies precision. Other experiments compare
        # arms at an identical hop, ratio, and update policy on common examples.
        key = (row["report_job"], row["dataset"], exp, c["hops"],
               None if exp == "E4" else c["bridge_ratio"], c.get("update_policy", "all"))
        label = f"{variant(c)}/{c.get('control', 'correct')}"
        if exp == "E4":
            label += f"/ratio={c['bridge_ratio']:g}"
        grouped[key].setdefault(label, {})[row["example_id"]] = row
    output = []
    for key, variants in sorted(grouped.items(), key=lambda item: str(item[0])):
        exp = key[2]
        if exp == "E4":
            baselines = [v for v in variants if v.endswith("/ratio=1")]
        elif exp in ("E3", "E5"):
            baselines = [v for v in variants if v == "latent/bf16/correct"]
        else:
            baselines = [v for v in variants if v.startswith("latent/") and v.endswith("/correct")]
        for baseline in baselines:
            for comparator in sorted(set(variants) - {baseline}):
                shared = sorted(set(variants[baseline]) & set(variants[comparator]))
                if not shared:
                    continue
                differences = [variants[baseline][i]["scores"]["exact_match"] - variants[comparator][i]["scores"]["exact_match"] for i in shared]
                ci = paired_interval(differences)
                output.append({"job": key[0], "job_status": variants[baseline][shared[0]]["report_job_status"],
                               "dataset": key[1], "experiment": exp, "hops": key[3],
                               "bridge_ratio": key[4], "update_policy": key[5],
                               "baseline": baseline, "comparator": comparator,
                               "contrast_type": "condition", "stratification": "all", "stratum": None,
                               "n_paired_examples": len(shared), "exact_match_difference": mean(differences),
                               "paired_bootstrap_ci95_low": ci[0], "paired_bootstrap_ci95_high": ci[1],
                               "interval_note": "empirical example bootstrap" if ci[0] is not None else "not estimated: constant differences or fewer than two examples",
                               "baseline_only_correct": sum(x == 1 for x in differences),
                               "comparator_only_correct": sum(x == -1 for x in differences),
                               "common_example_ids_sha256": digest(json.dumps(shared).encode())})
    # Version filtering is an intervention within an otherwise identical arm.
    # Keep it separate from cross-arm contrasts, whose key includes the policy.
    policy_groups = defaultdict(dict)
    for row in records:
        c = row["case"]
        if c["experiment"] != "E6" or c.get("repeat", 0) != 0 or is_beam(row):
            continue
        key = (row["report_job"], row["dataset"], c["hops"], c["bridge_ratio"],
               variant(c), c.get("control", "correct"))
        for stratification, stratum in (("all", None), ("category", row.get("category", "unknown"))):
            policy_groups[key + (stratification, stratum)].setdefault(c.get("update_policy", "all"), {})[row["example_id"]] = row
    for key, policies in sorted(policy_groups.items(), key=lambda item: str(item[0])):
        current, all_versions = policies.get("current", {}), policies.get("all", {})
        shared = sorted(set(current) & set(all_versions))
        if not shared:
            continue
        differences = [current[i]["scores"]["exact_match"] - all_versions[i]["scores"]["exact_match"] for i in shared]
        ci = paired_interval(differences)
        output.append({"job": key[0], "job_status": current[shared[0]]["report_job_status"],
                       "dataset": key[1], "experiment": "E6", "hops": key[2], "bridge_ratio": key[3],
                       "update_policy": "current_minus_all", "contrast_type": "update_policy",
                       "stratification": key[6], "stratum": key[7],
                       "baseline": f"{key[4]}/{key[5]}/current", "comparator": f"{key[4]}/{key[5]}/all",
                       "n_paired_examples": len(shared), "exact_match_difference": mean(differences),
                       "paired_bootstrap_ci95_low": ci[0], "paired_bootstrap_ci95_high": ci[1],
                       "interval_note": "empirical example bootstrap" if ci[0] is not None else "not estimated: constant differences or fewer than two examples",
                       "baseline_only_correct": sum(x == 1 for x in differences),
                       "comparator_only_correct": sum(x == -1 for x in differences),
                       "common_example_ids_sha256": digest(json.dumps(shared).encode())})
    return output


def collect_judge(judge_root, records, output):
    path = judge_root / "judgments.jsonl"
    rows, raw, issues = read_jsonl_snapshot(path)
    receipts = []
    if raw:
        receipts.append(archive_bytes(output, Path("beam-judge/judgments.jsonl.gz"), raw, path, True))
    for filename in ("manifest.json", "summary.json"):
        source = judge_root / filename
        if source.exists():
            receipts.append(archive_bytes(output, Path("beam-judge") / filename, source.read_bytes(), source))
    candidates = {(row["id"], digest(row["prediction"].encode())): row for row in records
                  if is_beam(row) and row["case"].get("repeat", 0) == 0}
    latest, unmatched = {}, 0
    for row in rows:
        key = (row.get("source_result_id"), row.get("candidate_sha256"))
        if key not in candidates:
            unmatched += 1
            continue
        latest[key] = row
    grouped = defaultdict(list)
    for key, source in candidates.items():
        case = tuple(source["case"].get(k) for k in CASE_KEYS)
        grouped[(source["report_job"], case, "all")].append((source, latest.get(key)))
        if source["case"]["experiment"] == "E6":
            grouped[(source["report_job"], case, source.get("category", "unknown"))].append((source, latest.get(key)))
    metrics = []
    for (job, case_values, category), values in sorted(grouped.items(), key=lambda item: str(item[0])):
        valid = [(src, grade) for src, grade in values if grade and grade.get("status") == "ok"]
        invalid = sum(bool(grade) and grade.get("status") != "ok" for _, grade in values)
        conversations = sorted({src.get("example_metadata", {}).get("conversation_id", src["example_id"].split(":")[0]) for src, _ in values})
        c = dict(zip(CASE_KEYS, case_values))
        metrics.append({"job": job, "job_status": values[0][0]["report_job_status"], "dataset": "beam", **c,
                        "variant": variant(c), "category": category,
                        "n_candidate_answers": len(values), "n_valid_judgments": len(valid),
                        "n_invalid_judgments": invalid, "n_pending_judgments": len(values) - len(valid) - invalid,
                        "n_conversations": len(conversations), "conversation_ids": conversations,
                        "mean_criterion_fraction": mean(g["judgment"]["criterion_fraction"] for _, g in valid),
                        "all_criteria_pass_rate": mean(g["judgment"]["all_criteria_met"] for _, g in valid),
                        "judge_mean_latency_ms": mean(g.get("judge_latency_ms") for _, g in valid),
                        "judge_mean_generated_tokens": mean(g.get("judge_generated_tokens") for _, g in valid),
                        "warm_generation_p50_ms": percentile([s["latency_ms"] for s, _ in values], .5),
                        "metric_role": "Exploratory same-model local rubric judge; not official BEAM",
                        "interval_note": "No population interval: probes clustered within selected conversations"})
    return {"status": "absent" if not raw else "unmatched_or_no_candidates" if not candidates else "complete_for_generation_snapshot" if all(m["n_pending_judgments"] == 0 and m["n_invalid_judgments"] == 0 for m in metrics) and not issues else "incomplete_or_invalid",
            "source_summary": read_json(judge_root / "summary.json"),
            "unmatched_judgment_records": unmatched, "issues": issues, "groups": metrics}, receipts


def csv_output(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def e2_plot_batches(metrics, stratification):
    """Keep structural jobs distinct even when they share a dataset name."""
    batches = defaultdict(list)
    for row in metrics:
        if row["experiment"] == "E2" and row["stratification"] == stratification:
            batches[(row["job"], row["dataset"])].append(row)
    return batches


def e2_cohort_curves(rows):
    """Connect only identical example cohorts and settings across hop counts."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["variant"], row["control"], row["bridge_ratio"], row["update_policy"],
               row["n_examples"], row["example_cohort_sha256"])
        groups[key].append(row)
    variant_counts = Counter(key[0] for key in groups)
    curves = []
    for key, values in sorted(groups.items(), key=lambda pair: str(pair[0])):
        line = sorted(values, key=lambda row: row["hops"])
        if len({row["hops"] for row in line}) != len(line):
            raise ValueError("Duplicate E2 hop coordinates within a job and example cohort")
        label = f"{key[0]} (n={key[4]})"
        if variant_counts[key[0]] > 1:
            label += f" · cohort {key[5][:6]}; {key[1]}; r={key[2]:g}; {key[3]}"
        curves.append((label, line))
    return curves


def plots(metrics, output, include_incomplete=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": .18, "savefig.dpi": 180})
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    visible = [m for m in metrics if m["dataset"] != "beam" and
               (m["job_status"] == "complete" or include_incomplete)]
    artifacts = []
    def finish(fig, name, title, rows):
        provisional = any(r["job_status"] != "complete" for r in rows)
        fig.suptitle(("INCOMPLETE SNAPSHOT — " if provisional else "") + title, fontsize=13, y=1.02)
        fig.tight_layout()
        path = figure_dir / (safe_name(name) + ".png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(str(path.relative_to(output)))
    e1 = defaultdict(list)
    for m in visible:
        if m["experiment"] == "E1" and m["stratification"] == "all" and m["dataset"] in {"hotpotqa", "musique"}:
            e1[(m["job"], m["dataset"])].append(m)
    arm_order = {name: i for i, name in enumerate(COLORS)}
    for (job, dataset), rows in sorted(e1.items()):
        rows = sorted(rows, key=lambda r: (arm_order.get(r["variant"], 99), r["hops"]))
        positions = list(range(len(rows)))
        colors = [COLORS.get(r["variant"], "#87979e") for r in rows]
        labels = [f"{r['variant']}\nn={r['n_examples']}" for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
        axes[0].bar(positions, [100*r["exact_match"] for r in rows], color=colors)
        for i, row in enumerate(rows):
            lo, hi = row["exact_match_ci95_low"], row["exact_match_ci95_high"]
            if lo is not None:
                axes[0].errorbar(i, 100*row["exact_match"],
                    yerr=[[100*max(0, row["exact_match"]-lo)], [100*max(0, hi-row["exact_match"])]],
                    color="#25313a", capsize=4, fmt="none")
        axes[0].set(ylabel="Exact match (%)", ylim=(0, 108), title="Answer quality · 95% Wilson intervals")
        axes[1].bar([i-.19 for i in positions], [r["warm_latency_p50_ms"]/1000 for r in rows],
                    width=.36, color=colors, label="p50")
        axes[1].bar([i+.19 for i in positions], [r["warm_latency_p95_ms"]/1000 for r in rows],
                    width=.36, color=colors, alpha=.45, hatch="//", label="p95")
        axes[1].set(ylabel="Warm generation latency (s)", title="Online model work; precompute excluded")
        axes[1].legend(fontsize=9)
        for ax in axes:
            ax.set_xticks(positions, labels, fontsize=9)
        finish(fig, f"e1_{job}", f"E1 · {dataset} · {job}", rows)
    for (job, dataset), rows in sorted(e2_plot_batches(visible, "all").items()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for legend, line in e2_cohort_curves(rows):
            color = COLORS.get(line[0]["variant"])
            axes[0].plot([r["hops"] for r in line], [100*r["exact_match"] for r in line], "o-", label=legend, color=color)
            axes[1].plot([r["hops"] for r in line], [r["warm_latency_p50_ms"]/1000 for r in line], "o-", label=legend, color=color)
        axes[0].set(ylabel="Exact match (%)", ylim=(-3, 103), xlabel="Scheduled evidence batches", title="Answer quality")
        axes[1].set(ylabel="Warm generation p50 (s)", xlabel="Scheduled evidence batches", title="Online model work; precompute excluded")
        for ax in axes:
            ax.set_xticks(sorted({r["hops"] for r in rows}))
        axes[0].legend(fontsize=8)
        finish(fig, f"e2_{job}", f"E2 · {dataset} · {job} · fixed evidence schedule", rows)
    for (job, dataset), chains in sorted(e2_plot_batches(visible, "chain_length").items()):
        lengths = sorted({r["stratum"] for r in chains})
        fig, axes = plt.subplots(len(lengths), 2, figsize=(11, max(4, 2.65*len(lengths))), squeeze=False)
        for i, length in enumerate(lengths):
            rows = [r for r in chains if r["stratum"] == length]
            for legend, line in e2_cohort_curves(rows):
                color = COLORS.get(line[0]["variant"])
                axes[i, 0].plot([r["hops"] for r in line], [100*r["exact_match"] for r in line], "o-", color=color, label=legend)
                axes[i, 1].plot([r["hops"] for r in line], [r["warm_latency_p50_ms"]/1000 for r in line], "o-", color=color)
            axes[i, 0].set(ylabel=f"{length} facts · EM (%)", ylim=(-3, 103), xlabel="Scheduled evidence batches")
            axes[i, 1].set(ylabel="Warm p50 (s)", xlabel="Scheduled evidence batches")
            for ax in axes[i]:
                ax.set_xticks(sorted({r["hops"] for r in rows}))
            axes[i, 0].legend(fontsize=7, loc="best")
        finish(fig, f"e2_{job}_by_chain_length", f"E2 · {dataset} · {job} · by chain length", chains)
    for experiment in ("E3", "E4", "E5"):
        batches = defaultdict(list)
        for row in visible:
            if row["experiment"] == experiment and row["stratification"] == "all":
                batches[(row["job"], row["dataset"])].append(row)
        for (job, dataset), rows in sorted(batches.items()):
            sort_key = "control" if experiment == "E3" else "bridge_ratio" if experiment == "E4" else "precision"
            order = {"correct": 0, "mismatched": 1, "zero": 2, "random": 3, "no_context": 4,
                     "bf16": 0, "int8": 1, "int4": 2}
            rows = sorted(rows, key=lambda r: order.get(r[sort_key], r[sort_key]))
            labels = [f"{100*r['bridge_ratio']:g}%" if experiment == "E4" else str(r[sort_key]) for r in rows]
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
            positions = list(range(len(rows)))
            color = ["#197c80" if i == 0 else "#87979e" for i in positions]
            axes[0].bar(positions, [100*r["exact_match"] for r in rows], color=color)
            for i, r in enumerate(rows):
                lo, hi = r["exact_match_ci95_low"], r["exact_match_ci95_high"]
                if lo is not None:
                    axes[0].errorbar(i, 100*r["exact_match"], yerr=[[100*max(0,r["exact_match"]-lo)], [100*max(0,hi-r["exact_match"])]], color="#25313a", capsize=3, fmt="none")
            axes[0].set(ylabel="Exact match (%)", ylim=(0, 108), title="Accuracy · Wilson intervals")
            if experiment == "E5":
                axes[1].bar(positions, [(r["mean_cache_bytes_per_token"] or 0)/1024 for r in rows], color=color)
                axes[1].set(ylabel="Packed KV KiB per token", title="Storage; attention dequantizes to BF16")
            else:
                axes[1].bar(positions, [r["warm_latency_p50_ms"]/1000 for r in rows], color=color)
                axes[1].set(ylabel="Warm generation p50 (s)", title="Online model work; precompute excluded")
            for ax in axes:
                ax.set_xticks(positions, labels, rotation=20 if experiment == "E3" else 0)
            finish(fig, f"{experiment.lower()}_{job}", f"{experiment} · {dataset} · n={','.join(map(str,sorted({r['n_examples'] for r in rows})))}", rows)
    return artifacts


def judge_plots(judge, output):
    """Use only fully judged, completed generation cohorts; never BEAM EM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for row in judge["groups"]:
        if row["category"] == "all" and row["job_status"] == "complete":
            groups[(row["job"], row["experiment"])].append(row)
    figures = []
    for (job, experiment), rows in sorted(groups.items()):
        if any(r["n_valid_judgments"] != r["n_candidate_answers"] for r in rows):
            continue
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        if experiment == "E2":
            for label in sorted({r["variant"] for r in rows}):
                line = sorted([r for r in rows if r["variant"] == label], key=lambda r: r["hops"])
                legend = f"{label} (n={line[0]['n_valid_judgments']})"
                axes[0].plot([r["hops"] for r in line], [100*r["mean_criterion_fraction"] for r in line], "o-", color=COLORS.get(label), label=legend)
                axes[1].plot([r["hops"] for r in line], [r["warm_generation_p50_ms"]/1000 for r in line], "o-", color=COLORS.get(label))
            for ax in axes:
                ax.set(xlabel="Scheduled evidence batches", xticks=sorted({r["hops"] for r in rows}))
            axes[1].set(ylabel="Warm generation p50 (s)", title="Answer generation; judging excluded")
            axes[0].legend(fontsize=8)
        else:
            rows = sorted(rows, key=lambda r: (r["variant"], r["update_policy"]))
            labels = [r["variant"] for r in rows]
            positions = list(range(len(rows)))
            colors = [COLORS.get(r["variant"], "#87979e") for r in rows]
            axes[0].bar(positions, [100*r["mean_criterion_fraction"] for r in rows], color=colors)
            axes[1].bar(positions, [100*r["all_criteria_pass_rate"] for r in rows], color=colors)
            for ax in axes:
                ax.set_xticks(positions, labels, rotation=15)
            axes[1].set(ylabel="All criteria satisfied (%)", ylim=(0, 105), title="Strict local rubric pass")
        axes[0].set(ylabel="Mean rubric criterion fraction (%)", ylim=(0, 105), title="Exploratory same-model judge")
        cohorts = sorted({r["n_conversations"] for r in rows})
        fig.suptitle(f"BEAM · {experiment} · selected conversations={cohorts} · not official benchmark scores", fontsize=12, y=1.03)
        fig.tight_layout()
        path = output / "figures" / f"beam_judge_{safe_name(job)}_{experiment.lower()}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append(str(path.relative_to(output)))
    return figures


def format_number(value, percentage=False):
    if value is None:
        return "—"
    return f"{value*100:.1f}%" if percentage else f"{value:.2f}"


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows])


def tables(report, include_incomplete=False):
    lines = ["# Experiment results", "", f"Generated {report['generated_utc']}. Status: **{report['status']}**.", "",
             "These are bounded local Qwen3 experiments. They do not establish the original adaptive multi-agent or BEAM-10M thesis. Incomplete jobs remain visible in receipts and CSVs; headline tables and figures use completed jobs only unless explicitly requested.", "", "## Completion", ""]
    amendment = report.get("amendment") or {}
    if amendment:
        planned = sum(j["planned_runs"] or 0 for j in report["jobs"])
        lines += [f"Completion is scoped to the **{len(report['jobs'])} reported jobs / {planned:,} planned conditions** below. "
                  f"The amended original matrix planned {amendment['original_planned_conditions']:,} conditions: "
                  f"{amendment['completed_initial_conditions']:,} initial conditions were completed and "
                  f"{amendment['canceled_original_remainder']:,} remaining original conditions were canceled. "
                  "The original resource guard was intentionally interrupted at this controlled boundary; its stopped status does not imply failure of the completed jobs.", "",
                  f"Amendment reason: {amendment.get('reason', 'See archived amendment receipt.')}", ""]
        if amendment.get("preserved_unused_partial_directories"):
            lines += ["Unused partial runs are excluded from these results and preserved separately: " +
                      ", ".join(f"`{path}`" for path in amendment["preserved_unused_partial_directories"]) + ".", ""]
    lines += [markdown_table(["Job", "State", "Successful / planned", "Unresolved errors"],
                             [[j["job"], j["status"], f"{j['successful_unique_runs']} / {j['planned_runs'] if j['planned_runs'] is not None else '?'}", j["failed_latest_runs"]] for j in report["jobs"]]), ""]
    visible = [r for r in report["metrics"] if r["dataset"] != "beam" and r["stratification"] == "all" and
               (include_incomplete or r["job_status"] == "complete")]
    for experiment in ("E1", "E2", "E3", "E4", "E5", "E6"):
        rows = [r for r in visible if r["experiment"] == experiment]
        if not rows:
            continue
        lines += [f"## {experiment}", "", markdown_table(
            ["Dataset / job", "Arm / KV", "Setting", "n", "EM", "F1", "Warm p50 / p95 (s)", "Host / internal tokens"],
            [[f"{r['dataset']} / {r['job']}", r["variant"], f"h={r['hops']}; r={r['bridge_ratio']:g}; {r['control']}; {r['update_policy']}",
              r["n_examples"], format_number(r["exact_match"], True), format_number(r["f1"], True),
              f"{r['warm_latency_p50_ms']/1000:.2f} / {r['warm_latency_p95_ms']/1000:.2f}",
              f"{r.get('mean_host_output_tokens',0):.1f} / {r.get('mean_internal_decoded_tokens',0):.1f}"] for r in rows]), ""]
    categories = [r for r in report["metrics"] if r["experiment"] == "E6" and r["stratification"] == "category" and
                  r["dataset"] != "beam" and (include_incomplete or r["job_status"] == "complete")]
    if categories:
        lines += ["## E6 by category", "", markdown_table(["Dataset", "Category", "Arm / KV", "Policy", "n", "EM"],
                  [[r["dataset"], r["stratum"], r["variant"], r["update_policy"], r["n_examples"], format_number(r["exact_match"],True)] for r in categories]), ""]
    policy_pairs = [r for r in report.get("paired", []) if r.get("contrast_type") == "update_policy" and
                    r["stratification"] == "all" and (include_incomplete or r["job_status"] == "complete")]
    if policy_pairs:
        lines += ["## E6 paired version filtering", "", markdown_table(
            ["Dataset", "Arm / KV / control", "n paired", "Current − all EM (percentage points)", "Current-only / all-only correct"],
            [[r["dataset"], r["baseline"].removesuffix("/current"), r["n_paired_examples"],
              f"{100*r['exact_match_difference']:+.1f}", f"{r['baseline_only_correct']} / {r['comparator_only_correct']}"] for r in policy_pairs]), ""]
    lines += ["## BEAM: exploratory local rubric judging", "",
              "BEAM EM/F1 are lexical proxies and are excluded from the accuracy tables above. The optional judge uses the same model family and is not the official BEAM evaluator. Questions are clustered within selected conversations; no population confidence intervals are reported. Judge inference cost is separate from answer generation.", ""]
    judge_cohorts = defaultdict(list)
    for row in report["judge"]["groups"]:
        if row["category"] == "all":
            judge_cohorts[(row["job"], row["experiment"])].append(row)
    judged = [r for cohort in judge_cohorts.values() for r in cohort
              if include_incomplete or all(g["job_status"] == "complete" and
                                            g["n_valid_judgments"] == g["n_candidate_answers"] for g in cohort)]
    if judged:
        lines += [markdown_table(["Job / experiment", "Arm / KV", "Hops", "Judging state", "Valid / candidates", "Invalid / pending", "Conversations", "Criterion fraction", "All criteria pass"],
                  [[f"{r['job']} / {r['experiment']}",r["variant"],r["hops"],
                    "complete" if r["n_valid_judgments"] == r["n_candidate_answers"] else "PARTIAL / INVALID",
                    f"{r['n_valid_judgments']} / {r['n_candidate_answers']}",
                    f"{r['n_invalid_judgments']} / {r['n_pending_judgments']}",r["n_conversations"],
                    format_number(r["mean_criterion_fraction"],True),format_number(r["all_criteria_pass_rate"],True)] for r in judged]), ""]
    else:
        lines += ["No fully judged, completed BEAM comparison cohorts are available in this snapshot. Partial valid, invalid, and pending counts remain in beam-judge.csv; default tables withhold partial-cohort scores.", ""]
    lines += ["## Interpretation and costs", "",
              "- E2 changes available evidence through a fixed schedule; it does not measure learned adaptive retrieval. Inspect the direct-text curve before attributing a gain to avoiding evidence-note compression. Structural jobs and chain-length strata have separate figures; connected curves contain identical example cohorts with exact n labels. No changing-cohort aggregate curve is plotted.",
              "- BF16 and int8 latent variants are separate. E5 measures a single storage quantization; multi-hop arms may quantize repeatedly. Attention runs in BF16 after dequantization.",
              "- Warm latency excludes model loading, document-cache precomputation, fixed retrieval/schedule selection, and shared final-question tokenization. The precompute measurement covers the union of documents selected across the experiment grid for an example, not a per-case cold-query latency or full BEAM ingestion.",
              "- Retrieval and schedule timing, truncation, cap-hit rates, cache bytes, and executed hops are in metrics.csv. fraction_document_truncation is the fraction of examples with at least one delivered truncated document; fraction_any_candidate_truncation also counts unused candidates. Headline hop counts are scheduled batches; empty batches do not create new evidence.",
              "- mean_support_recall measures delivered annotation-ID coverage. For synthetic updates, annotations include both old and current versions, so correct current-only filtering can lower this value without losing the required current fact.",
              "- Paired comparisons use common example IDs within a job. Constant-difference bootstrap intervals are left blank rather than implying certainty. Small-subset findings remain exploratory.", "", "## Audit artifacts", "",
              "Raw prediction ledgers and exact normalized inputs are gzip-compressed without altering their uncompressed bytes. Normalized inputs are verified against generation-manifest hashes; the three normalized BEAM corpora and data license notes are included. Exact executable sources are archived under artifacts/source/ by source-set digest, verified against the executed hashes. When current files differ, matching bytes are recovered from a recorded Git commit; unavailable or mismatching bytes fail the complete-release check. Generation manifests, available summaries/checkpoints, judge ledgers, and resource/data receipts are under artifacts/. artifact-manifest.json records SHA-256 hashes. These retain failures and partial trailing records; analysis excludes malformed records and lists them in summary.json.", ""]
    lines += [f"![{Path(path).stem}]({path})" for path in report["figures"]]
    return "\n".join(lines) + "\n"


def collect_input_artifacts(jobs, data_dir, output):
    """Archive exact normalized inputs, verifying each executed manifest hash.

    Public/raw archives and model weights are deliberately out of scope. A
    mismatching input is reported, never silently packaged as the executed one.
    """
    receipts, issues, candidates = [], [], {}
    data_root = data_dir.resolve()
    for job in jobs:
        identity = job.get("generation_identity") or {}
        source_name = identity.get("config", {}).get("dataset")
        expected = identity.get("dataset_sha256")
        if not source_name or not expected:
            continue
        source = Path(source_name).resolve()
        key = (source, expected)
        candidates.setdefault(key, []).append(job["job"])
    corpus_hashes = {}
    for (source, expected), names in candidates.items():
        if not source.is_relative_to(data_root) or source.suffix != ".jsonl":
            issues.append({"kind": "input_outside_normalized_data_scope", "source": str(source), "jobs": names})
            continue
        if not source.is_file():
            issues.append({"kind": "executed_input_missing", "source": str(source), "jobs": names})
            continue
        if source.stat().st_size > 128 * 1024**2:
            issues.append({"kind": "normalized_input_size_limit", "source": str(source), "jobs": names})
            continue
        raw = source.read_bytes()
        if digest(raw) != expected:
            issues.append({"kind": "executed_input_hash_mismatch", "source": str(source), "jobs": names,
                           "expected_sha256": expected, "actual_sha256": digest(raw)})
            continue
        receipt = archive_bytes(output, Path("data-inputs") / (expected[:12] + "_" + source.name + ".gz"), raw, source, True)
        receipt.update(used_by_jobs=names, verified_against_generation_manifest=True,
                       original_basename=source.name)
        receipts.append(receipt)
        for line in raw.splitlines():
            if not line.strip():
                continue
            metadata = json.loads(line).get("metadata", {})
            conversation, corpus_hash = metadata.get("conversation_id"), metadata.get("corpus_sha256")
            if conversation and corpus_hash:
                if conversation in corpus_hashes and corpus_hashes[conversation] != corpus_hash:
                    issues.append({"kind": "conflicting_corpus_hashes", "conversation": conversation})
                corpus_hashes[conversation] = corpus_hash
    for number in (1, 2, 3):
        source = data_dir / f"beam_1m_{number}_corpus.jsonl"
        if not source.is_file():
            continue
        if source.stat().st_size > 128 * 1024**2:
            issues.append({"kind": "normalized_corpus_size_limit", "source": str(source)})
            continue
        raw = source.read_bytes()
        expected = corpus_hashes.get(f"beam-1m-{number}")
        if expected is not None and digest(raw) != expected:
            issues.append({"kind": "executed_corpus_hash_mismatch", "source": str(source),
                           "expected_sha256": expected, "actual_sha256": digest(raw)})
            continue
        receipt = archive_bytes(output, Path("data-inputs") / (digest(raw)[:12] + "_" + source.name + ".gz"), raw, source, True)
        receipt.update(original_basename=source.name,
                       verified_against_normalized_input=expected is not None)
        receipts.append(receipt)
    dataset_licenses = {
        "hotpotqa": {"attribution": "Yang et al. (2018), HotpotQA", "license": "CC-BY-SA-4.0",
                      "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                      "homepage": "https://hotpotqa.github.io/",
                      "source_revision": "1908d6afbbead072334abe2965f91bd2709910ab",
                      "source_manifest": "manifest-hotpotqa.json",
                      "modifications": "Deterministic development distractor sample and JSONL schema normalization; source IDs, supplied context and support annotations retained."},
        "musique": {"attribution": "Trivedi et al. (2022), MuSiQue", "license": "CC-BY-4.0",
                     "license_url": "https://creativecommons.org/licenses/by/4.0/",
                     "homepage": "https://github.com/StonyBrookNLP/musique",
                     "source_revision": "download script 922ac98f19a201998dbdae6d7f2887a5258dbdeb; v1.0 archive pinned by SHA-256",
                     "source_manifest": "manifest-musique.json",
                     "modifications": "Deterministic answerable development sample and JSONL normalization; aliases, supplied paragraphs and support annotations retained."},
        "beam": {"attribution": "Tavakoli et al., BEAM: Beyond a Million Tokens", "license": "CC-BY-SA-4.0",
                  "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                  "homepage": "https://huggingface.co/datasets/Mohammadta/BEAM",
                  "source_revision": "b2da22eac88bb0874c64665f13457eb99835774a",
                  "source_manifest": "manifest-beam.json",
                  "modifications": "Three 1M conversation histories flattened chronologically; 60 probes normalized; retrieved evidence token-chunked and BM25-ranked; hop subset retains 30 probes."},
        "synthetic": {"attribution": "This project's eval/datasets.py, generator version 1",
                      "license": "project-generated; see repository license or obtain author permission",
                      "source_manifest": "manifest-synthetic.json",
                      "modifications": "Fictional nonce facts, links and revisions generated from recorded seeds; no personal user data."}}
    for dataset, details in dataset_licenses.items():
        manifest = read_json(data_dir / details["source_manifest"])
        details["download_receipts"] = [{k: value for k, value in source.items() if k in {"url", "sha256", "bytes"}}
                                        for source in manifest.get("sources", [])]
        details["seed"] = manifest.get("seed")
    for receipt in receipts:
        name = receipt["original_basename"]
        dataset = next((prefix for prefix in dataset_licenses if name.startswith(prefix)), None)
        receipt["data_license_key"] = dataset
        receipt["data_license"] = dataset_licenses[dataset]["license"] if dataset else "unspecified fixture or local input"
    license_manifest = {"scope": "Normalized benchmark data and corpus archives only; these licenses do not change the project code license.",
                        "restore": "Decompress each artifact and restore original_basename under data/. The uncompressed SHA-256 identifies exact executed inputs independent of generator revisions.",
                        "datasets": dataset_licenses,
                        "artifacts": [{k: value for k, value in receipt.items() if k in
                                       {"artifact", "source_sha256", "original_basename", "data_license_key", "data_license", "used_by_jobs"}}
                                      for receipt in receipts]}
    license_path = output / "artifacts" / "data-inputs" / "data-licenses.json"
    atomic_json(license_path, license_manifest)
    receipts.append({"artifact": str(license_path.relative_to(output)), "artifact_bytes": license_path.stat().st_size,
                     "artifact_sha256": digest(license_path.read_bytes()), "source": "generated per-input license manifest"})
    license_note = ("# Normalized data provenance and licenses\n\n"
                    "These are exact normalized inputs and public conversation corpora, not new benchmark releases. "
                    "Decompress files and restore each receipt's original_basename under data/ to replay. "
                    "Used inputs are checked against the immutable generation-manifest SHA-256; the manifest also records source revision, example limit, and run settings. "
                    "These licenses apply to benchmark data separately from the project code. data-licenses.json maps every archived input to its attribution, license, modifications, and source revision/download checksum.\n\n"
                    "- HotpotQA (Yang et al., 2018): CC BY-SA 4.0. https://hotpotqa.github.io/\n"
                    "- MuSiQue (Trivedi et al., 2022): CC BY 4.0. https://github.com/StonyBrookNLP/musique\n"
                    "- BEAM (Tavakoli et al., 2025/2026): CC BY-SA 4.0 for data; the code repository's MIT license does not replace the data license. https://huggingface.co/datasets/Mohammadta/BEAM\n"
                    "- Synthetic private registries, links, and revisions: generated by this project's eval/datasets.py from recorded seeds; no user's personal data. See the project license or obtain author permission for reuse.\n\n"
                    "Normalization and subsampling are modifications described in docs/data-and-protocol.md. "
                    "Per-input manifests and source URLs/revisions/checksums are included in data-provenance/. "
                    "Model weights and the large original download archives are not included.\n")
    note_path = output / "artifacts" / "data-inputs" / "DATA_LICENSES.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(license_note)
    receipts.append({"artifact": str(note_path.relative_to(output)), "artifact_bytes": note_path.stat().st_size,
                     "artifact_sha256": digest(note_path.read_bytes()), "source": "generated provenance note"})
    protocol = Path("docs/data-and-protocol.md")
    if protocol.exists():
        receipts.append(archive_bytes(output, Path("data-provenance/data-and-protocol.md"), protocol.read_bytes(), protocol))
    return receipts, issues


def collect_receipts(run_root, data_dir, matrix_path, output):
    receipts = []
    for name in ("guard-status.json", "matrix-status.json", "resources.jsonl", "amendment.json"):
        path = run_root / name
        if path.exists():
            compressed = path.suffix == ".jsonl"
            relative = Path("monitoring") / (name + ".gz" if compressed else name)
            receipts.append(archive_bytes(output, relative, path.read_bytes(), path, compressed))
    sources = sorted(set(data_dir.glob("manifest*.json")) | set(data_dir.glob("*.manifest.json")))
    for name in ("validation.json",):
        if (data_dir/name).exists():
            sources.append(data_dir/name)
    for path in sources:
        receipts.append(archive_bytes(output, Path("data-provenance") / path.name, path.read_bytes(), path))
    for relative in ("scripts/prepare_focused_e2.py", "docs/budget-amendment.md"):
        path = Path(relative)
        if path.exists():
            receipts.append(archive_bytes(output, Path("protocol-provenance") / relative, path.read_bytes(), path))
    for name in ("bm25_hashseed_audit.json", "bm25_hashseed_ledgers.json.gz", "bm25_determinism.patch", "read_timing_sidecar.patch"):
        path = Path("runs") / name
        if path.exists():
            receipts.append(archive_bytes(output, Path("reproducibility-audit") / name, path.read_bytes(), path))
    model_manifest = Path("models/model-manifest.json")
    if model_manifest.exists():
        receipts.append(archive_bytes(output, Path("model-provenance/model-manifest.json"), model_manifest.read_bytes(), model_manifest))
    if matrix_path and matrix_path.exists():
        receipts.append(archive_bytes(output, Path("config") / matrix_path.name, matrix_path.read_bytes(), matrix_path))
    helper_source = Path(__file__).with_name("report_paths.py")
    receipts.append(archive_bytes(output, Path("report-source/report_paths.py"), helper_source.read_bytes(), helper_source))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/overnight"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/overnight.json"))
    parser.add_argument("--no-matrix", action="store_true", help="Discover only existing jobs; useful for pilot reports")
    parser.add_argument("--judge-dir", type=Path, default=Path("runs/beam-judge"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/2026-09-11"))
    parser.add_argument("--include-incomplete", action="store_true", help="Include prominently marked partial results in figures/tables")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete", action="store_true", help="Exit 2 after writing if generation jobs are not all complete")
    args = parser.parse_args()
    validate_report_output(args.output_dir, args.run_root, args.judge_dir, args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = {} if args.no_matrix else read_json(args.matrix)
    jobs, records, receipts, issues = collect_jobs(args.run_root, matrix, args.output_dir)
    metrics, paired = metric_groups(records), paired_groups(records)
    judge, judge_receipts = collect_judge(args.judge_dir, records, args.output_dir)
    receipts.extend(judge_receipts)
    input_receipts, input_issues = collect_input_artifacts(jobs, args.data_dir, args.output_dir)
    receipts.extend(input_receipts)
    issues.extend(input_issues)
    source_receipts, source_issues, source_versions = collect_executable_artifacts(jobs, args.output_dir)
    receipts.extend(source_receipts)
    issues.extend(source_issues)
    receipts.extend(collect_receipts(args.run_root, args.data_dir, None if args.no_matrix else args.matrix, args.output_dir))
    complete = bool(jobs) and all(j["status"] == "complete" for j in jobs) and not input_issues and not source_issues
    figures = [] if args.no_plots else plots(metrics, args.output_dir, args.include_incomplete) + judge_plots(judge, args.output_dir)
    amendment = read_json(args.run_root / "amendment.json")
    report = {"report_version": REPORT_VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "status": "generation_complete" if complete else "INCOMPLETE_SNAPSHOT",
              "judge_status": judge["status"], "run_root": str(args.run_root),
              "report_code_sha256": digest(Path(__file__).read_bytes()),
              "jobs": jobs, "metrics": metrics, "paired": paired, "judge": judge,
              "source_versions": source_versions,
              "issues": issues, "figures": figures,
              "included_incomplete_in_headlines": args.include_incomplete,
              "raw_successful_records": len(records),
              "completion_scope": {"reported_jobs": [j["job"] for j in jobs],
                                   "reported_planned_conditions": sum(j["planned_runs"] or 0 for j in jobs),
                                   "reported_successful_conditions": sum(j["successful_unique_runs"] for j in jobs),
                                   "original_planned_conditions": amendment.get("original_planned_conditions"),
                                   "canceled_original_remainder": amendment.get("canceled_original_remainder")},
              "amendment": amendment or None,
              "resource_guard": read_json(args.run_root / "guard-status.json")}
    csv_output(args.output_dir / "metrics.csv", metrics)
    csv_output(args.output_dir / "paired.csv", paired)
    csv_output(args.output_dir / "beam-judge.csv", judge["groups"])
    atomic_json(args.output_dir / "summary.json", report)
    atomic_json(args.output_dir / "artifact-manifest.json", {"report_version": REPORT_VERSION, "artifacts": receipts})
    (args.output_dir / "tables.md").write_text(tables(report, args.include_incomplete))
    print(json.dumps({"output_dir": str(args.output_dir), "status": report["status"],
                      "jobs": len(jobs), "successful_records": len(records), "metrics": len(metrics),
                      "paired_comparisons": len(paired), "figures": len(figures), "judge_status": judge["status"]}))
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
