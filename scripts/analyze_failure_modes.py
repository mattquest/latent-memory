#!/usr/bin/env python3
"""Posthoc saved-answer diagnostics, using only stdlib and exact input datasets.

No model, tokenizer, or experiment-harness imports. Input files are never
modified. Partial trailing JSONL records are ignored and explicitly counted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


VERSION = "posthoc-failure-modes-v2-amended-scope"
RATIOS = (0.0, 0.1, 0.2, 1.0)
LIMITATIONS = [
    "Posthoc descriptive diagnostics only; these strata do not establish causal explanations.",
    "Supporting-document ID coverage does not prove that answer-bearing text survived truncation or was used.",
    "Selected truncation means evidence_ids intersect truncated_document_ids; unselected truncated documents do not count.",
    "UNKNOWN is the exact case-insensitive stripped sentinel; prose abstentions are not counted. Output-cap rate concerns final answers only.",
    "Only successful repeat-zero records enter accuracy denominators; missing conditions and errors are disclosed, not imputed.",
    "BEAM EM/F1, if present, describe lexical overlap, not its behavioral rubric score.",
    "Synthetic E2 keeps chain length, nominal budget, and actual executed evidence hops separate; evidence exhaustion is not adaptive-search failure.",
    "Executed evidence hops is the source record's nonempty evidence-batch count; direct_text still processes its selected evidence in one final prompt, not multiple agent steps.",
    "E4 paired comparisons require all four ratios and identical ordered evidence IDs for the same question and condition.",
]


def read_jsonl(path):
    """Read one byte snapshot; never truncate an in-progress source artifact."""
    content = Path(path).read_bytes()
    lines, rows, partial = content.splitlines(keepends=True), [], 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            partial = 1
    return rows, {"sha256": hashlib.sha256(content).hexdigest(),
                  "bytes_read": len(content), "ignored_partial_trailing_records": partial}


def mean(values):
    return statistics.mean(values) if values else None


def metrics(items):
    return {"n_examples": len(items),
            "exact_match": mean([x["record"]["scores"]["exact_match"] for x in items]),
            "f1": mean([x["record"]["scores"]["f1"] for x in items]),
            "unknown_rate": mean([x["unknown"] for x in items]),
            "output_cap_rate": mean([x["output_capped"] for x in items]),
            "mean_selected_documents": mean([len(x["selected_ids"]) for x in items])}


def condition(record, *, omit=()):
    return {key: value for key, value in record["case"].items()
            if key not in {"repeat", "experiment", *omit}}


def stable_key(value):
    return json.dumps(value, sort_keys=True)


def decorate(job, record, example):
    selected = list(record.get("evidence_ids", []))
    selected_set = set(selected)
    support = set(example.get("supporting_context_ids", []))
    truncated = set(record.get("truncated_document_ids", [])) & selected_set
    coverage = ("all_support_documents" if support <= selected_set else "missing_support_documents") if support else "unannotated"
    return {"job": job, "record": record, "example": example, "selected_ids": selected,
            "support_coverage": coverage, "missing_support_ids": sorted(support - selected_set),
            "selected_truncated_ids": sorted(truncated),
            "selected_support_truncated_ids": sorted(support & truncated),
            "unknown": record.get("prediction", "").strip().casefold() == "unknown",
            "output_capped": record.get("generation_stop") == "max_tokens"}


def public_e1(items):
    groups, pairs = defaultdict(list), defaultdict(dict)
    for item in items:
        record, example = item["record"], item["example"]
        if record["case"]["experiment"] != "E1" or example["dataset"].startswith("synthetic_"):
            continue
        key = stable_key({"job": item["job"], "dataset": example["dataset"], "condition": condition(record)})
        groups[key].append(item)
        case = record["case"]
        if case.get("precision") == "bf16" and case.get("control", "correct") == "correct" and case["arm"] in {"latent", "direct_text"}:
            pair_key = stable_key({"job": item["job"], "dataset": example["dataset"],
                                  "condition": condition(record, omit=("arm", "precision")),
                                  "example_id": example["id"]})
            pairs[pair_key][case["arm"]] = item
    output = []
    for key, group in sorted(groups.items()):
        strata = []
        for coverage in ("all_support_documents", "missing_support_documents", "unannotated"):
            subset = [x for x in group if x["support_coverage"] == coverage]
            if subset:
                strata.append({"support_coverage": coverage, **metrics(subset),
                               "selected_document_truncation_rate": mean([bool(x["selected_truncated_ids"]) for x in subset]),
                               "selected_support_truncation_rate": mean([bool(x["selected_support_truncated_ids"]) for x in subset])})
        truncation_strata = [{"selected_documents_truncated": value,
                              **metrics([x for x in group if bool(x["selected_truncated_ids"]) == value])}
                             for value in (False, True)]
        output.append({**json.loads(key), **metrics(group), "by_support_coverage": strata,
                       "by_selected_document_truncation": truncation_strata,
                       "truncated_examples": [{"example_id": x["example"]["id"],
                                               "selected_truncated_ids": x["selected_truncated_ids"],
                                               "selected_support_truncated_ids": x["selected_support_truncated_ids"]}
                                              for x in group if x["selected_truncated_ids"]]})
    pair_groups = defaultdict(lambda: {"n_matched": 0, "n_unpaired": 0,
                                     "n_evidence_mismatch": 0, "disagreements": []})
    for key, arms in sorted(pairs.items()):
        detail = json.loads(key)
        example_id = detail.pop("example_id")
        group = pair_groups[stable_key(detail)]
        if set(arms) != {"latent", "direct_text"}:
            group["n_unpaired"] += 1
            continue
        latent, direct = arms["latent"], arms["direct_text"]
        if latent["selected_ids"] != direct["selected_ids"]:
            group["n_evidence_mismatch"] += 1
            continue
        group["n_matched"] += 1
        lr, dr = latent["record"], direct["record"]
        if lr["prediction"] != dr["prediction"] or lr["scores"] != dr["scores"]:
            group["disagreements"].append({"example_id": example_id,
                "latent_prediction": lr["prediction"], "direct_prediction": dr["prediction"],
                "latent_scores": lr["scores"], "direct_scores": dr["scores"],
                "exact_match_disagreement": lr["scores"]["exact_match"] != dr["scores"]["exact_match"],
                "support_coverage": latent["support_coverage"],
                "missing_support_ids": latent["missing_support_ids"],
                "selected_truncated_ids": latent["selected_truncated_ids"]})
    pair_output = [{**json.loads(key), **value} for key, value in sorted(pair_groups.items())]
    return output, pair_output


def synthetic_e2(items):
    grouped = defaultdict(list)
    for item in items:
        record, example = item["record"], item["example"]
        chain = example.get("metadata", {}).get("chain_length")
        if record["case"]["experiment"] != "E2" or not example["dataset"].startswith("synthetic_") or chain is None:
            continue
        key = stable_key({"job": item["job"], "dataset": example["dataset"],
                          "condition": condition(record, omit=("hops",)), "chain_length": chain,
                          "nominal_hops": record["case"]["hops"],
                          "executed_evidence_hops": record["evidence_hops_executed"]})
        grouped[key].append(item)
    return [{**json.loads(key), **metrics(group)} for key, group in sorted(grouped.items())]


def recompute_e4(items):
    grouped = defaultdict(lambda: defaultdict(dict))
    for item in items:
        record, example = item["record"], item["example"]
        case = record["case"]
        if case["experiment"] != "E4" or float(case["bridge_ratio"]) not in RATIOS:
            continue
        key = stable_key({"job": item["job"], "dataset": example["dataset"],
                          "condition": condition(record, omit=("bridge_ratio",))})
        grouped[key][example["id"]][float(case["bridge_ratio"])] = item
    result = []
    for key, questions in sorted(grouped.items()):
        matched, incomplete, evidence_mismatch = [], [], []
        for example_id, arms in sorted(questions.items()):
            if set(arms) != set(RATIOS):
                incomplete.append(example_id)
                continue
            if any(arms[r]["selected_ids"] != arms[0.0]["selected_ids"] for r in RATIOS):
                evidence_mismatch.append(example_id)
                continue
            matched.append((example_id, arms))
        ratio_summaries = []
        for ratio in RATIOS:
            available = [arms[ratio] for arms in questions.values() if ratio in arms]
            complete = [arms[ratio] for _, arms in matched]
            ratio_summaries.append({"ratio": ratio, "available": metrics(available),
                "matched_all_four": metrics(complete),
                "paired_em_difference_vs_zero": mean([arms[ratio]["record"]["scores"]["exact_match"] -
                                                       arms[0.0]["record"]["scores"]["exact_match"] for _, arms in matched]),
                "paired_f1_difference_vs_zero": mean([arms[ratio]["record"]["scores"]["f1"] -
                                                       arms[0.0]["record"]["scores"]["f1"] for _, arms in matched])})
        result.append({**json.loads(key), "n_matched_all_four": len(matched),
                       "incomplete_question_ids": incomplete,
                       "evidence_mismatch_question_ids": evidence_mismatch,
                       "ratios": ratio_summaries,
                       "matched_questions": [{"example_id": example_id,
                           "scores_by_ratio": {str(r): {**arms[r]["record"]["scores"],
                               "bridge_recomputed_tokens": arms[r]["record"].get("counts", {}).get("bridge_recomputed_tokens"),
                               "actual_bridge_ratio": arms[r]["record"].get("cache", {}).get("metadata", {}).get("bridge_ratio_actual")}
                               for r in RATIOS}} for example_id, arms in matched]})
    return result


def completion_scope(root, jobs, guard):
    """Accept only a completed run or an exactly receipted SIGINT amendment.

    An amendment changes the reported job scope, not the original guard's
    status. Missing/unmanifested jobs and unresolved failures cannot disappear
    merely because the remaining summaries say complete.
    """
    observed = {p.name for p in root.iterdir() if p.is_dir() and any(
        (p / name).exists() for name in ("manifest.json", "results.jsonl", "summary.json", "checkpoint.json"))}
    names = {job["job"] for job in jobs}
    errors = []
    if observed != names or not jobs or not all(job["complete"] for job in jobs):
        errors.append("All observed jobs must have manifests and complete successful summaries")
    scope = {"kind": "original_run", "reported_jobs": sorted(names),
             "reported_successful_conditions": sum(job["successful_unique_runs"] for job in jobs)}
    path = root / "amendment.json"
    if not path.exists():
        if guard.get("status") != "complete" or guard.get("returncode") != 0:
            errors.append("A successful root guard is required without a validated amendment")
        return scope, None, errors

    raw = path.read_bytes()
    amendment = json.loads(raw)
    receipt = {"source": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "content": amendment}
    scope["kind"] = "amended_initial_jobs"
    if not isinstance(amendment, dict):
        return scope, receipt, errors + ["Amendment must be a JSON object"]
    completed_jobs = amendment.get("completed_jobs")
    if (not isinstance(completed_jobs, list) or not completed_jobs
            or any(not isinstance(name, str) or not name or Path(name).name != name
                   or name in {".", ".."} for name in completed_jobs)
            or len(set(completed_jobs)) != len(completed_jobs)
            or set(completed_jobs) != names or set(completed_jobs) != observed):
        errors.append("Amendment completed_jobs must exactly match the unique observed job set")
    count_keys = ("original_planned_conditions", "completed_initial_conditions", "canceled_original_remainder")
    counts = [amendment.get(key) for key in count_keys]
    scope.update({key: amendment.get(key) for key in count_keys})
    if any(type(value) is not int or value <= 0 for value in counts):
        errors.append("Amendment condition counts must be positive integers")
    else:
        original, completed, canceled = counts
        if original != completed + canceled:
            errors.append("Amendment original/completed/canceled arithmetic does not balance")
        if (completed != scope["reported_successful_conditions"]
                or any(type(job["planned_runs"]) is not int or job["planned_runs"] <= 0 for job in jobs)
                or sum(job["planned_runs"] or 0 for job in jobs) != completed):
            errors.append("Amendment completed count must match every job's planned and successful counts")
    if (amendment.get("guard_exit") != guard or guard.get("status") != "stopped"
            or guard.get("reason") != "interrupted"
            or type(guard.get("returncode")) is not int or guard["returncode"] not in {-2, 130}):
        errors.append("Amendment guard_exit must exactly preserve the interrupted SIGINT guard")
    return scope, receipt, errors


def analyze(run_dir, *, repository=None, require_complete=False):
    root = Path(run_dir).resolve()
    repository = Path(repository or Path(__file__).resolve().parents[1])
    items, jobs = [], []
    manifests = sorted(root.glob("*/manifest.json"))
    if not manifests:
        raise ValueError(f"No per-job manifests under {root}")
    for manifest_path in manifests:
        job_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        identity = manifest["identity"]
        dataset = Path(identity["config"]["dataset"])
        dataset = dataset if dataset.is_absolute() else repository / dataset
        data_rows, data_snapshot = read_jsonl(dataset)
        if data_snapshot["sha256"] != identity["dataset_sha256"]:
            raise ValueError(f"Dataset checksum mismatch for {job_dir.name}: {dataset}")
        if data_snapshot["ignored_partial_trailing_records"]:
            raise ValueError(f"Incomplete dataset: {dataset}")
        examples = {row["id"]: row for row in data_rows}
        if len(examples) != len(data_rows):
            raise ValueError(f"Duplicate dataset IDs: {dataset}")
        results_path = job_dir / "results.jsonl"
        rows, snapshot = read_jsonl(results_path) if results_path.exists() else ([], {})
        latest_all = {row["id"]: row for row in rows}
        latest = {key: row for key, row in latest_all.items() if row.get("status") == "ok"}
        unresolved_errors = len(latest_all) - len(latest)
        selected = [row for row in latest.values() if row["case"].get("repeat", 0) == 0]
        for row in selected:
            example = examples[row["example_id"]]
            if row["dataset"] != example["dataset"]:
                raise ValueError(f"Result/dataset mismatch for {row['id']}")
            items.append(decorate(job_dir.name, row, example))
        summary_path = job_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        complete = bool(summary and summary.get("stop_reason") == "complete"
                        and summary.get("planned_runs") == len(latest)
                        and not unresolved_errors
                        and all(summary.get(key, len(latest)) == len(latest)
                                for key in ("completed_unique_runs", "n_successful_runs"))
                        and not snapshot.get("ignored_partial_trailing_records"))
        jobs.append({"job": job_dir.name, "dataset": str(dataset.resolve()),
                     "dataset_sha256": data_snapshot["sha256"], "results_snapshot": snapshot,
                     "successful_unique_runs": len(latest), "repeat_zero_records": len(selected),
                     "error_records": sum(row.get("status") != "ok" for row in rows),
                     "unresolved_error_records": unresolved_errors,
                     "planned_runs": summary.get("planned_runs"), "complete": complete})
    guard_path = root / "guard-status.json"
    guard = json.loads(guard_path.read_text()) if guard_path.exists() else {}
    scope, amendment, completion_errors = completion_scope(root, jobs, guard)
    complete = not completion_errors
    if require_complete and not complete:
        raise ValueError("Run is incomplete: " + "; ".join(completion_errors))
    e1, disagreements = public_e1(items)
    return {"analysis_version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
            "completeness": "complete" if complete else "incomplete",
            "completion_scope": scope, "completion_errors": completion_errors,
            "amendment_receipt": amendment, "resource_guard": guard,
            "diagnostic_role": "posthoc_descriptive_not_causal", "jobs": jobs,
            "public_e1": e1, "public_e1_bf16_latent_vs_direct": disagreements,
            "synthetic_e2": synthetic_e2(items), "e4_recompute": recompute_e4(items),
            "limitations": LIMITATIONS}


def markdown(report):
    pct = lambda x: "—" if x is None else f"{100*x:.1f}%"
    arm = lambda x: f"{x['condition']['arm']} / {x['condition'].get('precision', 'unspecified')}"
    lines = ["# Posthoc failure-mode diagnostics", "", f"**Snapshot: {report['completeness'].upper()}.** "
             "Descriptive diagnostics only; no causal attribution."]
    scope = report.get("completion_scope", {})
    if scope.get("kind") == "amended_initial_jobs":
        lines += ["", f"Amended initial-job scope: {scope.get('reported_successful_conditions')} successful conditions; "
                  f"the original plan had {scope.get('original_planned_conditions')} and "
                  f"{scope.get('canceled_original_remainder')} were canceled. "
                  "Canceled conditions are not completed results. The preserved root guard remains interrupted; "
                  "the amendment and exact job/count checks determine this snapshot's completion status."]
    if report.get("completion_errors"):
        lines += ["", "Completion checks: " + "; ".join(report["completion_errors"])]
    lines += ["", "## Public E1 by support-document coverage", "",
             "| Job | Arm | Nominal hops | Support coverage | N | EM | F1 | Selected truncation | UNKNOWN | Output cap |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in report["public_e1"]:
        for s in group["by_support_coverage"]:
            lines.append(f"| {group['job']} | {arm(group)} | {group['condition']['hops']} | {s['support_coverage']} | "
                         f"{s['n_examples']} | {pct(s['exact_match'])} | {pct(s['f1'])} | "
                         f"{pct(s['selected_document_truncation_rate'])} | {pct(s['unknown_rate'])} | {pct(s['output_cap_rate'])} |")
    lines += ["", "Supporting IDs do not establish that the answer text survived truncation. JSON includes separate truncation strata and intersected document IDs.",
              "", "## Matched BF16 latent versus direct text", ""]
    for group in report["public_e1_bf16_latent_vs_direct"]:
        ids = [x["example_id"] for x in group["disagreements"] if x["exact_match_disagreement"]]
        lines.append(f"- {group['job']}, {group['condition']['hops']} nominal hops: {group['n_matched']} matched pairs; "
                     f"{group['n_unpaired']} unpaired; {group['n_evidence_mismatch']} evidence mismatches. "
                     f"EM disagreement IDs: {', '.join(ids) or 'none'}. Prediction/F1 differences are retained in JSON.")
    lines += ["", "## Synthetic E2: chain length and actual execution", "",
              "| Job | Arm | Chain length | Nominal hops | Executed evidence hops | N | EM | F1 | UNKNOWN | Output cap |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in report["synthetic_e2"]:
        lines.append(f"| {g['job']} | {arm(g)} | {g['chain_length']} | {g['nominal_hops']} | {g['executed_evidence_hops']} | "
                     f"{g['n_examples']} | {pct(g['exact_match'])} | {pct(g['f1'])} | {pct(g['unknown_rate'])} | {pct(g['output_cap_rate'])} |")
    lines += ["", "## E4: same questions at all four ratios", "",
              "| Job | Nominal hops | Ratio | Matched N | EM | F1 | EM difference vs 0 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in report["e4_recompute"]:
        for s in g["ratios"]:
            lines.append(f"| {g['job']} | {g['condition']['hops']} | {s['ratio']} | {g['n_matched_all_four']} | "
                         f"{pct(s['matched_all_four']['exact_match'])} | {pct(s['matched_all_four']['f1'])} | {pct(s['paired_em_difference_vs_zero'])} |")
    lines += ["", "Available-arm denominators, incomplete questions, and matched per-question scores are retained in JSON.",
              "", "## Limits", "", *[f"- {text}" for text in report["limitations"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_dir, require_complete=args.require_complete)
    output = args.output_dir.resolve()
    run_root = args.run_dir.resolve()
    if output == run_root or run_root in output.parents:
        raise ValueError("Keep diagnostic outputs outside the source run directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("failure-modes.json", json.dumps(report, indent=2, sort_keys=True) + "\n"),
                          ("failure-modes.md", markdown(report))):
        temporary = output / (name + ".tmp")
        temporary.write_text(content)
        temporary.replace(output / name)
    print(json.dumps({"completeness": report["completeness"], "jobs": len(report["jobs"]),
                      "output_dir": str(output)}))


if __name__ == "__main__":
    main()
