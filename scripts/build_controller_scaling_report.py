#!/usr/bin/env python3
"""Strict CPU-only report of a paired model-size / controller-reasoning matrix.

The matrix declares every expected job and generation setting before execution.
No partial, duplicated, misidentified, or unverifiable ledger contributes headline
metrics. Supporting IDs are read only here, after inference, to measure retrieval
productivity; annotation exposure is not proof that the model used the evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from eval.adaptive_experiments import _paired_statistics, aggregate
from scripts.build_adaptive_report import archive, csv_rows, inspect_run, load_json, safe_source, sha
from scripts.report_paths import validate_report_output

VERSION = "controller-scaling-report-v1"
MATRIX_VERSION = "controller-scaling-matrix-v1"
ARMS = {"basic_rag", "iterative_text", "incremental_kv_bf16", "relay_kv_bf16"}
OPERATIONAL = {"output_dir", "resume", "max_runtime_seconds", "max_case_seconds"}
OLD_PROTOCOL = "adaptive-global-bm25-cold-v1"
CONTROLLER_PROTOCOL = "adaptive-global-bm25-controller-v2"
COUNT_FIELDS = ("controller_reasoning_tokens", "controller_action_tokens", "controller_forced_tokens")


def protocol_for(budget, controller_temperature=0.0, controller_top_p=1.0, controller_top_k=0):
    return (OLD_PROTOCOL if (budget, controller_temperature, controller_top_p, controller_top_k) == (0, 0.0, 1.0, 0)
            else CONTROLLER_PROTOCOL)


def expected_protocol(budget, config):
    return protocol_for(budget, *(config.get(key, default) for key, default in
                        (("controller_temperature", 0.0), ("controller_top_p", 1.0), ("controller_top_k", 0))))


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def shared_configuration(config):
    return {key: value for key, value in config.items()
            if key not in OPERATIONAL | {"max_reasoning_tokens"}}


def read_matrix(path, root=REPOSITORY):
    matrix = json.loads(Path(path).read_text())
    if matrix.get("schema_version") != MATRIX_VERSION or matrix.get("cohort") not in {"development", "frozen_test"}:
        raise ValueError("Declare the controller-scaling matrix schema and development/frozen_test cohort")
    jobs = matrix.get("jobs", [])
    if len(jobs) != 4 or len({job.get("id") for job in jobs}) != 4:
        raise ValueError("Matrix must declare exactly four unique model × reasoning jobs")
    labels = {job.get("model_label") for job in jobs}
    if len(labels) != 2 or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Declare exactly two model labels")
    if {(job.get("model_label"), job.get("max_reasoning_tokens")) for job in jobs} != set(itertools.product(labels, (0, 256))):
        raise ValueError("Each model must have reasoning budgets 0 and 256")
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]):
            raise ValueError("Job IDs must be simple safe directory names")
        if not job.get("model_revision") or not job.get("model_path"):
            raise ValueError("Each job must pin its model path and revision")
        protocol = expected_protocol(job["max_reasoning_tokens"], matrix.get("shared_config", {}))
        if job.get("protocol", protocol) != protocol:
            raise ValueError("Job protocol does not match its reasoning budget")
        safe_source(root, job["run_dir"])
        safe_source(root, job["model_path"])
    directories = [safe_source(root, job["run_dir"]) for job in jobs]
    if len(set(directories)) != len(directories):
        raise ValueError("Each job must have a distinct source directory")
    shared = matrix.get("shared_config", {})
    arms = shared.get("arms", [])
    if arms != ["iterative_text"] and (set(arms) != ARMS or len(arms) != 4):
        raise ValueError("Declare either the iterative_text pilot or the four comparison arms in shared_config")
    if shared_configuration(shared) != shared:
        raise ValueError("shared_config must contain generation settings only; omit operational/reasoning settings")
    for field in ("dataset", "corpus"):
        if matrix.get(field) != shared.get(field):
            raise ValueError(f"Matrix {field} must exactly match shared_config")
    for label in labels:
        same_model = [job for job in jobs if job["model_label"] == label]
        if len({(job["model_path"], job["model_revision"]) for job in same_model}) != 1:
            raise ValueError("A model label must refer to one pinned checkpoint across reasoning budgets")
    if len({(job["model_path"], job["model_revision"]) for job in jobs}) != 2:
        raise ValueError("Two model labels must refer to different pinned checkpoints")
    for field in ("questions_sha256", "corpus_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", matrix.get(field, "")):
            raise ValueError(f"Matrix requires a pinned {field}")
    if not matrix.get("data_manifest"):
        raise ValueError("Matrix requires a data-selection provenance manifest")
    return matrix


def inspect_trace(row, question, config, corpus_ids):
    """Independently derive support discovery and verify trace/token accounting."""
    support = set(question.get("supporting_context_ids", []))
    if not support or not support <= corpus_ids:
        raise ValueError("Every evaluation question requires valid nonempty supporting IDs")
    if row.get("category") != question.get("category") or row.get("split") != question.get("split"):
        raise ValueError("Result category/split differs from the frozen question")
    if row.get("protocol_version") != expected_protocol(config["max_reasoning_tokens"], config):
        raise ValueError("Row protocol differs from its reasoning budget")
    expected_controller = {name: config[name] for name in ("max_reasoning_tokens", "max_action_tokens", "max_answer_tokens",
                                                          "controller_temperature", "controller_top_p", "controller_top_k")}
    if row.get("controller_config") != expected_controller:
        raise ValueError("Row controller budgets differ from the declared job")
    counts = row["counts"]
    if any(not finite_nonnegative(value) for value in counts.values()) or any(int(v) != v for v in counts.values()):
        raise ValueError("Token and work counts must be finite nonnegative integers")
    if any(not finite_nonnegative(value) for value in row["component_ms"].values()):
        raise ValueError("Component timings must be finite and nonnegative")
    if abs(sum(row["component_ms"].values()) - row["cold_end_to_end_ms"]) > max(.1, row["cold_end_to_end_ms"] * .001):
        raise ValueError("Component timings do not account for the end-to-end duration")
    if counts["max_model_context_tokens"] > config["max_model_context_tokens"]:
        raise ValueError("Recorded model context exceeds the declared cap")
    expected_prefill = sum(counts[key] for key in ("controller_prefill_tokens", "final_prefill_tokens",
                           "cache_prefill_tokens", "bridge_recomputed_tokens", "controller_forced_tokens"))
    if counts["total_prefill_tokens"] != expected_prefill or counts["primary_model_prefill_tokens"] != expected_prefill:
        raise ValueError("Prefill work omits or double-counts ordinary, cache, bridge, or forced-control inputs")
    if counts["all_models_prefill_tokens"] != expected_prefill + counts["reranker_input_tokens"]:
        raise ValueError("All-model prefill work differs from primary plus reranker work")
    if len(row["output_token_ids"]) != counts["host_output_tokens"] or counts["host_output_tokens"] > config["max_answer_tokens"]:
        raise ValueError("Host output token accounting or cap mismatch")
    events = row.get("trace")
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError("Missing or malformed retrieval/controller trace")
    if any(event.get("event") not in {"retrieval", "decision"} for event in events):
        raise ValueError("Unexpected trace event for the declared scaling arms")
    retrieved = [event for event in events if event.get("event") == "retrieval"]
    decisions = [event for event in events if event.get("event") in {"decision", "query_expansion"}]
    if len(retrieved) != row["executed_retrieval_rounds"] or not 1 <= len(retrieved) <= row["case"]["max_rounds"]:
        raise ValueError("Executed retrieval rounds do not match trace or cap")
    if counts["retrieval_calls"] != len(retrieved) or counts["controller_calls"] != len(decisions):
        raise ValueError("Retrieval/controller call counts differ from trace")
    if row["invalid_actions"] != sum(not event["action"]["valid"] for event in decisions):
        raise ValueError("Invalid-action count differs from controller decisions")
    seen, steps = [], []
    for expected_round, event in enumerate(retrieved, 1):
        new = event.get("new_document_ids", [])
        hits = [entry[0] for entry in event.get("hit_ids_and_scores", [])]
        if event.get("round") != expected_round or len(new) != len(set(new)) or set(new) & set(seen):
            raise ValueError("Retrieval sequence is not ordered or repeats newly added IDs")
        if not set(new) <= corpus_ids or not set(new) <= set(hits):
            raise ValueError("New evidence IDs are absent from corpus or retrieved candidates")
        if len(new) > (config["initial_top_k"] if expected_round == 1 else config["documents_per_round"]):
            raise ValueError("Per-round document budget exceeded")
        seen.extend(new)
        if event.get("accumulated_document_ids") != seen:
            raise ValueError("Accumulated trace evidence differs from ordered new IDs")
        if not finite_nonnegative(event.get("evidence_tokens")) or event["evidence_tokens"] > config["max_evidence_tokens"]:
            raise ValueError("Trace evidence exceeds declared token budget")
        discovered = sorted(support & set(new))
        steps.append({"round": expected_round, "new_document_ids": new,
                      "new_support_ids": discovered, "new_support_count": len(discovered),
                      "cumulative_support_count": len(support & set(seen)),
                      "support_annotation_coverage": len(support & set(seen)) / len(support),
                      "evidence_tokens": event["evidence_tokens"]})
    expected_document_cap = config["initial_top_k"] if row["case"]["arm"] == "basic_rag" else config["max_documents"]
    if row.get("evidence_ids") != seen or counts["retrieved_documents"] != len(seen) or len(seen) > expected_document_cap:
        raise ValueError("Final evidence IDs/count/cap differ from retrieval trace")
    if counts["retrieved_evidence_tokens"] != retrieved[-1]["evidence_tokens"]:
        raise ValueError("Final evidence token count differs from retrieval trace")
    coverage = len(support & set(seen)) / len(support)
    if row.get("support_annotation_coverage") != coverage:
        raise ValueError("Support annotation coverage differs from frozen annotations and delivered IDs")
    reasoning = action = forced = sampled = 0
    reasoning_stops, action_stops = Counter(), Counter()
    for event in decisions:
        seed_payload = [config["seed"], row["example_id"], event["event"], event.get("round", 0)]
        expected_seed = int(sha(json.dumps(seed_payload, sort_keys=True).encode())[:8], 16)
        if event.get("controller_seed") != expected_seed:
            raise ValueError("Controller sampling seed differs from the shared per-question/per-decision seed")
        decode = event.get("controller_decode", {})
        if decode.get("controller_seed") != expected_seed:
            raise ValueError("Nested controller seed differs from the per-decision seed")
        if decode.get("context_reservation_tokens", config["max_model_context_tokens"] + 1) > config["max_model_context_tokens"]:
            raise ValueError("Controller context reservation exceeds the declared model cap")
        if (config["controller_temperature"], config["controller_top_p"], config["controller_top_k"]) != (0.0, 1.0, 0):
            expected_sampling = {"temperature": config["controller_temperature"], "top_p": config["controller_top_p"],
                                 "top_k": config["controller_top_k"], "seed": expected_seed}
            if decode.get("sampling") != expected_sampling:
                raise ValueError("Executed controller sampler differs from declared temperature/filter/seed")
        generated_reasoning = decode["reasoning_output_token_ids"]
        generated_action = decode["action_token_ids"]
        forced_control = decode["forced_control_token_ids"]
        sampled_reasoning = decode["reasoning_sampled_token_ids"]
        sampled_action = decode["action_sampled_token_ids"]
        if sampled_reasoning[:len(generated_reasoning)] != generated_reasoning or sampled_action[:len(generated_action)] != generated_action:
            raise ValueError("Sampled token IDs do not retain the recorded generated tokens")
        expected_sampled = len(sampled_reasoning) + len(sampled_action)
        if decode["sampled_tokens"] != expected_sampled:
            raise ValueError("Sampled token count differs from reasoning/action sampling IDs including EOS")
        if len(sampled_reasoning) - len(generated_reasoning) != int(decode["reasoning_stop_reason"] == "eos"):
            raise ValueError("Reasoning EOS sampling count differs from its stop reason")
        if len(sampled_action) - len(generated_action) != int(decode["action_stop_reason"] == "eos"):
            raise ValueError("Action EOS sampling count differs from its stop reason")
        if generated_action != event.get("action_token_ids"):
            raise ValueError("Trace action token IDs differ from controller decode")
        lengths = {"reasoning_generated_tokens": len(generated_reasoning),
                   "action_generated_tokens": len(generated_action), "forced_control_tokens": len(forced_control)}
        if any(decode[key] != value for key, value in lengths.items()):
            raise ValueError("Controller token counts differ from recorded token IDs")
        if len(generated_reasoning) > config["max_reasoning_tokens"] or len(generated_action) > config["max_action_tokens"]:
            raise ValueError("Controller generation exceeded reasoning/action cap")
        reasoning += len(generated_reasoning)
        action += len(generated_action)
        forced += len(forced_control)
        sampled += expected_sampled
        reasoning_stops[decode["reasoning_stop_reason"]] += 1
        action_stops[decode["action_stop_reason"]] += 1
    if [counts[key] for key in COUNT_FIELDS] != [reasoning, action, forced]:
        raise ValueError("Aggregate reasoning/action/forced token counts differ from trace")
    if counts["internal_decoded_tokens"] != reasoning + action:
        raise ValueError("Internal decoded token count differs from reasoning plus action")
    if counts["controller_sampled_tokens_including_eos"] != sampled:
        raise ValueError("Aggregate sampling count differs from controller trace including EOS")
    return {"example_id": row["example_id"], "support_ids": sorted(support), "steps": steps,
            "support_count": len(support), "initial_support_count": steps[0]["new_support_count"],
            "final_support_count": len(support & set(seen)),
            "new_support_after_initial": sum(step["new_support_count"] for step in steps[1:]),
            "productive_followup_rounds": sum(step["new_support_count"] > 0 for step in steps[1:]),
            "reasoning_stops": dict(reasoning_stops), "action_stops": dict(action_stops)}


def inspect_matrix(matrix, root=REPOSITORY):
    issues, inspections, productivity = [], {}, {}
    shared = matrix["shared_config"]
    for job in matrix["jobs"]:
        run_dir = safe_source(root, job["run_dir"])
        try:
            inspection = inspect_run(run_dir, root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({"job": job["id"], "kind": "unreadable_run", "error": str(exc)})
            continue
        inspections[job["id"]] = inspection
        identity = inspection["manifest"].get("identity", {})
        actual = identity.get("config", {})
        expected = {**shared, "max_reasoning_tokens": job["max_reasoning_tokens"]}
        for condition, kind in (
            (inspection["status"] == "complete", "incomplete_or_unverified_job"),
            (inspection["duplicate_record_ids"] == 0, "duplicate_result_rows"),
            (inspection["historical_error_records"] == 0, "historical_error_records"),
            (inspection["runner_summary"].get("planned_runs") == inspection["planned_unique_runs"], "runner_planned_count_mismatch"),
            (shared_configuration(actual) == shared, "shared_generation_configuration_mismatch"),
            (actual.get("max_reasoning_tokens") == job["max_reasoning_tokens"], "reasoning_budget_mismatch"),
            (identity.get("protocol") == expected_protocol(job["max_reasoning_tokens"], shared), "protocol_mismatch"),
            (identity.get("questions_sha256") == matrix["questions_sha256"], "matrix_question_hash_mismatch"),
            (identity.get("corpus_sha256") == matrix["corpus_sha256"], "matrix_corpus_hash_mismatch"),
            (identity.get("backend", {}).get("model_revision") == job["model_revision"], "model_revision_mismatch"),
            (Path(identity.get("backend", {}).get("model_path", "")).resolve() == safe_source(root, job["model_path"]), "model_path_mismatch"),
        ):
            if not condition:
                issues.append({"job": job["id"], "kind": kind})
        issues.extend({"job": job["id"], **issue} for issue in inspection["issues"])
        verified = inspection["verified_inputs"]
        if not {"dataset", "corpus"} <= set(verified):
            continue
        questions = [json.loads(line) for line in verified["dataset"]["raw"].splitlines() if line.strip()][:shared["limit"]]
        if len(questions) != shared["limit"] or len({q["id"] for q in questions}) != len(questions):
            issues.append({"job": job["id"], "kind": "question_cohort_count_or_unique_id_mismatch"})
        corpus_ids = {json.loads(line)["id"] for line in verified["corpus"]["raw"].splitlines() if line.strip()}
        question_map = {q["id"]: q for q in questions}
        productivity[job["id"]] = {}
        for row in inspection["records"]:
            if row.get("status") != "ok":
                continue
            try:
                detail = inspect_trace(row, question_map[row["example_id"]], expected, corpus_ids)
                productivity[job["id"]][row["id"]] = detail
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                issues.append({"job": job["id"], "kind": "trace_or_budget_verification_failed", "id": row["id"], "error": str(exc)})
    # The same executable files must implement every condition; model weights and
    # explicit reasoning policy are the only model/controller differences allowed.
    source_maps = [entry["manifest"].get("identity", {}).get("source_sha256") for entry in inspections.values()]
    if source_maps and any(source_map != source_maps[0] for source_map in source_maps[1:]):
        issues.append({"kind": "cross_job_executable_source_mismatch"})
    return inspections, productivity, issues


def compute_results(matrix, inspections, productivity):
    metrics, trajectories, details, grouped = [], [], [], {}
    for job in matrix["jobs"]:
        records = inspections[job["id"]]["records"]
        metadata = {"job_id": job["id"], "model_label": job["model_label"], "max_reasoning_tokens": job["max_reasoning_tokens"]}
        for metric in aggregate(records, matrix["shared_config"]["seed"])["metrics"]:
            selected = [row for row in records if row["case"] == {"arm": metric["arm"], "max_rounds": metric["max_rounds"]}
                        and (metric["category"] == "all" or row.get("category") == metric["category"])]
            data = [productivity[job["id"]][row["id"]] for row in selected]
            metric.update(metadata,
                correct=sum(row["scores"]["exact_match"] for row in selected),
                mean_new_support_after_initial=statistics.mean(item["new_support_after_initial"] for item in data),
                fraction_questions_gain_new_support=statistics.mean(item["new_support_after_initial"] > 0 for item in data),
                mean_productive_followup_rounds=statistics.mean(item["productive_followup_rounds"] for item in data),
                fraction_all_support_delivered=statistics.mean(item["final_support_count"] == item["support_count"] for item in data),
                max_recorded_mlx_peak_bytes=max(row.get("memory", {}).get("peak_bytes", 0) for row in selected),
                controller_reasoning_stops=dict(sum((Counter(item["reasoning_stops"]) for item in data), Counter())),
                controller_action_stops=dict(sum((Counter(item["action_stops"]) for item in data), Counter())))
            metrics.append(metric)
            if metric["category"] != "all":
                continue
            key = (job["id"], metric["arm"], metric["max_rounds"])
            grouped[key] = {row["example_id"]: row for row in selected}
            for row, detail in zip(selected, data):
                details.append({**metadata, "arm": metric["arm"], "max_rounds": metric["max_rounds"], **detail})
            for round_number in range(1, metric["max_rounds"] + 1):
                reached = [item["steps"][round_number - 1] for item in data if len(item["steps"]) >= round_number]
                cumulative = [item["steps"][min(round_number, len(item["steps"])) - 1] for item in data]
                decisions = [(row, event) for row in selected for event in row["trace"]
                             if event.get("event") == "decision" and event["round"] == round_number]
                def canonical(query):
                    return " ".join(re.findall(r"\w+", query.lower()))
                repeated = sum(event["action"]["kind"] == "search" and
                    canonical(event["action"]["query"]) in {canonical(query) for query in row["search_queries"][:round_number]}
                    for row, event in decisions)
                trajectories.append({**metadata, "arm": metric["arm"], "max_rounds": metric["max_rounds"],
                    "round": round_number, "n_questions": len(data), "n_reached_round": len(reached),
                    "n_decisions_after_round": len(decisions),
                    "n_repeated_query_decisions": repeated,
                    "n_invalid_decisions": sum(not event["action"]["valid"] for _, event in decisions),
                    "n_answer_decisions": sum(event["action"]["kind"] == "answer" for _, event in decisions),
                    "n_stopped_after_round": sum(row["executed_retrieval_rounds"] == round_number for row in selected),
                    "reasoning_tokens_after_round": sum(event["controller_decode"]["reasoning_generated_tokens"] for _, event in decisions),
                    "action_tokens_after_round": sum(event["controller_decode"]["action_generated_tokens"] for _, event in decisions),
                    "new_support_count": sum(step["new_support_count"] for step in reached),
                    "mean_new_support_per_original_question": sum(step["new_support_count"] for step in reached) / len(data),
                    "mean_new_support_per_reached_question": statistics.mean(step["new_support_count"] for step in reached) if reached else None,
                    "fraction_original_questions_with_new_support": sum(step["new_support_count"] > 0 for step in reached) / len(data),
                    "mean_cumulative_support_coverage": statistics.mean(step["support_annotation_coverage"] for step in cumulative),
                    "fraction_all_support_delivered": statistics.mean(step["cumulative_support_count"] == item["support_count"] for step, item in zip(cumulative, data))})
    comparisons = []
    jobs = {job["id"]: job for job in matrix["jobs"]}
    for comparator, target in itertools.combinations(grouped, 2):
        same_job = comparator[0] == target[0]
        same_arm_budget = comparator[1:] == target[1:]
        if not same_job and not same_arm_budget:
            continue
        # Within a job compare methods at the same round cap, and compare each
        # larger iterative cap against smaller caps of that same method.
        if same_job and comparator[1] != target[1] and comparator[2] != target[2] and "basic_rag" not in (comparator[1], target[1]):
            continue
        left, right = grouped[comparator], grouped[target]
        if set(left) != set(right):
            raise ValueError("Paired comparison lost exact question identity matching")
        pairs = [(left[key], right[key]) for key in sorted(left)]
        stats = _paired_statistics(pairs, matrix["shared_config"]["seed"])
        count_keys = set.intersection(*(set(row["counts"]) for row in [*left.values(), *right.values()]))
        stats["mean_count_differences"] = {key: statistics.mean(b["counts"][key] - a["counts"][key] for a, b in pairs) for key in sorted(count_keys)}
        stats["identical_predictions"] = sum(a["prediction"] == b["prediction"] for a, b in pairs)
        stats["identical_output_token_ids"] = sum(a["output_token_ids"] == b["output_token_ids"] for a, b in pairs)
        stats["identical_evidence_ids"] = sum(a["evidence_ids"] == b["evidence_ids"] for a, b in pairs)
        stats["identical_search_queries"] = sum(a["search_queries"] == b["search_queries"] for a, b in pairs)
        stats["mean_new_support_after_initial_difference"] = statistics.mean(
            productivity[target[0]][b["id"]]["new_support_after_initial"] -
            productivity[comparator[0]][a["id"]]["new_support_after_initial"] for a, b in pairs)
        stats["mean_productive_followup_rounds_difference"] = statistics.mean(
            productivity[target[0]][b["id"]]["productive_followup_rounds"] -
            productivity[comparator[0]][a["id"]]["productive_followup_rounds"] for a, b in pairs)
        left_job, right_job = jobs[comparator[0]], jobs[target[0]]
        family = "within_job"
        if not same_job:
            family = ("reasoning_budget" if left_job["model_label"] == right_job["model_label"] else
                      "model_size" if left_job["max_reasoning_tokens"] == right_job["max_reasoning_tokens"] else "model_and_reasoning")
        comparisons.append({"comparison_family": family,
            "comparator_job": comparator[0], "comparator_arm": comparator[1], "comparator_max_rounds": comparator[2],
            "target_job": target[0], "target_arm": target[1], "target_max_rounds": target[2], **stats})
    return metrics, trajectories, details, comparisons


LIMITATIONS = [
    "This is a paired fixed subset of MuSiQue's development-source questions, not a population sample of all large-store queries. Development cohorts cannot establish held-out gains.",
    "Nominal question-level bootstrap/Wilson intervals are descriptive and unadjusted for multiple comparisons or shared source facts; they do not establish population superiority.",
    "New supporting IDs measure annotation exposure after inference. A delivered paragraph may be truncated, and exposure does not prove understanding, use, or causal necessity.",
    "Early stops contribute zero new support at later rounds; cumulative coverage is carried forward. Reached-round productivity has a selected denominator and is descriptive.",
    "Each configured round budget runs a separate adaptive trajectory. Accuracy at intermediate steps is not observed unless a separate run ends at that budget.",
    "Cold query latency includes retrieval, controller reasoning/action decoding, per-question tokenization, cache construction and final generation. Model loading and offline index setup are separate.",
    "Token work, sampled RSS and MLX allocator peaks are resource measures, not dollar cost or measured energy. Peak allocator values may carry earlier jobs' high-water marks within a process.",
    "Each model uses its own freshly constructed KV caches. Checkpoints have incompatible caches; no cross-model cache sharing or transfer is evaluated.",
    "Reasoning remains decoded text; there is no trained latent query head, learned compressor, or ingestion-precomputed whole-corpus KV store in this experiment.",
]


def plots(metrics, trajectories, output, cohort):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    colors = {"basic_rag": "#374151", "iterative_text": "#2563eb", "incremental_kv_bf16": "#059669", "relay_kv_bf16": "#7c3aed"}
    jobs = list(dict.fromkeys(row["job_id"] for row in metrics))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, squeeze=False)
    for axis, job in zip(axes.flat, jobs):
        for arm in sorted(ARMS):
            rows = sorted((row for row in metrics if row["job_id"] == job and row["arm"] == arm and row["category"] == "all"), key=lambda row: row["max_rounds"])
            if not rows:
                continue
            axis.plot([row["cold_p50_ms"] / 1000 for row in rows], [row["exact_match"] for row in rows], "o-", color=colors[arm], label=arm)
            for row in rows:
                axis.annotate(str(row["max_rounds"]), (row["cold_p50_ms"] / 1000, row["exact_match"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        axis.set(title=job, xlabel="Cold query p50 (seconds)", ylabel="Exact match", ylim=(-.025, 1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.tick_params(axis="both", labelleft=True, labelbottom=True)
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Controller scaling · {cohort} · labels are maximum retrieval rounds")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(directory / "accuracy_latency.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True, squeeze=False)
    for axis, job in zip(axes.flat, jobs):
        for arm in sorted(ARMS):
            candidates = [row for row in trajectories if row["job_id"] == job and row["arm"] == arm]
            if not candidates:
                continue
            cap = max(row["max_rounds"] for row in candidates)
            rows = sorted((row for row in candidates if row["max_rounds"] == cap), key=lambda row: row["round"])
            axis.plot([row["round"] for row in rows], [row["mean_cumulative_support_coverage"] for row in rows], "o-", color=colors[arm], label=arm)
        axis.set(title=job, xlabel="Retrieval step in largest-budget trajectory", ylabel="Mean annotation coverage", ylim=(-.025, 1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.tick_params(axis="y", labelleft=True)
        axis.grid(alpha=.2)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Evidence discovery · all questions retained after early stops")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(directory / "support_trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for job in jobs:
        candidates = [row for row in trajectories if row["job_id"] == job and row["arm"] == "iterative_text"]
        cap = max(row["max_rounds"] for row in candidates)
        rows = sorted((row for row in candidates if row["max_rounds"] == cap), key=lambda row: row["round"])
        steps = [row["round"] for row in rows]
        axes[0].plot(steps, [row["mean_new_support_per_original_question"] for row in rows], "o-", label=job)
        axes[1].plot(steps, [row["n_repeated_query_decisions"] / row["n_questions"] for row in rows], "o-", label=job)
    axes[0].set(ylabel="New supporting IDs per original question", title="New evidence exposed at each step")
    axes[1].set(ylabel="Fraction of original questions", title="Repeated search decisions after each step", ylim=(-.025, 1.025))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.set_xlabel("Retrieval step · iterative text, largest round budget")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Productive search and repetition · early-stopped questions remain in denominator")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(directory / "search_productivity.png", dpi=160)
    plt.close(fig)
    return ["figures/accuracy_latency.png", "figures/support_trajectories.png", "figures/search_productivity.png"]


def tables(summary):
    lines = [f"# Controller scaling — {summary['cohort']}", "", "Every headline condition passed exact source, input, job, score and trace verification.", "",
             "Cold timings include all online query work. Loading and index setup are separate in `summary.json`. Token counts are measured work, not dollar prices.", "",
             "| Job | Arm | Round cap | Correct / n | F1 | Cold p50 / p95 (s) | Mean reasoning / action / host tokens | New support after first retrieval |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        if row["category"] != "all":
            continue
        counts = row["mean_counts"]
        lines.append(f"| {row['job_id']} | {row['arm']} | {row['max_rounds']} | {int(row['correct'])} / {row['n']} | {100*row['f1']:.2f}% | {row['cold_p50_ms']/1000:.3f} / {row['cold_p95_ms']/1000:.3f} | {counts['controller_reasoning_tokens']:.1f} / {counts['controller_action_tokens']:.1f} / {counts['host_output_tokens']:.1f} | {row['mean_new_support_after_initial']:.3f} |")
    lines += ["", "Detailed paired results are in [pairs.csv](pairs.csv); budget and model contrasts keep exactly the same question IDs. Each difference is target minus comparator and each latency ratio is target divided by comparator.", "",
              "Round-by-round full-cohort and reached-round denominators are in [trajectories.csv](trajectories.csv). Per-question supporting-ID discovery is in [productivity.json](productivity.json).", "", "## Limits", ""]
    lines += [f"- {note}" for note in summary["limitations"]]
    lines += ["", "## Figures", ""] + [f"![{Path(path).stem}]({path})" for path in summary["figures"]]
    return "\n".join(lines) + "\n"


def build(matrix_path, output, root=REPOSITORY, no_plots=False):
    matrix_path, output, root = Path(matrix_path).resolve(), Path(output).resolve(), Path(root).resolve()
    matrix = read_matrix(matrix_path, root)
    inputs = [safe_source(root, matrix[key]) for key in ("dataset", "corpus", "data_manifest")]
    run_dirs = [safe_source(root, job["run_dir"]) for job in matrix["jobs"]]
    validate_report_output(output, matrix_path, *inputs, *run_dirs)
    output.mkdir(parents=True, exist_ok=True)
    inspections, productivity, issues = inspect_matrix(matrix, root)
    receipts = [archive(output, Path("matrix") / (matrix_path.name + ".gz"), matrix_path.read_bytes(), matrix_path, compress=True)]
    for job in matrix["jobs"]:
        inspection = inspections.get(job["id"])
        if not inspection:
            continue
        run_dir = safe_source(root, job["run_dir"])
        for name in ("results.jsonl", "manifest.json", "summary.json", "checkpoint.json", "resources.jsonl", "guard-status.json", "process.log"):
            path = run_dir / name
            if path.exists():
                raw = inspection["raw_results"] if name == "results.jsonl" else path.read_bytes()
                receipts.append(archive(output, Path("runs") / job["id"] / (name + ".gz"), raw, path, compress=True))
        for name, source in inspection["verified_inputs"].items():
            receipts.append(archive(output, Path("inputs") / job["id"] / (name + ".jsonl.gz"), source["raw"], source["path"], compress=True,
                                    verified_against_generation_manifest=True, original_basename=source["path"].name))
        for source in inspection["verified_sources"]:
            receipts.append(archive(output, Path("source") / job["id"] / (source["relative"] + ".gz"), source["raw"], source["path"], compress=True,
                                    verified_against_generation_manifest=True))
    # Hash-check every data-selection output, including excluded/development
    # questions when the preparer lists them, before labeling the export complete.
    provenance_path = inputs[-1]
    try:
        provenance = json.loads(provenance_path.read_text())
        receipts.append(archive(output, Path("provenance") / (provenance_path.name + ".gz"), provenance_path.read_bytes(), provenance_path, compress=True))
        outputs = provenance.get("outputs", [])
        pinned_paths = set()
        # The new data package is relocatable and records paths relative to its
        # own manifest. Historical adaptive manifests used repository paths.
        provenance_base = provenance_path.parent if provenance.get("protocol") == "controller-scaling-data-v1" else root
        for entry in outputs:
            source = safe_source(root, provenance_base / entry["path"])
            raw = source.read_bytes()
            if sha(raw) != entry["sha256"]:
                raise ValueError(f"Data-selection output changed: {entry['path']}")
            pinned_paths.add(source)
            relative = source.relative_to(root)
            receipts.append(archive(output, Path("provenance/outputs") / (str(relative) + ".gz"), raw, source, compress=True, verified_against_data_manifest=True))
        if not set(inputs[:2]) <= pinned_paths:
            raise ValueError("Data-selection provenance must pin the evaluated questions and corpus")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append({"kind": "data_selection_provenance_unverified", "error": str(exc)})
    for source in (Path(__file__).resolve(), Path(__file__).with_name("build_adaptive_report.py"), Path(__file__).with_name("report_paths.py")):
        receipts.append(archive(output, Path("report-source") / (source.name + ".gz"), source.read_bytes(), source, compress=True))
    for name in matrix.get("provenance_files", []):
        source = safe_source(root, name)
        receipts.append(archive(output, Path("extra-provenance") / (str(source.relative_to(root)) + ".gz"), source.read_bytes(), source, compress=True))
    complete = len(inspections) == len(matrix["jobs"]) and not issues
    metrics, trajectories, details, pairs = compute_results(matrix, inspections, productivity) if complete else ([], [], [], [])
    setup = [{"job_id": job["id"], "setup_timings_ms": inspections[job["id"]]["manifest"].get("setup_timings_ms", {}),
              "offline_corpus": inspections[job["id"]]["manifest"].get("corpus", {}),
              "backend": inspections[job["id"]]["manifest"].get("identity", {}).get("backend", {}),
              "runner_elapsed_seconds": inspections[job["id"]]["runner_summary"].get("elapsed_seconds")}
             for job in matrix["jobs"] if job["id"] in inspections]
    completion = {job: {key: entry[key] for key in ("status", "planned_unique_runs", "successful_unique_runs", "missing_unique_runs", "unresolved_errors", "historical_error_records", "duplicate_record_ids")}
                  for job, entry in inspections.items()}
    summary = {"report_version": VERSION, "report_status": "complete" if complete else "INCOMPLETE_OR_UNVERIFIED",
               "cohort": matrix["cohort"], "matrix_sha256": sha(matrix_path.read_bytes()),
               "questions_sha256": matrix["questions_sha256"], "corpus_sha256": matrix["corpus_sha256"],
               "completion": completion, "verification_issues": issues, "metrics": metrics,
               "paired_comparisons": pairs, "trajectories": trajectories,
               "separate_setup_and_model_metadata": setup, "limitations": LIMITATIONS,
               "successful_unique_conditions": sum(entry["successful_unique_runs"] for entry in inspections.values())}
    summary["figures"] = plots(metrics, trajectories, output, matrix["cohort"]) if complete and not no_plots else []
    csv_rows(output / "metrics.csv", metrics)
    csv_rows(output / "pairs.csv", pairs)
    csv_rows(output / "trajectories.csv", trajectories)
    (output / "productivity.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "tables.md").write_text(tables(summary) if complete else "# Controller scaling — incomplete or unverified\n\nHeadline metrics and figures are withheld. See `summary.json` verification issues and archived source ledgers.\n")
    (output / "DATA_LICENSE.md").write_text("# MuSiQue attribution\n\nTrivedi et al. (2022), [MuSiQue](https://github.com/StonyBrookNLP/musique), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).\n\nModified by deduplicating development-source paragraphs into a shared global corpus and selecting local development/frozen test questions. Selection manifests, hashes and raw normalized inputs are archived. This is not an official full-Wikipedia evaluation. Model weights are not redistributed.\n")
    (output / "artifact-manifest.json").write_text(json.dumps({"report_version": VERSION, "artifacts": receipts}, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = build(args.matrix, args.output_dir, no_plots=args.no_plots)
    print(json.dumps({"status": summary["report_status"], "conditions": summary["successful_unique_conditions"], "issues": summary["verification_issues"]}))
    return 2 if args.require_complete and summary["report_status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
