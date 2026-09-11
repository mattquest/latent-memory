#!/usr/bin/env python3
"""Extract frozen controller workloads without inference or assumed provider speeds.

The output preserves observed answers/quality. Optional explicit provider profiles
produce conditional latency scenarios, not measured cloud performance or quality.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.real_experiments import answer_scores

VERSION = "controller-provider-workloads-v1"
NATIVE = "native-short-controller-screen-v1"
LEGACY = "adaptive-global-bm25-controller-v2"
MODEL_FAMILIES = {
    "native/14b": ("Qwen/Qwen3-14B", "Qwen3-14B"),
    "native/122b": ("Qwen/Qwen3.5-122B-A10B", "Qwen3.5-122B-A10B-5bit"),
    "prior/8b_short": ("Qwen/Qwen3-8B", "Qwen3-8B"),
    "prior/8b_thinking": ("Qwen/Qwen3-8B", "Qwen3-8B"),
    "prior/14b_short": ("Qwen/Qwen3-14B", "Qwen3-14B"),
    "prior/14b_thinking": ("Qwen/Qwen3-14B", "Qwen3-14B"),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative token/count value: {label}")
    return value


def milliseconds(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid finite nonnegative timing: {label}")
    return value


def tokens(value, label):
    if not isinstance(value, list):
        raise ValueError(f"Missing saved token IDs: {label}")
    for token in value:
        count(token, label)
    return value


def equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Workload reconciliation failed: {label}: {actual!r} != {expected!r}")


def sampled(output, samples, stop, label):
    output, samples = tokens(output, label), tokens(samples, label)
    if stop not in ("eos", "max_tokens", "max_reasoning_tokens", "end_think", "not_started", "disabled"):
        raise ValueError(f"Unsupported generation stop: {stop}")
    equal(samples[:len(output)], output, label + " sampled prefix")
    equal(len(samples), len(output) + int(stop == "eos"), label + " EOS accounting")
    return len(samples)


def native_calls(row):
    calls = []
    for index, generation in enumerate(row["generations"]):
        phase = generation["phase"]
        if phase not in ("decision", "final"):
            raise ValueError("Unsupported native generation phase")
        stats = generation["stats"]
        prompt = tokens(generation["prompt_token_ids"], "native prompt")
        output = tokens(generation["output_token_ids"], "native output")
        samples = stats["sampled_token_ids"]
        n_sampled = sampled(output, samples, stats["stop_reason"], "native")
        equal(len(prompt), stats["prefill_tokens"], "native prefill")
        equal(len(output), stats["generated_tokens"], "native generated")
        equal(n_sampled, stats["sampled_tokens_including_eos"], "native samples")
        for key in ("prefill_ms", "generation_ms", "model_inference_ms", "elapsed_ms"):
            milliseconds(stats[key], key)
        if not math.isclose(stats["prefill_ms"] + stats["generation_ms"], stats["model_inference_ms"], abs_tol=1e-6):
            raise ValueError("Native model phase times do not reconcile")
        calls.append({"call_index": index, "phase": "controller" if phase == "decision" else "final",
            "round": generation["round"], "initial_prefill_tokens": len(prompt), "forced_control_prefill_tokens": 0,
            "reasoning_generated_tokens": 0, "reasoning_content_tokens": 0, "reasoning_sampled_tokens": 0,
            "action_generated_tokens": len(output) if phase == "decision" else 0,
            "final_generated_tokens": len(output) if phase == "final" else 0,
            "action_sampled_tokens": n_sampled if phase == "decision" else 0,
            "final_sampled_tokens": n_sampled if phase == "final" else 0,
            "sampled_tokens_including_eos": n_sampled, "sample_count_basis": "saved token IDs",
            "prompt_token_ids_sha256": digest(canonical(prompt)), "output_token_ids_sha256": digest(canonical(output)),
            "sampled_token_ids_sha256": digest(canonical(samples)), "input_record_sha256": digest(canonical(generation["messages"])),
            "stop_reason": stats["stop_reason"], "max_output_tokens": stats["max_output_tokens"],
            "sampling": stats["sampling"], "local_phase_ms": {"initial_prefill": stats["prefill_ms"],
                "generation": stats["generation_ms"]}, "local_call_inner_elapsed_ms": stats["elapsed_ms"],
            "source_pointer": f"generations[{index}]", "mapping_limit": "No measured provider execution; checkpoint, template and sampler equivalence require separate verification."})
    equal([c["phase"] for c in calls].count("final"), 1, "one final call")
    equal(calls[-1]["phase"], "final", "final call ordering")
    equal(row["output_token_ids"], row["generations"][-1]["output_token_ids"], "final output IDs")
    equal(row["raw_output"], row["generations"][-1]["raw_output"], "final raw answer")
    return calls


def legacy_calls(row):
    if row["case"]["arm"] != "iterative_text":
        raise ValueError("KV relay/incremental/reranker workloads are unsupported by this text-API extractor")
    calls = []
    for index, event in enumerate(row["trace"]):
        if event["event"] != "decision":
            continue
        stats = event["controller_decode"]
        equal(stats["prefix_tokens"], 0, "legacy text-only prefix")
        reasoning = tokens(stats["reasoning_output_token_ids"], "reasoning")
        content = tokens(stats["reasoning_token_ids"], "reasoning content")
        action = tokens(stats["action_token_ids"], "action")
        forced = tokens(stats["forced_control_token_ids"], "forced controls")
        nr = sampled(reasoning, stats["reasoning_sampled_token_ids"], stats["reasoning_stop_reason"], "reasoning")
        na = sampled(action, stats["action_sampled_token_ids"], stats["action_stop_reason"], "action")
        equal(reasoning[:len(content)], content, "reasoning content prefix")
        for actual, key in [(len(reasoning), "reasoning_generated_tokens"), (len(action), "action_generated_tokens"),
                            (len(forced), "forced_control_tokens"), (len(reasoning) + len(action), "generated_tokens"),
                            (nr + na, "sampled_tokens")]:
            equal(actual, stats[key], key)
        equal(event["action_token_ids"], action, "trace action")
        phase_ms = stats.get("phase_ms")
        if phase_ms is not None:
            for key, value in phase_ms.items():
                milliseconds(value, key)
        calls.append({"call_index": len(calls), "phase": "controller", "round": event["round"],
            "initial_prefill_tokens": count(stats["prefill_tokens"], "controller prefill"),
            "forced_control_prefill_tokens": len(forced), "reasoning_generated_tokens": len(reasoning),
            "reasoning_content_tokens": len(content), "reasoning_sampled_tokens": nr,
            "action_generated_tokens": len(action), "final_generated_tokens": 0,
            "action_sampled_tokens": na, "final_sampled_tokens": 0, "sampled_tokens_including_eos": nr + na,
            "sample_count_basis": "saved reasoning/action token IDs; generated end-think is included in reasoning",
            "prompt_token_ids_sha256": None, "input_record_sha256": digest(canonical(event)),
            "reasoning_token_ids_sha256": digest(canonical(reasoning)), "action_token_ids_sha256": digest(canonical(action)),
            "forced_control_token_ids_sha256": digest(canonical(forced)),
            "stop_reason": stats["stop_reason"], "reasoning_stop_reason": stats["reasoning_stop_reason"],
            "max_reasoning_tokens": stats["max_reasoning_tokens"], "max_action_tokens": stats["max_action_tokens"],
            "sampling": stats["sampling"], "local_phase_ms": phase_ms, "local_call_inner_elapsed_ms": None,
            "source_pointer": f"trace[{index}].controller_decode",
            "mapping_limit": "One logical controller call; forced closure is an input segment on the same transient state, not another network request. Ordinary provider APIs may not support this exact phase control."})
    output = tokens(row["output_token_ids"], "legacy final")
    stop = row["generation_stop"]
    if stop not in ("eos", "max_tokens"):
        raise ValueError("Cannot infer final sampling count from an unknown stop reason")
    final_sampled = len(output) + int(stop == "eos")
    calls.append({"call_index": len(calls), "phase": "final", "round": row["executed_retrieval_rounds"],
        "initial_prefill_tokens": count(row["counts"]["final_prefill_tokens"], "final prefill"),
        "forced_control_prefill_tokens": 0, "reasoning_generated_tokens": 0, "reasoning_content_tokens": 0,
        "reasoning_sampled_tokens": 0, "action_generated_tokens": 0, "final_generated_tokens": len(output),
        "action_sampled_tokens": 0, "final_sampled_tokens": final_sampled, "sampled_tokens_including_eos": final_sampled,
        "sample_count_basis": "derived: saved final output length plus one iff generation_stop=eos, as implemented by pinned MLXBackend.decode; EOS identity was not saved",
        "prompt_token_ids_sha256": None, "output_token_ids_sha256": digest(canonical(output)),
        "sampled_token_ids_sha256": None, "input_record_sha256": digest(canonical(row)),
        "stop_reason": stop, "max_output_tokens": row["controller_config"]["max_answer_tokens"],
        "sampling": {"temperature": 0.0}, "local_phase_ms": None, "local_call_inner_elapsed_ms": None,
        "source_pointer": "counts.final_prefill_tokens + output_token_ids + generation_stop",
        "mapping_limit": "Final prompt count is saved; exact final prompt/EOS token arrays were not persisted."})
    return calls


def extract_condition(row, source_kind):
    if row.get("status") != "ok":
        raise ValueError("Only completed successful records can become projected workloads")
    if source_kind == "native":
        equal(row["protocol"], NATIVE, "native protocol")
        if row["arm"] not in ("basic_rag", "iterative_text"):
            raise ValueError("Unsupported native arm")
        calls, arm = native_calls(row), row["arm"]
        model_keys = ("controller_ms", "final_ms")
    elif source_kind == "legacy":
        equal(row["protocol_version"], LEGACY, "legacy protocol")
        calls, arm = legacy_calls(row), row["case"]["arm"]
        model_keys = ("controller_ms", "final_generation_ms")
        for key in ("cache_construction_ms", "reranking_ms"):
            equal(row["component_ms"][key], 0, "unsupported auxiliary work")
        for key in ("cache_prefill_tokens", "bridge_recomputed_tokens", "reranker_input_tokens"):
            equal(row["counts"][key], 0, "unsupported auxiliary token work")
    else:
        raise ValueError("Unknown workload source schema")
    totals = {key: sum(c[key] for c in calls) for key in (
        "initial_prefill_tokens", "forced_control_prefill_tokens", "reasoning_generated_tokens", "reasoning_content_tokens",
        "action_generated_tokens", "final_generated_tokens", "reasoning_sampled_tokens", "action_sampled_tokens",
        "final_sampled_tokens", "sampled_tokens_including_eos")}
    totals.update(logical_model_calls=len(calls), controller_calls=sum(c["phase"] == "controller" for c in calls),
                  final_calls=1, generated_tokens=sum(totals[k] for k in ("reasoning_generated_tokens", "action_generated_tokens", "final_generated_tokens")))
    expected = row["counts"]
    equal(totals["controller_calls"], expected["controller_calls"], "controller calls")
    equal(sum(c["initial_prefill_tokens"] for c in calls if c["phase"] == "controller"), expected["controller_prefill_tokens"], "controller prefill total")
    equal(totals["initial_prefill_tokens"] + totals["forced_control_prefill_tokens"], expected["total_prefill_tokens"], "all prefill including forced controls")
    if source_kind == "native":
        checks = {"controller_generated_tokens": totals["action_generated_tokens"], "final_generated_tokens": totals["final_generated_tokens"],
                  "controller_sampled_tokens_including_eos": totals["action_sampled_tokens"],
                  "final_sampled_tokens_including_eos": totals["final_sampled_tokens"], "total_generated_tokens": totals["generated_tokens"]}
    else:
        checks = {"controller_reasoning_tokens": totals["reasoning_generated_tokens"], "controller_action_tokens": totals["action_generated_tokens"],
                  "controller_forced_tokens": totals["forced_control_prefill_tokens"], "host_output_tokens": totals["final_generated_tokens"],
                  "internal_decoded_tokens": totals["reasoning_generated_tokens"] + totals["action_generated_tokens"],
                  "controller_sampled_tokens_including_eos": totals["reasoning_sampled_tokens"] + totals["action_sampled_tokens"]}
    for key, value in checks.items():
        equal(value, expected[key], key)
    component = row["component_ms"]
    for key, value in component.items():
        milliseconds(value, key)
    elapsed = milliseconds(row["cold_end_to_end_ms"], "cold total")
    if not math.isclose(sum(component.values()), elapsed, abs_tol=1e-5):
        raise ValueError("Local cold timing components do not reconcile")
    wrapper_ms = sum(component[k] for k in model_keys)
    within_call_ms = None
    if source_kind == "native":
        within_call_ms = wrapper_ms - sum(sum(c["local_phase_ms"].values()) for c in calls)
        if within_call_ms < -1e-5:
            raise ValueError("Native model phases exceed the measured call windows")
        within_call_ms = max(0.0, within_call_ms)
    return {"case_id": row["id"], "example_id": row["example_id"], "arm": arm, "category": row["category"],
        "question": row["question"], "answers": row["answers"], "observed_prediction": row["prediction"],
        "observed_scores_unchanged": row["scores"], "source_row_sha256": digest(canonical(row)),
        "search_queries": row["search_queries"], "evidence_ids": row["evidence_ids"], "stop_reason": row["stop_reason"],
        "calls": calls, "totals": totals, "local_observed_ms": {"cold_end_to_end": elapsed,
            "model_call_windows": wrapper_ms, "outer_host_work": sum(v for k, v in component.items() if k not in model_keys),
            "native_within_call_wrapper_work": within_call_ms, "saved_components": component},
        "timing_limit": "Outer host work is measured outside model-call windows. Older calls do not separate all template/sampling/wrapper work inside those windows; native generation phases also include sampling runtime, not GPU-only timing."}


def verify_sources(run_dir, manifest, repository):
    receipts = []
    for name, expected in manifest["identity"]["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Unsafe or malformed source receipt")
        candidates = [(run_dir / "source_snapshot" / relative, "run source snapshot"), (repository / relative, "matching current bytes")]
        found = next(((p.read_bytes(), origin) for p, origin in candidates if p.is_file() and digest(p.read_bytes()) == expected), None)
        if found is None:
            ref = manifest.get("machine", {}).get("git_head") or manifest.get("git_commit")
            if ref and re.fullmatch(r"[0-9a-f]{40}", ref):
                result = subprocess.run(["git", "show", f"{ref}:{name}"], cwd=repository, capture_output=True, check=False)
                if result.returncode == 0 and digest(result.stdout) == expected:
                    found = (result.stdout, "recorded git commit " + ref)
        if found is None:
            raise ValueError("Exact executed source unavailable: " + name)
        receipts.append({"path": name, "sha256": expected, "verified_from": found[1]})
    return receipts


def extract_run(run_dir, label, source_kind, questions, input_hashes, repository=ROOT):
    run_dir = Path(run_dir)
    raw = {name: (run_dir / name).read_bytes() for name in ("results.jsonl", "manifest.json", "summary.json")}
    manifest, summary = json.loads(raw["manifest.json"]), json.loads(raw["summary.json"])
    identity = manifest["identity"]
    equal(identity["questions_sha256"], input_hashes["questions"], "questions hash")
    equal(identity["corpus_sha256"], input_hashes["corpus"], "corpus hash")
    sources = verify_sources(run_dir, manifest, Path(repository))
    rows = [json.loads(line) for line in raw["results.jsonl"].splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate result IDs are not additional workload samples")
    if source_kind == "native":
        equal(summary["status"], "complete", "native completion")
        equal(summary["successful_conditions"], len(rows), "native completion count")
        equal(summary["planned_conditions"], len(rows), "native planned count")
        arms = identity["arms"]
    else:
        equal(summary["stop_reason"], "complete", "legacy completion")
        equal(summary["successful_unique_runs"], len(rows), "legacy completion count")
        equal(summary["planned_runs"], len(rows), "legacy planned count")
        arms = identity["config"]["arms"]
    expected = {(q, arm) for q in questions for arm in arms}
    workloads = []
    for line_number, row in enumerate(rows, 1):
        q = questions[row["example_id"]]
        equal(row["question"], q["question"], "question text")
        equal(row["answers"], q["answers"], "answers")
        equal(row["scores"], answer_scores(row["prediction"], q["answers"]), "unchanged scores")
        workload = extract_condition(row, source_kind)
        workload.update(run_label=label, source_kind=source_kind, source_line=line_number,
                        model_revision=identity["backend"]["model_revision"], model_path=identity["backend"]["model_path"])
        workloads.append(workload)
    equal({(w["example_id"], w["arm"]) for w in workloads}, expected, "complete declared question/arm pairs")
    equal(len(workloads), len(expected), "unique question/arm pairs")
    return {"label": label, "source_kind": source_kind, "directory": str(run_dir), "source_sha256": sources,
        "input_receipts": [{"path": str(run_dir / name), "sha256": digest(value), "bytes": len(value)} for name, value in raw.items()],
        "backend_identity": identity["backend"], "config": identity["config"], "workloads": workloads}


def validate_profile(profile):
    for key in ("id", "model_id", "provider", "date", "source_url", "snapshot_sha256", "metric_semantics"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError("Provider profile requires nonempty " + key)
    date.fromisoformat(profile["date"])
    url = urlparse(profile["source_url"])
    if url.scheme not in ("http", "https") or not url.netloc:
        raise ValueError("Provider profile needs a source HTTP(S) URL")
    if not re.fullmatch(r"[0-9a-f]{64}", profile["snapshot_sha256"]):
        raise ValueError("Provider snapshot SHA256 must be explicit")
    for key in ("ttft_s", "reported_tps"):
        if milliseconds(profile.get(key), key) <= 0:
            raise ValueError("Provider TTFT and throughput must be positive finite values")
    if profile["metric_semantics"] not in ("decode_tps_from_tpot", "ambiguous_reported_throughput"):
        raise ValueError("Unsupported provider metric semantics")
    targets = profile.get("applies_to")
    if (not isinstance(targets, list) or not targets or
            any(not isinstance(target, str) or not target for target in targets) or len(set(targets)) != len(targets)):
        raise ValueError("Provider profile requires unique explicit applies_to run labels")
    for target in targets:
        if target not in MODEL_FAMILIES or profile["model_id"] != MODEL_FAMILIES[target][0]:
            raise ValueError("Provider model_id does not match the declared source run: " + target)
    return profile


def load_profiles(path):
    """Verify optional internal snapshots without exporting their contents."""
    path = Path(path)
    raw = path.read_bytes()
    profiles = json.loads(raw)["profiles"]
    root = path.parent.resolve()
    for profile in profiles:
        validate_profile(profile)
        relative = profile.get("snapshot_path")
        profile["snapshot_verified"] = False
        if relative is None:
            continue
        if not isinstance(relative, str) or not relative or ".." in Path(relative).parts:
            raise ValueError("Unsafe provider snapshot path")
        source = Path(relative)
        if not source.is_absolute():
            # Accept either profile-directory-relative or repository-relative
            # receipts; both must identify a file confined to this profile directory.
            repository_source = ROOT / source
            source = repository_source if repository_source.is_relative_to(root) else root / source
        if root not in source.resolve().parents:
            raise ValueError("Provider snapshot must stay inside the profile directory")
        if source.is_symlink() or any(p.is_symlink() for p in source.parents if p != root and root in p.parents):
            raise ValueError("Provider snapshot symlinks are not accepted")
        equal(digest(source.read_bytes()), profile["snapshot_sha256"], "provider source snapshot hash")
        profile["snapshot_verified"] = True
    return profiles, raw


def project_condition(workload, profile):
    """Two explicit what-if formulas; neither is a bound or confidence interval."""
    validate_profile(profile)
    label = workload["run_label"]
    if label not in profile["applies_to"]:
        raise ValueError("Provider profile does not apply to the workload source run")
    expected_model, expected_checkpoint = MODEL_FAMILIES[label]
    if (profile["model_id"] != expected_model or
            Path(workload["model_path"]).name != expected_checkpoint):
        raise ValueError("Provider model family and recorded local checkpoint do not match")
    host = milliseconds(workload["local_observed_ms"]["outer_host_work"], "outer host") / 1000
    calls = []
    for call in workload["calls"]:
        generated = sum(count(call[k], k) for k in ("reasoning_generated_tokens", "action_generated_tokens", "final_generated_tokens"))
        a = profile["ttft_s"] + max(generated - 1, 0) / profile["reported_tps"]
        b = (max(profile["ttft_s"], generated / profile["reported_tps"])
             if profile["metric_semantics"] == "ambiguous_reported_throughput" else None)
        calls.append({"call_index": call["call_index"], "phase": call["phase"], "generated_tokens_used": generated,
                      "decode_rate_assumption_s": a, "inclusive_rate_proxy_s": b})
    if not calls:
        raise ValueError("A projected condition must contain recorded model calls")
    return {"case_id": workload["case_id"], "example_id": workload["example_id"], "arm": workload["arm"],
        "run_label": workload["run_label"], "profile_id": profile["id"],
        "observed_scores_unchanged": workload["observed_scores_unchanged"], "observed_prediction": workload["observed_prediction"],
        "outer_host_s": host, "calls": calls,
        "decode_rate_assumption_s": host + sum(call["decode_rate_assumption_s"] for call in calls),
        "inclusive_rate_proxy_s": (host + sum(call["inclusive_rate_proxy_s"] for call in calls)
                                    if profile["metric_semantics"] == "ambiguous_reported_throughput" else None)}


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    left = math.floor(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)


def project_workloads(workloads, profiles):
    equal(len({profile["id"] for profile in profiles}), len(profiles), "unique profile IDs")
    runs = {run["label"]: run for run in workloads["runs"]}
    projected = []
    for profile in profiles:
        validate_profile(profile)
        if set(profile["applies_to"]) - set(runs):
            raise ValueError("Profile references an unknown source run")
        for label in profile["applies_to"]:
            for row in runs[label]["workloads"]:
                equal(row["run_label"], label, "workload source label")
            projected.extend(project_condition(row, profile) for row in runs[label]["workloads"])
    grouped = defaultdict(list)
    for row in projected:
        grouped[(row["profile_id"], row["run_label"], row["arm"])].append(row)
    summaries = []
    for (profile_id, label, arm), rows in sorted(grouped.items()):
        summary = {"profile_id": profile_id, "run_label": label, "arm": arm, "conditions": len(rows),
                   "observed_correct_unchanged": sum(row["observed_scores_unchanged"]["exact_match"] for row in rows)}
        for scenario in ("decode_rate_assumption_s", "inclusive_rate_proxy_s"):
            values = [row[scenario] for row in rows if row[scenario] is not None]
            summary[scenario] = ({"n": len(values), "mean": sum(values) / len(values), "p50": quantile(values, .5), "p95": quantile(values, .95)} if values else None)
        summaries.append(summary)
    pairs = []
    contrasts = [("native/" + model, "basic_rag", "native/" + model, "iterative_text") for model in ("14b", "122b")]
    contrasts += [("prior/" + model + "_short", "iterative_text", "prior/" + model + "_thinking", "iterative_text") for model in ("8b", "14b")]
    for profile in profiles:
        for left_label, left_arm, right_label, right_arm in contrasts:
            left = {r["example_id"]: r for r in grouped.get((profile["id"], left_label, left_arm), [])}
            right = {r["example_id"]: r for r in grouped.get((profile["id"], right_label, right_arm), [])}
            if not left or not right:
                continue
            equal(set(left), set(right), "paired projection question IDs")
            pair = {"profile_id": profile["id"], "baseline": left_label + "/" + left_arm,
                    "comparison": right_label + "/" + right_arm, "n_paired": len(left), "example_ids": sorted(left)}
            for scenario in ("decode_rate_assumption_s", "inclusive_rate_proxy_s"):
                if left[next(iter(left))][scenario] is None:
                    pair[scenario] = None
                    continue
                differences = [right[key][scenario] - left[key][scenario] for key in left]
                ratios = [right[key][scenario] / left[key][scenario] for key in left]
                pair[scenario] = {"mean_paired_difference_s": sum(differences) / len(differences),
                                  "median_paired_comparison_over_baseline": quantile(ratios, .5)}
            pairs.append(pair)
    return {"schema_version": "controller-provider-projection-v1", "profiles": profiles,
        "conditions": projected, "summaries": summaries, "paired_summaries": pairs,
        "unprofiled_runs": sorted(set(runs) - {label for p in profiles for label in p["applies_to"]}),
        "method": {"decode_rate_assumption_s": "outer_host_s + sum(TTFT_s + max(generated_tokens - 1, 0) / reported_tps)",
                   "inclusive_rate_proxy_s": "outer_host_s + sum(max(TTFT_s, generated_tokens / reported_tps)); only for ambiguous reported throughput",
                   "generated_tokens": "Actual generated reasoning (including generated end-think), actions and final output; excludes sampled EOS and forced input tokens."},
        "limitations": ["These are distinct what-if scenarios, not confidence intervals, hard bounds or measured provider latency.",
            "TTFT includes unmatched prefill/network/queue work. No extra prefill-token/rate charge is added and no prefill scaling slope is inferred.",
            "EOS termination, forced-closure prefill latency and within-call host work are not separately modeled. The forced phase protocol assumes custom server support.",
            "Measured outer host overhead stays local. Provider model/checkpoint/precision/template/sampling changes can alter quality and trajectories; observed local accuracy is held fixed, not transferred.",
            "Profile routing checks the explicit model family and recorded checkpoint basename. The native 122B checkpoint is a local 5-bit conversion; family matching does not establish provider precision or execution equivalence.",
            "Native and prior screens use different prompts/controllers and are summarized separately. Original KV-relay experiments and posthoc final ablations are unsupported/excluded."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=ROOT / "runs/large-controller/pilot-v1")
    parser.add_argument("--controller-root", type=Path, default=ROOT / "runs/controller-scaling/pilot-v1")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/controller-scaling/dev.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/controller-scaling/corpus.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/large-controller/provider-latency")
    parser.add_argument("--profiles", type=Path, help="Optional explicit sourced provider profiles; no default rates")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protected = [args.native_root.resolve(), args.controller_root.resolve(), *[ROOT / x for x in ("data", "models", "engine", "eval", "scripts", "tests")]]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("Output overlaps protected input/code directories")
    raw_questions, raw_corpus = args.dataset.read_bytes(), args.corpus.read_bytes()
    questions_list = [json.loads(line) for line in raw_questions.splitlines() if line.strip()]
    questions = {q["id"]: q for q in questions_list}
    equal(len(questions_list), 12, "declared twelve-question screen")
    equal(len(questions), 12, "unique questions")
    input_hashes = {"questions": digest(raw_questions), "corpus": digest(raw_corpus)}
    runs = [extract_run(args.native_root / label, "native/" + label, "native", questions, input_hashes) for label in ("14b", "122b")]
    runs += [extract_run(args.controller_root / label, "prior/" + label, "legacy", questions, input_hashes)
             for label in ("8b_short", "8b_thinking", "14b_short", "14b_thinking")]
    equal(sum(len(run["workloads"]) for run in runs), 96, "both complete screens")
    groups = defaultdict(list)
    for run in runs:
        for w in run["workloads"]:
            groups[(run["label"], w["arm"])].append(w)
    groups = [{"run_label": label, "arm": arm, "conditions": len(rows),
               "observed_correct_unchanged": sum(row["observed_scores_unchanged"]["exact_match"] for row in rows),
               "totals": {key: sum(row["totals"][key] for row in rows) for key in rows[0]["totals"]}}
              for (label, arm), rows in sorted(groups.items())]
    result = {"schema_version": VERSION, "projection_status": "Frozen workload ledger; any conditional provider scenarios are stored separately in projections.json",
        "extractor_sha256": digest(Path(__file__).read_bytes()), "conditions": 96, "distinct_questions": 12,
        "dataset": {"path": str(args.dataset), "sha256": input_hashes["questions"]},
        "corpus": {"path": str(args.corpus), "sha256": input_hashes["corpus"]}, "groups": groups, "runs": runs,
        "unsupported": ["Original KV-relay/incremental-cache experiments cannot be mapped to an ordinary text-chat API without implementing their cache operations.",
                        "Posthoc final-answer ablations are intentionally excluded.",
                        "Provider accuracy, trajectory, sampling, context processing, and model/checkpoint equivalence are not measured or assumed.",
                        "Reasoning and action are one logical controller call; forced closure is separate input work and requires provider-specific execution support."]}
    output.mkdir(parents=True, exist_ok=True)
    target = output / "workloads.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(target)
    if args.profiles:
        profiles, profiles_raw = load_profiles(args.profiles)
        projection = project_workloads(result, profiles)
        projection.update(workloads_sha256=digest(target.read_bytes()), profiles_sha256=digest(profiles_raw),
                          extractor_sha256=digest(Path(__file__).read_bytes()))
        projection_target = output / "projections.json"
        projection_temporary = projection_target.with_suffix(".json.tmp")
        projection_temporary.write_text(json.dumps(projection, indent=2, sort_keys=True, allow_nan=False) + "\n")
        projection_temporary.replace(projection_target)
    print(json.dumps({"output": str(target), "conditions": 96, "logical_model_calls": sum(g["totals"]["logical_model_calls"] for g in groups),
                      "projection_status": result["projection_status"]}))


if __name__ == "__main__":
    main()
