#!/usr/bin/env python3
"""Strict CPU-only export of the native, nonthinking local controller screen.

Replay retrieval, tokenization and accounting; never load model weights. A
partial or unverifiable matrix retains its raw artifacts but has no headlines.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from eval.adaptive_experiments import Corpus, canonical_query, parse_action
from eval.controller_screen import PROTOCOL, ScreenConfig, bounded_fragment, messages_for
from eval.real_experiments import answer_scores, clean_prediction
from scripts.build_adaptive_report import archive, csv_rows, safe_source, snapshot
from scripts.report_paths import validate_report_output

VERSION = "large-controller-report-v1"
ARMS = ["basic_rag", "iterative_text"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def expected_id(config, example_id, arm, revision):
    return sha(json.dumps([PROTOCOL, config, example_id, arm, revision], sort_keys=True).encode())[:24]


def expected_seed(config, example_id, round_number):
    return int(sha(json.dumps([config["seed"], example_id, "decision", round_number], sort_keys=True).encode())[:8], 16)


def read_json(path):
    return json.loads(path.read_text())


def tokenizer_for(model_path):
    """Load local tokenizer metadata only, without importing MLX or model code."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    generation = read_json(model_path / "generation_config.json")
    eos = generation.get("eos_token_id", tokenizer.eos_token_id)
    return LocalTokenizer(tokenizer, set(eos if isinstance(eos, list) else [eos]))


def expected_loaded_config(model_config, generation_config, versions):
    """Replay only known metadata mutations from the recorded MLX-LM loader.

    MLX-LM 0.31.3 utils.load_config overlays generation-config EOS. Its
    qwen3_5.TextModelArgs.__post_init__ renames the shared rope_parameters key
    from rope_type to type only when type is absent. No field is discarded
    from comparison, and other architectures/versions get no RoPE rewrite.
    """
    expected, normalizations = copy.deepcopy(model_config), []
    if generation_config.get("eos_token_id"):
        expected["eos_token_id"] = copy.deepcopy(generation_config["eos_token_id"])
    if expected.get("model_type") == "qwen3_5_moe" and versions.get("mlx-lm") == "0.31.3":
        rope = expected.get("text_config", {}).get("rope_parameters")
        if isinstance(rope, dict) and "type" not in rope and "rope_type" in rope:
            rope["type"] = rope.pop("rope_type")
            normalizations.append({"path": "text_config.rope_parameters", "old_key": "rope_type",
                "new_key": "type", "value": copy.deepcopy(rope["type"]), "library": "mlx-lm", "version": "0.31.3",
                "source": "mlx_lm/models/qwen3_5.py:TextModelArgs.__post_init__"})
    return expected, normalizations


class LocalTokenizer:
    """Keep the model's EOS set outside Transformers' special-token setters."""
    def __init__(self, tokenizer, eos_token_ids):
        self.tokenizer, self.eos_token_ids = tokenizer, eos_token_ids

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        # MLX-LM's wrapper requests the legacy token-list return shape even
        # when installed Transformers defaults to a BatchEncoding mapping.
        kwargs["return_dict"] = False
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def verify_generation(generation, phase, round_number, messages, config, question, tokenizer, native_cache_types):
    require(type(generation["round"]) is int, "Generation round must be an integer")
    require(generation["phase"] == phase and generation["round"] == round_number, "Generation phase/round mismatch")
    require(generation["messages"] == messages, "Generation messages differ from the source-derived evidence prompt")
    prompt = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    require(generation["prompt_token_ids"] == prompt and bool(prompt), "Native prompt token IDs differ")
    output, stats = generation["output_token_ids"], generation["stats"]
    require(all(type(token) is int and token >= 0 for token in prompt + output), "Invalid token IDs")
    require(generation["raw_output"] == tokenizer.decode(output, skip_special_tokens=True), "Raw output differs from token decoding")
    budget = config["max_action_tokens"] if phase == "decision" else config["max_answer_tokens"]
    require(len(output) <= budget and len(prompt) + budget <= config["max_context_tokens"], "Complete generation budget exceeded")
    expected_sampling = ({"temperature": config["controller_temperature"], "top_p": config["controller_top_p"],
        "top_k": config["controller_top_k"], "presence_penalty": config["controller_presence_penalty"],
        "seed": expected_seed(config, question["id"], round_number)} if phase == "decision" else
        {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "presence_penalty": 0.0, "seed": None})
    for key, value in expected_sampling.items():
        require(stats["sampling"][key] == value, "Generation sampling/seed mismatch: " + key)
    require(stats["sampling"]["presence_scope"] == "unique generated tokens in this call; prompt excluded", "Presence-penalty scope changed")
    require(stats["thinking"] is False, "Thinking must remain disabled")
    require(stats["native_cache_types"] == native_cache_types, "Native cache types changed")
    require(all(type(stats[key]) is int and stats[key] >= 0 for key in
        ("prefill_tokens", "generated_tokens", "sampled_tokens_including_eos", "context_reservation_tokens", "max_output_tokens")), "Generation counters must be integers")
    require(stats["prefill_tokens"] == len(prompt) and stats["generated_tokens"] == len(output), "Generation token counters differ")
    require(stats["context_reservation_tokens"] == len(prompt) + budget and stats["max_output_tokens"] == budget, "Generation reservation differs")
    sampled = stats["sampled_token_ids"]
    require(all(type(token) is int and token >= 0 for token in sampled), "Invalid sampled token IDs")
    require(stats["sampled_tokens_including_eos"] == len(sampled), "Sampled token count differs")
    eos = set(tokenizer.eos_token_ids)
    require(not (set(output) & eos), "Output includes terminal EOS")
    if stats["stop_reason"] == "eos":
        require(len(sampled) == len(output) + 1 and sampled[:-1] == output and sampled[-1] in eos and len(sampled) <= budget, "EOS ledger mismatch")
    else:
        require(stats["stop_reason"] == "max_tokens" and sampled == output and len(output) == budget, "Output-cap ledger mismatch")
    for key in ("template_ms", "prefill_ms", "generation_ms", "model_inference_ms", "elapsed_ms"):
        require(numeric(stats[key]), "Invalid generation timing: " + key)
    require(close(stats["model_inference_ms"], stats["prefill_ms"] + stats["generation_ms"]), "Generation timing sum differs")
    require(stats["elapsed_ms"] + 1e-6 >= stats["model_inference_ms"], "Generation elapsed time omits inference")
    return stats


def inspect_row(row, question, job, config, corpus, tokenizer, backend):
    """Independently replay all retrieval and prompt bookkeeping from raw inputs."""
    arm = row["arm"]
    require(all(type(value) is int and value >= 0 for value in row["counts"].values()), "Condition counters must be nonnegative integers")
    require(type(row["executed_retrieval_rounds"]) is int and type(row["invalid_actions"]) is int, "Round/action counters must be integers")
    require(arm in ARMS and row["status"] == "ok" and row["protocol"] == PROTOCOL, "Unexpected arm/status/protocol")
    require(row["id"] == expected_id(config, question["id"], arm, job["model_revision"]), "Condition ID differs")
    require(row["model_revision"] == job["model_revision"], "Condition model revision differs")
    for key in ("question", "answers", "category", "split"):
        require(row[key] == question[key], "Question/scoring metadata differs: " + key)
    require(all(numeric(value) for value in row["scores"].values()) and numeric(row["support_annotation_coverage"]), "Scores and support coverage must be finite numbers")
    require(row["prediction"] == clean_prediction(row["raw_output"]), "Cleaned prediction differs")
    require(row["scores"] == answer_scores(row["prediction"], question["answers"]), "Recomputed answer scores differ")
    adapter = TokenizerAdapter(tokenizer)
    require(len(adapter.encode(question["question"])) <= config["max_question_tokens"], "Question token budget exceeded")
    fragments, queries, skipped, steps = [], [], [], []
    trace, generations = row["trace"], row["generations"]
    trace_index = generation_index = 0
    query, stop, invalid = question["question"], "round_budget", 0
    rounds = 1 if arm == "basic_rag" else config["max_rounds"]
    document_limit = config["initial_top_k"] if arm == "basic_rag" else config["max_documents"]
    support = set(question["supporting_context_ids"])
    require(bool(support) and support <= corpus.by_id.keys(), "Invalid evaluation support IDs")
    for round_number in range(1, rounds + 1):
        queries.append(query)
        hits = corpus.search(query, min(len(corpus.documents), config["max_documents"] + config["initial_top_k"]))
        seen, new_ids = {fragment["id"] for fragment in fragments}, []
        new_limit = config["initial_top_k"] if round_number == 1 else config["documents_per_round"]
        for document, _score in hits:
            if document["id"] in seen:
                continue
            if len(new_ids) == new_limit or len(fragments) == document_limit:
                break
            fragment = bounded_fragment(adapter, document, config["max_document_tokens"])
            joined = "".join(item["text"] for item in fragments + [fragment])
            if len(adapter.encode(joined)) > config["max_evidence_tokens"]:
                skipped.append(document["id"])
                continue
            fragments.append(fragment)
            new_ids.append(fragment["id"])
            seen.add(fragment["id"])
        evidence_tokens = adapter.encode("".join(fragment["text"] for fragment in fragments))
        event = trace[trace_index]
        expected_event = {"event": "retrieval", "round": round_number, "query": query,
            "hit_ids_and_scores": [[doc["id"], score] for doc, score in hits], "new_document_ids": new_ids,
            "accumulated_document_ids": [fragment["id"] for fragment in fragments], "evidence_tokens": len(evidence_tokens),
            "evidence_token_ids_sha256": sha(json.dumps(evidence_tokens).encode())}
        require(event == expected_event, "Retrieval trace differs from deterministic corpus replay")
        trace_index += 1
        steps.append({"round": round_number, "new_document_ids": new_ids,
            "new_support_ids": sorted(support & set(new_ids)), "cumulative_support_ids": sorted(support & seen),
            "support_coverage": len(support & seen) / len(support)})
        if arm == "basic_rag":
            stop = "single_retrieval_baseline"
            break
        if not new_ids:
            stop = "no_new_evidence"
            break
        if len(fragments) == document_limit:
            stop = "document_budget"
            break
        if round_number == rounds:
            break
        messages = messages_for(question["question"], fragments, "decision", queries, rounds - round_number)
        generation = generations[generation_index]
        verify_generation(generation, "decision", round_number, messages, config, question, tokenizer, backend["native_cache_types"])
        action = parse_action(generation["raw_output"], allow_answer_payload=True)
        require(trace[trace_index] == {"event": "decision", "round": round_number,
            "raw_action": generation["raw_output"], "action": action, "generation_index": generation_index}, "Decision trace differs")
        trace_index += 1
        generation_index += 1
        if not action["valid"]:
            invalid, stop = invalid + 1, "invalid_action_fallback_to_final"
            break
        if action["kind"] == "answer":
            stop = "controller_answer"
            break
        query = action["query"]
        if canonical_query(query) in {canonical_query(value) for value in queries}:
            stop = "repeated_query_fallback_to_final"
            break
    final = generations[generation_index]
    verify_generation(final, "final", round_number, messages_for(question["question"], fragments, "final", queries, rounds - round_number),
                      config, question, tokenizer, backend["native_cache_types"])
    require(generation_index + 1 == len(generations) and trace_index == len(trace), "Unexpected extra generation/trace event")
    require(row["raw_output"] == final["raw_output"] and row["output_token_ids"] == final["output_token_ids"], "Scored output differs from final call")
    require(row["executed_retrieval_rounds"] == round_number and row["stop_reason"] == stop and row["invalid_actions"] == invalid, "Stop/round/invalid-action counters differ")
    require(row["search_queries"] == queries and row["delivered_fragments"] == fragments, "Queries or delivered fragments differ")
    require(row["evidence_ids"] == [fragment["id"] for fragment in fragments] and row["skipped_evidence_budget_ids"] == skipped, "Evidence identity differs")
    require(close(row["support_annotation_coverage"], steps[-1]["support_coverage"]), "Annotated support coverage differs")
    decisions = generations[:-1]
    counts = {"retrieval_calls": round_number, "controller_calls": len(decisions),
        "controller_prefill_tokens": sum(len(g["prompt_token_ids"]) for g in decisions),
        "controller_generated_tokens": sum(len(g["output_token_ids"]) for g in decisions),
        "controller_sampled_tokens_including_eos": sum(g["stats"]["sampled_tokens_including_eos"] for g in decisions),
        "final_prefill_tokens": len(final["prompt_token_ids"]), "final_generated_tokens": len(final["output_token_ids"]),
        "final_sampled_tokens_including_eos": final["stats"]["sampled_tokens_including_eos"],
        "retrieved_documents": len(fragments), "retrieved_evidence_tokens": len(evidence_tokens),
        "max_model_context_tokens": max(g["stats"]["context_reservation_tokens"] for g in generations)}
    counts["total_prefill_tokens"] = counts["controller_prefill_tokens"] + counts["final_prefill_tokens"]
    counts["total_generated_tokens"] = counts["controller_generated_tokens"] + counts["final_generated_tokens"]
    require(row["counts"] == counts, "Condition token/retrieval counters differ")
    require(numeric(row["cold_end_to_end_ms"]) and row["cold_end_to_end_ms"] > 0, "Invalid cold latency")
    require(set(row["component_ms"]) == {"retrieval_ms", "controller_ms", "final_ms", "other_ms"}, "Unexpected timing components")
    require(all(numeric(value) for value in row["component_ms"].values()), "Invalid component timing")
    require(close(sum(row["component_ms"].values()), row["cold_end_to_end_ms"]), "End-to-end timing sum differs")
    for phase, calls in (("controller", decisions), ("final", [final])):
        require(row["component_ms"][phase + "_ms"] + 1e-6 >= sum(g["stats"]["elapsed_ms"] for g in calls), "Phase timing omits generation work")
    for memory in [row["memory"], *(g["stats"]["memory"] for g in generations)]:
        require(all(numeric(value) for value in memory.values()), "Invalid memory measurement")
        require(memory["memory_limit_bytes"] == int(job["memory_limit_gib"] * 1024**3), "Memory ceiling differs")
    return steps


def inspect_checkpoint(manifest, job, root):
    receipt = manifest["checkpoint_receipt"]
    path = safe_source(root, receipt["manifest_path"])
    raw = path.read_bytes()
    require(sha(raw) == receipt["manifest_sha256"], "Checkpoint receipt hash differs")
    checkpoint = json.loads(raw)
    require(checkpoint["revision"] == job["model_revision"] and checkpoint["download_complete"] is True, "Checkpoint receipt incomplete or wrong revision")
    require(receipt["revision"] == checkpoint["revision"] and receipt["repo_id"] == checkpoint["repo_id"], "Executed checkpoint receipt identity differs")
    model_path = safe_source(root, job["model_path"])
    metadata, records = [], []
    for item in checkpoint["files"]:
        name = item.get("path", item.get("name"))
        require(Path(name).name == name, "Unexpected nested checkpoint receipt file")
        source = model_path / name
        require(not source.is_symlink() and source.is_file(), "Missing or redirected checkpoint file")
        require(source.stat().st_size == item.get("size_bytes", item.get("bytes")), "Checkpoint file size differs")
        records.append({"path": name, "bytes": source.stat().st_size, "sha256": item["sha256"]})
        if not name.endswith(".safetensors"):
            content = source.read_bytes()
            require(sha(content) == item["sha256"], "Checkpoint metadata hash differs: " + name)
            metadata.append(source)
    require(metadata, "Checkpoint receipt contains no verified metadata")
    require(receipt["verified_files"] == records and receipt["verified_bytes"] == sum(item["bytes"] for item in records), "Executed checkpoint file ledger differs")
    require({p.name for p in model_path.glob("*.safetensors")} == {row["path"] for row in records if row["path"].endswith(".safetensors")}, "Missing or unrecorded checkpoint weights")
    return path, metadata


def inspect_guard(directory, job, max_seconds):
    guard = read_json(directory / "guard-status.json")
    require(guard["status"] == "complete" and guard["returncode"] == 0 and guard["reason"] is None, "Resource guard did not complete cleanly")
    limits = guard["limits"]
    expected = {"max_seconds": max_seconds, "max_rss_gib": job["memory_limit_gib"], "min_available_gib": 12,
                "min_disk_gib": 40, "poll_seconds": 5, "min_battery_percent": 20, "nice_level": 0}
    require(all(limits[key] == value for key, value in expected.items()), "Resource guard limits differ from the declared policy")
    require(limits["allow_battery"] is True, "Recorded battery policy differs")
    samples, _raw, issues = snapshot(directory / "resources.jsonl")
    require(samples and not issues, "Resource samples missing or malformed")
    for sample in samples:
        require(all(numeric(sample[key]) for key in ("rss_gib", "available_gib", "disk_free_gib", "elapsed_seconds")), "Invalid resource sample")
        require(sample["rss_gib"] <= job["memory_limit_gib"] and sample["available_gib"] >= 12 and sample["disk_free_gib"] >= 40, "A resource sample violates the guard policy")
    require(close(guard["peak_rss_gib"], max(sample["rss_gib"] for sample in samples)), "Guard peak RSS differs from resource samples")
    return guard


def inspect_matrix(matrix_path, root, tokenizer_factory):
    matrix = read_json(matrix_path)
    require(matrix["protocol"] == PROTOCOL and matrix["arms"] == ARMS, "Unexpected protocol/arms")
    config = matrix["config"]
    require(config == asdict(ScreenConfig(**config)), "Full explicit configuration required")
    ScreenConfig(**config).validate()
    require(type(matrix["limit"]) is int and matrix["limit"] > 0, "Positive question count required")
    jobs = matrix["jobs"]
    require(len(jobs) == 2 and len({job["id"] for job in jobs}) == 2 and len({job["model_revision"] for job in jobs}) == 2, "Exactly two unique checkpoint jobs required")
    require(all(job["id"] and Path(job["id"]).name == job["id"] for job in jobs), "Unsafe job ID")
    require(len({str(safe_source(root, job["run_dir"])) for job in jobs}) == 2, "Run directories must be distinct")
    inputs = {key: safe_source(root, matrix[key]) for key in ("dataset", "corpus", "data_manifest")}
    for key, digest in (("dataset", "questions_sha256"), ("corpus", "corpus_sha256")):
        require(sha(inputs[key].read_bytes()) == matrix[digest], "Matrix input hash differs: " + key)
    questions = [json.loads(line) for line in inputs["dataset"].read_bytes().splitlines()]
    require(len(questions) == matrix["limit"] and len({q["id"] for q in questions}) == len(questions), "Cohort must contain exactly all declared unique questions")
    require(all(q["split"] in {"dev", "development"} for q in questions), "This screen is development only")
    corpus = Corpus(inputs["corpus"])
    by_id = {question["id"]: question for question in questions}
    inspections, issues, source_identity, runtime_identity = [], [], None, None
    for job in jobs:
        inspection = {"job": job, "rows": [], "steps": {}, "sources": [], "checkpoint_files": [], "issues": []}
        inspections.append(inspection)
        try:
            run_dir = safe_source(root, job["run_dir"])
            manifest, runner = read_json(run_dir / "manifest.json"), read_json(run_dir / "summary.json")
            inspection.update(manifest=manifest, runner_summary=runner)
            identity = manifest["identity"]
            for key in ("protocol", "config", "arms", "limit", "questions_sha256", "corpus_sha256"):
                require(identity[key] == matrix[key], "Manifest/matrix mismatch: " + key)
            require(identity["model_revision"] == job["model_revision"], "Manifest model revision differs")
            require(safe_source(root, identity["model_path"]) == safe_source(root, job["model_path"]), "Manifest model path differs")
            backend = identity["backend"]
            runtime = {key: backend[key] for key in ("backend", "prefill_batch_size", "versions", "chat_template", "weight_dtype_policy")}
            if runtime_identity is None:
                runtime_identity = runtime
            require(runtime == runtime_identity, "Model jobs used different runtime/library policies")
            require(backend["model_revision"] == job["model_revision"] and safe_source(root, backend["model_path"]) == safe_source(root, job["model_path"]), "Backend checkpoint differs")
            require(backend["max_context"] == config["max_context_tokens"] and backend["wired_limit_modified"] is False, "Backend context/system memory policy differs")
            require(backend["chat_template"] == "checkpoint native; enable_thinking=False", "Native nonthinking template required")
            require(backend["prefill_batch_size"] <= 256 and backend["prefill_batch_size"] > 0, "Prefill batch bound differs")
            require(sha(json.dumps(backend["model_config"], sort_keys=True).encode()) == backend["config_sha256"], "Backend configuration hash differs")
            sources = identity["source_sha256"]
            require(bool(sources), "Executed source hashes required")
            require({"eval/controller_screen.py", "engine/text_backend_mlx.py"} <= sources.keys(), "Core generation sources are unpinned")
            if "source_files" in matrix:
                require(set(matrix["source_files"]) == sources.keys(), "Declared/executed source file sets differ")
            if source_identity is None:
                source_identity = sources
            require(sources == source_identity, "Model jobs executed different source sets or bytes")
            for name, digest in sources.items():
                source = safe_source(root, name)
                require(sha(source.read_bytes()) == digest, "Executed source hash differs: " + name)
                inspection["sources"].append(source)
                if "source_files" in matrix:
                    frozen = safe_source(root, run_dir / "source_snapshot" / name)
                    require(sha(frozen.read_bytes()) == digest, "Executed source snapshot differs: " + name)
                    inspection["sources"].append(frozen)
            require(manifest["matrix_sha256"] == sha(matrix_path.read_bytes()), "Executed matrix hash differs")
            execution_order = []
            for question in questions:
                arms = list(ARMS)
                order_seed = int(sha(f'{config["seed"]}:{question["id"]}:arm-order'.encode())[:8], 16)
                random.Random(order_seed).shuffle(arms)
                execution_order.extend(expected_id(config, question["id"], arm, job["model_revision"]) for arm in arms)
            require(manifest["execution_order"] == execution_order, "Condition execution order differs from fixed seed")
            if "preflight_dir" in job:
                preflight_dir = safe_source(root, job["preflight_dir"])
                preflight_path = preflight_dir / "preflight.json"
                preflight, prior_manifest = read_json(preflight_path), read_json(preflight_dir / "manifest.json")
                require(preflight["status"] == "passed" and all(value is True for value in preflight["checks"].values()) and preflight["checks"], "Synthetic preflight did not pass")
                require(preflight["backend"]["model_revision"] == job["model_revision"] and manifest["preflight_sha256"] == sha(preflight_path.read_bytes()), "Preflight model or result identity differs")
                require(prior_manifest["source_sha256"] == sources, "Preflight generation source differs")
                inspection["guard"] = inspect_guard(run_dir, job, 2700)
                inspect_guard(preflight_dir, job, 600)
                for name in ("preflight.json", "manifest.json", "checkpoint-receipt.json", "resources.jsonl", "guard-status.json", "process.log"):
                    source = preflight_dir / name
                    require(source.is_file(), "Missing preflight artifact: " + name)
                    inspection["sources"].append(source)
                for name, digest in prior_manifest["source_sha256"].items():
                    frozen = safe_source(root, preflight_dir / "source_snapshot" / name)
                    require(sha(frozen.read_bytes()) == digest, "Preflight source snapshot differs: " + name)
                    inspection["sources"].append(frozen)
            receipt_path, metadata = inspect_checkpoint(manifest, job, root)
            inspection["checkpoint_files"] = [receipt_path, *metadata]
            model_path = safe_source(root, job["model_path"])
            model_config, normalizations = expected_loaded_config(read_json(model_path / "config.json"),
                read_json(model_path / "generation_config.json"), backend["versions"])
            require(backend["model_config"] == model_config, "Backend configuration differs from verified checkpoint metadata")
            inspection["backend_config_normalizations"] = normalizations
            require(backend["model_type"] == model_config["model_type"] and backend["quantization"] == model_config.get("quantization", model_config.get("quantization_config")), "Backend architecture or quantization differs")
            require(backend["weight_dtype_policy"] == "unchanged checkpoint precision; no cast or dequantization", "Weight precision policy changed")
            checkpoint_files = [{"name": row["path"], "bytes": row["bytes"]} for row in manifest["checkpoint_receipt"]["verified_files"] if row["path"].startswith("model") and row["path"].endswith(".safetensors")]
            require(backend["checkpoint_files"] == sorted(checkpoint_files, key=lambda row: row["name"]), "Backend loaded checkpoint shard list differs")
            tokenizer = tokenizer_factory(safe_source(root, job["model_path"]))
            rows, _raw, row_issues = snapshot(run_dir / "results.jsonl")
            require(not row_issues, "Malformed or torn result ledger")
            inspection["raw_rows"] = rows
            expected = {expected_id(config, q["id"], arm, job["model_revision"]) for q in questions for arm in ARMS}
            require(len(rows) == len(expected) and len({row["id"] for row in rows}) == len(rows), "Incomplete or duplicate result conditions")
            require({row["id"] for row in rows} == expected, "Unexpected/missing condition IDs")
            require([row["id"] for row in rows] == execution_order, "Ledger condition order differs from executed manifest")
            require(runner["status"] == "complete" and runner["planned_conditions"] == len(expected) and runner["successful_conditions"] == len(expected) and runner["historical_errors"] == 0, "Runner summary incomplete or contains errors")
            for row in rows:
                inspection["steps"][row["id"]] = inspect_row(row, by_id[row["example_id"]], job, config, corpus, tokenizer, backend)
            inspection["rows"] = rows
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            issue = {"job_id": job["id"], "kind": "job_verification_failed", "error": str(exc)}
            inspection["issues"].append(issue)
            issues.append(issue)
    return matrix, inputs, questions, inspections, issues


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def aggregate(matrix, inspections):
    metrics, trajectories, pairs, conditions = [], [], [], []
    lookup = {}
    for inspection in inspections:
        job = inspection["job"]
        for row in inspection["rows"]:
            steps = inspection["steps"][row["id"]]
            diagnostic = {"job_id": job["id"], "model_label": job["model_label"], "condition_id": row["id"],
                "example_id": row["example_id"], "arm": row["arm"], "prediction": row["prediction"],
                "exact_match": row["scores"]["exact_match"], "f1": row["scores"]["f1"],
                "cold_end_to_end_ms": row["cold_end_to_end_ms"], "stop_reason": row["stop_reason"],
                "initial_support_ids": steps[0]["new_support_ids"], "final_support_ids": steps[-1]["cumulative_support_ids"],
                "new_support_after_initial": sum(len(step["new_support_ids"]) for step in steps[1:]), "steps": steps}
            conditions.append(diagnostic)
            lookup[job["id"], row["arm"], row["example_id"]] = row
        for arm in ARMS:
            rows = [row for row in inspection["rows"] if row["arm"] == arm]
            selected = [row for row in conditions if row["job_id"] == job["id"] and row["arm"] == arm]
            latencies = [row["cold_end_to_end_ms"] for row in rows]
            mean = lambda values: statistics.mean(values)
            metrics.append({"job_id": job["id"], "model_label": job["model_label"], "arm": arm, "n": len(rows),
                "exact_match": mean(row["scores"]["exact_match"] for row in rows), "exact_match_count": sum(row["scores"]["exact_match"] for row in rows),
                "f1": mean(row["scores"]["f1"] for row in rows), "cold_p50_ms": percentile(latencies, .5), "cold_p95_ms": percentile(latencies, .95),
                "mean_counts": {key: mean(row["counts"][key] for row in rows) for key in rows[0]["counts"]},
                "mean_component_ms": {key: mean(row["component_ms"][key] for row in rows) for key in rows[0]["component_ms"]},
                "new_support_after_initial_total": sum(row["new_support_after_initial"] for row in selected),
                "questions_with_new_support": sum(row["new_support_after_initial"] > 0 for row in selected),
                "mean_initial_support_coverage": mean(row["steps"][0]["support_coverage"] for row in selected),
                "mean_final_support_coverage": mean(row["support_annotation_coverage"] for row in rows),
                "full_support_questions": sum(row["support_annotation_coverage"] == 1 for row in rows),
                "mean_rounds": mean(row["executed_retrieval_rounds"] for row in rows),
                "stop_counts": dict(Counter(row["stop_reason"] for row in rows)),
                "invalid_actions": sum(row["invalid_actions"] for row in rows),
                "controller_cap_hits": sum(g["stats"]["stop_reason"] == "max_tokens" for row in rows for g in row["generations"] if g["phase"] == "decision"),
                "final_cap_hits": sum(row["generations"][-1]["stats"]["stop_reason"] == "max_tokens" for row in rows),
                "final_command_leaks": sum(row["raw_output"].strip().upper().startswith(("SEARCH:", "ANSWER:")) or row["raw_output"].strip() == "ANSWER" for row in rows),
                "max_mlx_peak_bytes": max(row["memory"]["peak_bytes"] for row in rows)})
            for round_number in range(1, matrix["config"]["max_rounds"] + 1):
                reached = [(row, step) for row in selected for step in row["steps"] if step["round"] == round_number]
                if not reached:
                    continue
                trajectories.append({"job_id": job["id"], "model_label": job["model_label"], "arm": arm, "round": round_number,
                    "n_reached": len(reached), "n_cohort": len(rows), "new_support_count": sum(len(step["new_support_ids"]) for _, step in reached),
                    "mean_cumulative_support_coverage_among_reached": mean(step["support_coverage"] for _, step in reached),
                    "repeated_stops": sum(row["stop_reason"] == "repeated_query_fallback_to_final" and row["steps"][-1]["round"] == round_number for row in selected)})
    jobs = matrix["jobs"]
    specifications = [("retrieval_arm", job, job, ARMS[0], ARMS[1]) for job in jobs]
    specifications += [("model", jobs[0], jobs[1], arm, arm) for arm in ARMS]
    for family, reference, comparison, reference_arm, comparison_arm in specifications:
        ids = sorted(qid for jid, arm, qid in lookup if jid == reference["id"] and arm == reference_arm)
        matches = [(lookup[reference["id"], reference_arm, qid], lookup[comparison["id"], comparison_arm, qid]) for qid in ids]
        pairs.append({"comparison_family": family, "reference_job": reference["id"], "comparison_job": comparison["id"],
            "reference_arm": reference_arm, "comparison_arm": comparison_arm, "n_paired": len(matches), "paired_example_ids": ids,
            "exact_match_difference": statistics.mean(b["scores"]["exact_match"] - a["scores"]["exact_match"] for a, b in matches),
            "mean_f1_difference": statistics.mean(b["scores"]["f1"] - a["scores"]["f1"] for a, b in matches),
            "recovered_example_ids": [a["example_id"] for a, b in matches if a["scores"]["exact_match"] == 0 and b["scores"]["exact_match"] == 1],
            "regressed_example_ids": [a["example_id"] for a, b in matches if a["scores"]["exact_match"] == 1 and b["scores"]["exact_match"] == 0],
            "mean_cold_latency_difference_ms": statistics.mean(b["cold_end_to_end_ms"] - a["cold_end_to_end_ms"] for a, b in matches),
            "median_paired_latency_ratio": statistics.median(b["cold_end_to_end_ms"] / a["cold_end_to_end_ms"] for a, b in matches)})
    return metrics, trajectories, pairs, conditions


def plots(metrics, trajectories, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2563eb", "#b45309"]
    labels = list(dict.fromkeys(row["model_label"] for row in metrics))
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for row in metrics:
        color = colors[labels.index(row["model_label"]) % len(colors)]
        ax.scatter(row["cold_p50_ms"] / 1000, row["exact_match"] * 100, color=color,
                   marker="o" if row["arm"] == "basic_rag" else "s", s=75)
        ax.annotate(f'{row["model_label"]} · {row["arm"]}\n{row["exact_match_count"]}/{row["n"]}',
                    (row["cold_p50_ms"] / 1000, row["exact_match"] * 100), xytext=(5, 7), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Cold end-to-end median latency (seconds)", ylabel="Exact match (%)", ylim=(-3, 110),
           title="Development screen: accuracy and local latency")
    ax.margins(x=.3)
    ax.grid(alpha=.2)
    fig.savefig(output / "accuracy-latency.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for index, label in enumerate(labels):
        rows = [row for row in trajectories if row["model_label"] == label and row["arm"] == "iterative_text"]
        axes[0].plot([row["round"] for row in rows], [row["new_support_count"] for row in rows], "o-", label=label, color=colors[index % 2])
        axes[1].plot([row["round"] for row in rows], [row["repeated_stops"] for row in rows], "o-", label=label, color=colors[index % 2])
        for row in rows:
            axes[0].annotate(f'n={row["n_reached"]}', (row["round"], row["new_support_count"]), xytext=(3, 7), textcoords="offset points", fontsize=8)
    for axis in axes:
        axis.set_xlabel("Retrieval round (round 1 is initial evidence)")
        axis.set_ylim(bottom=-.1)
        axis.grid(alpha=.2)
        axis.legend()
        axis.xaxis.get_major_locator().set_params(integer=True)
    axes[0].set_ylabel("New annotated support document IDs (total)")
    axes[1].set_ylabel("Questions stopped by repeated search")
    fig.suptitle("Observed trajectories; later rounds contain a selected subset")
    fig.canvas.draw()
    fig.savefig(output / "retrieval-productivity.png", dpi=180)
    plt.close(fig)


def build(matrix_path, output, root=REPOSITORY, no_plots=False, tokenizer_factory=None):
    root, matrix_path, output = Path(root).resolve(), Path(matrix_path).resolve(), Path(output).resolve()
    validate_report_output(output, *(root / name for name in ("runs", "models", "data", "scripts", "eval", "engine", "docs", "configs")))
    matrix, inputs, questions, inspections, issues = inspect_matrix(matrix_path, root, tokenizer_factory or tokenizer_for)
    output.mkdir(parents=True, exist_ok=True)
    receipts, archived = [], set()

    def pack(source, category):
        source = safe_source(root, source)
        if source in archived:
            return
        archived.add(source)
        relative = source.relative_to(root)
        receipts.append(archive(output, Path(category) / (str(relative) + ".gz"), source.read_bytes(), source, compress=True))

    pack(matrix_path, "matrix")
    for source in inputs.values():
        pack(source, "inputs")
    for inspection in inspections:
        run_dir = safe_source(root, inspection["job"]["run_dir"])
        for name in ("manifest.json", "results.jsonl", "summary.json", "checkpoint.json", "checkpoint-receipt.json", "errors.jsonl", "resources.jsonl", "guard-status.json", "process.log"):
            source = run_dir / name
            if source.exists():
                pack(source, "runs")
        for source in inspection["sources"] + inspection["checkpoint_files"]:
            pack(source, "provenance")
    try:
        data_manifest = read_json(inputs["data_manifest"])
        base = inputs["data_manifest"].parent if data_manifest.get("protocol") == "controller-scaling-data-v1" else root
        verified = set()
        for entry in data_manifest["outputs"]:
            source = safe_source(root, base / entry["path"])
            require(sha(source.read_bytes()) == entry["sha256"], "Data-selection provenance hash differs: " + entry["path"])
            verified.add(source)
            pack(source, "data-provenance")
        require({inputs["dataset"], inputs["corpus"]} <= verified, "Data-selection manifest does not pin both evaluated inputs")
        for name in matrix.get("provenance_files", []):
            pack(safe_source(root, name), "extra-provenance")
        for directory in sorted((root / "runs/large-controller").glob("download-*")):
            guard_path = directory / "guard-status.json"
            require(guard_path.is_file(), "Download guard is incomplete")
            guard = read_json(guard_path)
            require(guard["status"] == "complete" and guard["returncode"] == 0, "Download guard did not complete cleanly")
            for source in sorted(directory.rglob("*")):
                if source.is_file():
                    pack(source, "download-provenance")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append({"kind": "provenance_verification_failed", "error": str(exc)})
    for name in ("scripts/build_large_controller_report.py", "scripts/build_adaptive_report.py", "scripts/report_paths.py"):
        source = root / name
        if source.exists():
            pack(source, "report-source")
    complete = not issues
    metrics, trajectories, pairs, conditions = aggregate(matrix, inspections) if complete else ([], [], [], [])
    summary = {"report_version": VERSION, "report_status": "complete" if complete else "INCOMPLETE_OR_UNVERIFIED",
        "protocol": matrix["protocol"], "cohort": "development", "planned_conditions": len(questions) * len(ARMS) * len(matrix["jobs"]),
        "successful_verified_conditions": sum(len(inspection["rows"]) for inspection in inspections),
        "verification_issues": issues, "metrics": metrics, "paired_comparisons": pairs, "trajectories": trajectories,
        "model_setup": [{"job_id": item["job"]["id"], "backend": item.get("manifest", {}).get("identity", {}).get("backend"),
            "setup_timings_ms": item.get("manifest", {}).get("setup_timings_ms"), "runner_summary": item.get("runner_summary"),
            "resource_guard": item.get("guard"), "backend_config_normalizations": item.get("backend_config_normalizations", [])} for item in inspections],
        "limitations": ["All twelve previously inspected development questions; the 96-question test split is unused.",
            "Model family, dense/MoE architecture, tokenizer, training and BF16/5-bit precision change together; this is not an isolated parameter-count effect.",
            "Both arms use cold native text caches; no latent cache, independent KV relay or persistent-cache amortization is tested.",
            "Cold query means fresh prompt/cache within a process that loads its model once. No benchmark warmup runs; first-call compilation or paging remains included online.",
            "Annotated support presence does not establish that a needed relation exists, survives truncation, or is used.",
            "Per-round groups are selected by early stopping, not randomized round-budget comparisons.",
            "Runtime and token counts are measured local cost proxies; no dollar prices or energy measurements are inferred.",
            "Weights are not redistributed or rehashed during export; their pinned download receipts, metadata hashes and current file sizes are checked."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "conditions.json").write_text(json.dumps(conditions, indent=2, sort_keys=True) + "\n")
    for name, rows in (("metrics.csv", metrics), ("paired-comparisons.csv", pairs), ("trajectories.csv", trajectories)):
        csv_rows(output / name, rows)
    if complete:
        table = "# Large local controller development screen\n\nAll results use the same development questions. Differences combine model and precision changes.\n\n| Model | Arm | EM | F1 | Cold p50 / p95 (s) | New support IDs |\n| --- | --- | ---: | ---: | ---: | ---: |\n"
        for row in metrics:
            table += f'| {row["model_label"]} | {row["arm"]} | {row["exact_match_count"]}/{row["n"]} | {row["f1"]:.3f} | {row["cold_p50_ms"]/1000:.2f} / {row["cold_p95_ms"]/1000:.2f} | {row["new_support_after_initial_total"]} |\n'
        if not no_plots:
            plots(metrics, trajectories, output)
    else:
        table = "# Incomplete or unverified controller screen\n\nHeadline metrics and plots are withheld. See summary.json and preserved raw artifacts.\n"
        for name in ("accuracy-latency.png", "retrieval-productivity.png"):
            (output / name).unlink(missing_ok=True)
    (output / "tables.md").write_text(table)
    (output / "DATA_LICENSE.md").write_text("MuSiQue: Trivedi et al. (2022), https://github.com/StonyBrookNLP/musique, CC BY 4.0. Modified by deduplicating public development paragraphs into a shared corpus and selecting local question splits. Selection provenance and normalized inputs are archived; model weights are not redistributed.\n")
    (output / "artifact-manifest.json").write_text(json.dumps({"report_version": VERSION, "artifacts": receipts}, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("configs/large-controller-screen.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/2026-09-11-large-controller"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = build(args.matrix, args.output, no_plots=args.no_plots)
    print(json.dumps({key: summary[key] for key in ("report_status", "planned_conditions", "successful_verified_conditions", "verification_issues")}, indent=2))
    if args.require_complete and summary["report_status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
