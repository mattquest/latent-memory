#!/usr/bin/env python3
"""Bounded real-weight mechanics checks on a fixed synthetic retrieval trace.

This is a development implementation check, never a benchmark result. It reads
no development/test questions and does not change prompts or numerical limits.
Run serially under scripts/run_guarded.py; this script loads the normal two
models once and keeps only a small per-trace cache.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.adaptive_experiments import (
    AdaptiveConfig, Evidence, PROLOGUE, Work, decision_text, document_text,
    final_text, write_json,
)
from eval.real_experiments import machine_metadata

VERSION = "adaptive-real-mechanics-preflight-v1"
TOLERANCES = {"maximum_relative_rmse": .01, "maximum_absolute_floor": .25,
              "maximum_absolute_rms_multiplier": .03}
QUESTION = "Which sea receives water downstream from Lark Creek?"
QUERIES = [QUESTION, "Bracken Stream downstream river", "River Alder mouth sea"]
TRACE = [
    [{"id": "creek", "title": "Lark Creek", "text": "Lark Creek flows into Bracken Stream."},
     {"id": "orchard", "title": "Orchard", "text": "The orchard grows pears and apples."},
     {"id": "bus", "title": "Town transport", "text": "The green bus stops at the market."},
     {"id": "clock", "title": "Clock tower", "text": "The clock tower has a copper roof."},
     {"id": "garden", "title": "Garden", "text": "The garden opens at nine in the morning."},
     {"id": "pottery", "title": "Pottery", "text": "The pottery workshop makes blue bowls."}],
    [{"id": "stream", "title": "Bracken Stream", "text": "Bracken Stream joins the River Alder."},
     {"id": "music", "title": "Music hall", "text": "The music hall hosts a string quartet."},
     {"id": "library", "title": "Library", "text": "The library has three reading rooms."}],
    [{"id": "river", "title": "River Alder", "text": "The River Alder drains into the North Sea."},
     {"id": "bakery", "title": "Bakery", "text": "The bakery sells rye bread."},
     {"id": "museum", "title": "Museum", "text": "The museum displays old maps."}],
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compare_logits(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("Logit vectors must have matching nonempty shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite logits")
    difference = np.abs(reference - candidate)
    scale = float(np.sqrt(np.mean(reference ** 2)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    relative_rmse = rmse / max(scale, 1e-12)
    limit = max(TOLERANCES["maximum_absolute_floor"], TOLERANCES["maximum_absolute_rms_multiplier"] * scale)
    reference_argmax, candidate_argmax = int(np.argmax(reference)), int(np.argmax(candidate))
    # Stable descending order chooses the smallest token ID when BF16 logits tie,
    # matching the actual greedy argmax operation used by the native decoder.
    ref_top = np.argsort(-reference, kind="stable")[:10]
    candidate_top = np.argsort(-candidate, kind="stable")[:10]
    return {"vocabulary_size": int(reference.size), "reference_rms": scale,
            "rmse": rmse, "relative_rmse": relative_rmse,
            "max_absolute_difference": float(difference.max()),
            "p95_absolute_difference": float(np.quantile(difference, .95)),
            "maximum_absolute_limit": limit,
            "reference_argmax": reference_argmax, "candidate_argmax": candidate_argmax,
            "reference_top_two_margin": float(reference[ref_top[0]] - reference[ref_top[1]]),
            "candidate_top_two_margin": float(candidate[candidate_top[0]] - candidate[candidate_top[1]]),
            "reference_top10": [{"token_id": int(i), "logit": float(reference[i])} for i in ref_top],
            "candidate_top10": [{"token_id": int(i), "logit": float(candidate[i])} for i in candidate_top],
            "argmax_equal": reference_argmax == candidate_argmax,
            "scale_aware_close": bool(relative_rmse <= TOLERANCES["maximum_relative_rmse"]
                                       and difference.max() <= limit)}


def block_digest(backend, block):
    """Hash every K/V value, not only the token IDs or Python object identity."""
    backend.synchronize()
    result = hashlib.sha256(json.dumps({"tokens": block.token_ids, "lengths": block.block_lengths,
        "quant": block.quant, "metadata": block.metadata}, sort_keys=True).encode())
    for array in block.arrays():
        result.update(str((array.shape, array.dtype)).encode())
        # BF16 -> FP32 is exact and permits a portable NumPy byte representation.
        values = np.asarray(array.astype(backend.mx.float32)) if array.dtype == backend.mx.bfloat16 else np.asarray(array)
        result.update(values.tobytes())
    return result.hexdigest()


def expect_cap_error(fn):
    try:
        fn()
    except ValueError as exc:
        return {"rejected": True, "exception": type(exc).__name__, "message": str(exc)}
    return {"rejected": False, "message": "Oversized context was accepted"}


def execute(backend, reranker, output, result):
    config = AdaptiveConfig("synthetic_mechanics_only", "forced_trace_no_search", str(output), limit=1)
    config.validate()
    result["config"] = asdict(config)
    result["forced_trace"] = TRACE
    result["question"] = QUESTION
    result["queries"] = QUERIES
    tokens = backend.encode(PROLOGUE)
    text = Evidence(backend, config, "iterative_text", Work(), list(tokens))
    incremental = Evidence(backend, config, "incremental_kv_bf16", Work(), list(tokens))
    incremental.initialize()
    snapshots = [(incremental.prefix, block_digest(backend, incremental.prefix))]
    arrays = {}
    result["rounds"] = []
    for round_index, documents in enumerate(TRACE, 1):
        hits = [(document, 1.0) for document in documents]
        text.add(hits, config.max_documents, len(documents))
        incremental.add(hits, config.max_documents, len(documents))
        exact_tokens = tuple(text.token_ids) == incremental.prefix.token_ids == tuple(incremental.token_ids)
        snapshots.append((incremental.prefix, block_digest(backend, incremental.prefix)))
        suffix_text = (decision_text(QUESTION, QUERIES[:round_index], 5 - round_index)
                       if round_index < len(TRACE) else final_text(QUESTION))
        suffix = backend.encode(suffix_text)
        full_prompt = text.token_ids + suffix
        started = time.perf_counter()
        reference = np.asarray(backend.next_logits(None, full_prompt).astype(backend.mx.float32)).copy()
        reference_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        candidate = np.asarray(backend.next_logits(incremental.prefix, suffix).astype(backend.mx.float32)).copy()
        candidate_ms = (time.perf_counter() - started) * 1000
        output_ids = backend.decode(incremental.prefix, suffix, max_tokens=2)
        backend.synchronize()
        measurements = compare_logits(reference, candidate)
        preserved = [block_digest(backend, block) == before for block, before in snapshots]
        row = {"round": round_index, "accumulated_document_ids": list(text.document_ids),
               "exact_retained_tokens": exact_tokens, "retained_token_ids": list(text.token_ids),
               "suffix_token_ids": suffix, "retained_tokens": len(text.token_ids),
               "full_input_plus_normal_output_reservation": len(full_prompt) + (32 if round_index < 3 else 48),
               "all_source_snapshots_immutable": all(preserved), "snapshot_checks": preserved,
               "full_prefill_logits_ms": reference_ms, "incremental_logits_ms": candidate_ms,
               "transient_decode_token_ids": output_ids, "measurements": measurements}
        row["passed"] = (exact_tokens and all(preserved) and measurements["argmax_equal"]
                         and measurements["scale_aware_close"]
                         and row["full_input_plus_normal_output_reservation"] <= 8192)
        result["rounds"].append(row)
        arrays[f"round{round_index}_fulltext_logits"] = reference
        arrays[f"round{round_index}_incremental_logits"] = candidate
        write_json(output / "preflight.json", result)
    # Independent-document reuse has different attention history. Check immutability,
    # token geometry and bridge accounting; do not demand numerical equivalence.
    independent = [backend.prefill(backend.encode(document_text(TRACE[i][0]))[:300]) for i in range(3)]
    prologue = backend.prefill(tokens)
    sources = [prologue, *independent]
    fingerprints = [block_digest(backend, block) for block in sources]
    relayed = backend.concat(sources, bridge_ratio=.2)
    relay_fingerprint = block_digest(backend, relayed)
    packed = backend.quantize(relayed, bits=8)
    packed_fingerprint = block_digest(backend, packed)
    backend.decode(packed, backend.encode(final_text(QUESTION)), max_tokens=2)
    unchanged = [block_digest(backend, block) == before for block, before in zip(sources, fingerprints)]
    result["independent_relay"] = {"source_snapshots_immutable": unchanged,
        "relayed_snapshot_immutable": block_digest(backend, relayed) == relay_fingerprint,
        "packed_snapshot_immutable": block_digest(backend, packed) == packed_fingerprint,
        "token_ids_match_source_concatenation": relayed.token_ids == tuple(token for block in sources for token in block.token_ids),
        "bridge_tokens_recomputed": relayed.metadata.get("bridge_tokens_recomputed"),
        "bf16_nbytes": relayed.nbytes, "int8_nbytes": packed.nbytes,
        "numerical_equivalence_required": False}
    valid_token = backend.encode("a")[0]
    oversized = [valid_token] * 8193
    result["context_caps"] = {"backend_configured_max_context": backend.max_context,
        "prefill_8193": expect_cap_error(lambda: backend.prefill(oversized)),
        "decode_8192_plus_one_output": expect_cap_error(lambda: backend.decode(None, oversized[:8192], max_tokens=1))}
    original_tokens = text.token_ids
    text.token_ids = [valid_token] * 8192
    result["context_caps"]["harness_complete_suffix_and_output_reservation"] = expect_cap_error(
        lambda: text.generate(final_text(QUESTION), 48, controller=False))
    text.token_ids = original_tokens
    documents = [{"id": "unrelated", "title": "Bread", "text": "Sourdough bread is made by fermenting flour and water."},
                 {"id": "relevant", "title": "Willowford river", "text": "The River Alder flows through Willowford."}]
    query = "Which river flows through Willowford?"
    scores = [float(value) for value in reranker.score(query, documents)]
    result["reranker"] = {"question": query, "documents": documents, "scores": scores,
        "stats": reranker.last_stats, "relevant_above_unrelated": len(scores) == 2 and all(np.isfinite(scores)) and scores[1] > scores[0]}
    np.savez_compressed(output / "logits.npz", **arrays)
    result["full_logits_artifact"] = {"path": "logits.npz", "sha256": digest((output / "logits.npz").read_bytes())}
    relay = result["independent_relay"]
    return (all(row["passed"] for row in result["rounds"]) and all(unchanged)
            and relay["relayed_snapshot_immutable"] and relay["packed_snapshot_immutable"]
            and relay["token_ids_match_source_concatenation"] and relay["bridge_tokens_recomputed"] > 0
            and relay["int8_nbytes"] < relay["bf16_nbytes"]
            and backend.max_context == 8192
            and all(value["rejected"] for value in result["context_caps"].values() if isinstance(value, dict))
            and result["reranker"]["relevant_above_unrelated"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--reranker", default="models/Qwen3-Reranker-0.6B")
    parser.add_argument("--reranker-revision", default="e61197ed45024b0ed8a2d74b80b4d909f1255473")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/adaptive-preflight"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "preflight.json").exists():
        parser.error("Preflight output already exists; use a fresh output directory to preserve measurements")
    result = {"version": VERSION, "status": "running", "tolerances": TOLERANCES,
              "scope": "Fixed synthetic mechanics trace only; no benchmark questions, performance inference, or prompt tuning.",
              "tolerance_interpretation": "Predeclared empirical development tolerances, not numerical error guarantees.",
              "machine": machine_metadata(), "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in (
                  "scripts/preflight_adaptive.py", "eval/adaptive_experiments.py", "engine/backends_mlx.py", "engine/reranker_mlx.py")}}
    started = time.perf_counter()
    try:
        from engine.backends_mlx import MLXBackend
        from engine.reranker_mlx import MLXQwen3Reranker
        backend = MLXBackend(args.model, revision=args.revision, max_context=8192)
        reranker = MLXQwen3Reranker(args.reranker, revision=args.reranker_revision)
        result["setup_ms"] = (time.perf_counter() - started) * 1000
        result["backend_identity"] = backend.metadata()
        result["reranker_identity"] = reranker.metadata()
        passed = execute(backend, reranker, args.output_dir, result)
        result["status"] = "passed" if passed else "failed"
        result["memory"] = backend.memory_stats()
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output_dir / "preflight.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
