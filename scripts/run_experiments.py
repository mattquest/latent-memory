#!/usr/bin/env python3
"""Run bounded real Qwen3 KV-relay diagnostics, or aggregate an existing run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.real_experiments import ExperimentConfig, _atomic_json, load_records, run_suite, summarize


def csv_values(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiments", default="E1,E3,E4,E5")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hops", default="1,3,5,10,20")
    parser.add_argument("--equal-hops", type=int, default=3)
    parser.add_argument("--bridge-ratios", default="0,0.1,0.2,1")
    parser.add_argument("--bridge-ratio", type=float, default=0.2)
    parser.add_argument("--relay-precision", choices=("bf16", "int8", "int4"), default="int8")
    parser.add_argument("--latent-precisions", default="",
                        help="Optional comma-separated precisions for E1/E2/E6 latent arms, e.g. bf16,int8")
    parser.add_argument("--precisions", default="bf16,int8,int4")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-document-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=48)
    parser.add_argument("--max-relay-tokens", type=int, default=192)
    parser.add_argument("--max-runtime-seconds", type=float, default=18000)
    parser.add_argument("--evidence-per-hop", type=int, default=2)
    parser.add_argument("--schedule", choices=("lexical", "source", "oracle"), default="lexical")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        directory = Path(args.output_dir)
        summary = summarize(load_records(directory / "results.jsonl"), args.seed)
        _atomic_json(directory / "summary.recomputed.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    if not args.dataset:
        parser.error("--dataset is required unless --summarize-only")
    config = ExperimentConfig(
        dataset=args.dataset, output_dir=args.output_dir,
        experiments=csv_values(args.experiments), limit=args.limit, seed=args.seed,
        hops=csv_values(args.hops, int), equal_hops=args.equal_hops,
        bridge_ratios=csv_values(args.bridge_ratios, float), bridge_ratio=args.bridge_ratio,
        relay_precision=args.relay_precision,
        latent_precisions=csv_values(args.latent_precisions),
        precisions=csv_values(args.precisions), max_context_tokens=args.max_context_tokens,
        max_document_tokens=args.max_document_tokens, max_answer_tokens=args.max_answer_tokens,
        max_relay_tokens=args.max_relay_tokens, max_runtime_seconds=args.max_runtime_seconds,
        evidence_per_hop=args.evidence_per_hop, schedule=args.schedule,
        repeats=args.repeats, resume=not args.no_resume)
    config.validate()
    from engine.backends_mlx import MLXBackend
    backend = MLXBackend(args.model, revision=args.revision)
    summary = run_suite(backend, config)
    print(json.dumps({"summary": str(Path(args.output_dir) / "summary.json"),
                      "stop_reason": summary["stop_reason"],
                      "completed_unique_runs": summary["completed_unique_runs"],
                      "planned_runs": summary["planned_runs"]}, indent=2))
    return 0 if summary["stop_reason"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
