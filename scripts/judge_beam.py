#!/usr/bin/env python3
"""Posthoc, condition-blind local rubric judge; never an official BEAM score."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.local_rubric_judge import judge_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-runtime-seconds", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from engine.backends_mlx import MLXBackend
    backend = MLXBackend(args.model, revision=args.revision)
    summary = judge_results(backend, args.dataset, args.results, args.output_dir,
                            max_tokens=args.max_tokens, seed=args.seed,
                            max_runtime_seconds=args.max_runtime_seconds, limit=args.limit)
    print(json.dumps(summary, indent=2))
    return 0 if summary["stop_reason"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
