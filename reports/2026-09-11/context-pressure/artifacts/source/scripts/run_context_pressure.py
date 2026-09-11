#!/usr/bin/env python3
"""Run the separate matched-trace context-pressure diagnostic after adaptive test."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.context_pressure import PressureConfig, run_pressure_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/adaptive_musique_test.jsonl")
    parser.add_argument("--corpus", default="data/adaptive_musique_corpus.jsonl")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision")
    parser.add_argument("--evidence-budget", type=int, default=1536)
    parser.add_argument("--summary-budget", type=int, default=384)
    parser.add_argument("--expected-questions", type=int, default=96)
    parser.add_argument("--source-rounds", type=int, default=5)
    parser.add_argument("--max-answer-tokens", type=int, default=48)
    parser.add_argument("--max-runtime-seconds", type=float, default=18000)
    parser.add_argument("--max-case-seconds", type=float, default=240)
    args = parser.parse_args()
    config = PressureConfig(dataset=args.dataset, corpus=args.corpus, source_dir=args.source_dir,
        output_dir=args.output_dir, evidence_budget=args.evidence_budget, summary_budget=args.summary_budget,
        expected_questions=args.expected_questions, source_rounds=args.source_rounds, max_answer_tokens=args.max_answer_tokens,
        max_runtime_seconds=args.max_runtime_seconds, max_case_seconds=args.max_case_seconds)
    config.validate()
    revision = args.revision
    if revision is None:
        revision = json.loads((Path(args.source_dir) / "manifest.json").read_text())["identity"]["backend"]["model_revision"]
    from engine.backends_mlx import MLXBackend
    backend = MLXBackend(args.model, revision=revision, max_context=config.max_model_context)
    result = run_pressure_suite(backend, config)
    print(json.dumps({"status": result["status"], "successful_runs": result["successful_runs"], "planned_runs": result["planned_runs"]}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
