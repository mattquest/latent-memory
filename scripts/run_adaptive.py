#!/usr/bin/env python3
"""Run or summarize cold global-corpus adaptive RAG comparisons."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.adaptive_experiments import AdaptiveConfig, CORE_ARMS, read_records, run_adaptive_suite, save_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--reranker", help="Optional local Qwen3-Reranker-0.6B directory")
    parser.add_argument("--reranker-revision", default="e61197ed45024b0ed8a2d74b80b4d909f1255473")
    parser.add_argument("--dataset")
    parser.add_argument("--corpus", default="data/adaptive_musique_corpus.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", help="Comma-separated arms; default core six plus reranked_rag when --reranker is supplied")
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", default="5")
    parser.add_argument("--initial-top-k", type=int, default=6)
    parser.add_argument("--documents-per-round", type=int, default=3)
    parser.add_argument("--max-documents", type=int, default=18)
    parser.add_argument("--max-document-tokens", type=int, default=300)
    parser.add_argument("--max-evidence-tokens", type=int, default=6144)
    parser.add_argument("--max-model-context-tokens", type=int, default=8192)
    parser.add_argument("--max-action-tokens", type=int, default=32)
    parser.add_argument("--max-reasoning-tokens", type=int, default=0,
                        help="Optional transient controller thinking budget before the separate action budget")
    parser.add_argument("--memory-limit-gib", type=float, default=32,
                        help="MLX memory allowance, at most 48 GiB; default 32")
    parser.add_argument("--controller-temperature", type=float, default=0.0)
    parser.add_argument("--controller-top-p", type=float, default=1.0)
    parser.add_argument("--controller-top-k", type=int, default=0)
    parser.add_argument("--max-answer-tokens", type=int, default=48)
    parser.add_argument("--bridge-ratio", type=float, default=.2)
    parser.add_argument("--max-runtime-seconds", type=float, default=18000)
    parser.add_argument("--max-case-seconds", type=float, default=180)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        directory = Path(args.output_dir)
        previous = json.loads((directory / "summary.json").read_text()) if (directory / "summary.json").exists() else {}
        summary = save_summary(directory, read_records(directory / "results.jsonl"), args.seed,
                               stop_reason=previous.get("stop_reason", "artifact_snapshot"),
                               planned_runs=previous.get("planned_runs"))
    else:
        if not args.dataset:
            parser.error("--dataset is required for generation")
        arms = tuple(args.arms.split(",")) if args.arms else CORE_ARMS + (("reranked_rag",) if args.reranker else ())
        if "reranked_rag" in arms and not args.reranker:
            parser.error("reranked_rag requires --reranker")
        config = AdaptiveConfig(dataset=args.dataset, corpus=args.corpus, output_dir=args.output_dir,
            arms=arms, limit=args.limit, seed=args.seed,
            rounds=tuple(int(value) for value in args.rounds.split(",")),
            initial_top_k=args.initial_top_k, documents_per_round=args.documents_per_round, max_documents=args.max_documents,
            max_document_tokens=args.max_document_tokens, max_evidence_tokens=args.max_evidence_tokens,
            max_model_context_tokens=args.max_model_context_tokens,
            max_action_tokens=args.max_action_tokens, max_reasoning_tokens=args.max_reasoning_tokens,
            controller_temperature=args.controller_temperature, controller_top_p=args.controller_top_p,
            controller_top_k=args.controller_top_k,
            max_answer_tokens=args.max_answer_tokens,
            bridge_ratio=args.bridge_ratio, max_runtime_seconds=args.max_runtime_seconds,
            max_case_seconds=args.max_case_seconds, resume=not args.no_resume)
        config.validate()
        from engine.backends_mlx import MLXBackend
        load_started = time.perf_counter()
        backend = MLXBackend(args.model, revision=args.revision, max_context=args.max_model_context_tokens,
                             memory_limit_gb=args.memory_limit_gib)
        backend.synchronize()
        setup_timings_ms = {"generator_load_and_model_validation_ms": (time.perf_counter() - load_started) * 1000}
        reranker = None
        if args.reranker:
            from engine.reranker_mlx import MLXQwen3Reranker
            load_started = time.perf_counter()
            reranker = MLXQwen3Reranker(args.reranker, revision=args.reranker_revision)
            reranker.synchronize()
            setup_timings_ms["reranker_load_and_checkpoint_verification_ms"] = (time.perf_counter() - load_started) * 1000
        summary = run_adaptive_suite(backend, config, reranker=reranker, setup_timings_ms=setup_timings_ms)
    print(json.dumps({"output_dir": args.output_dir, "successful_runs": summary["successful_unique_runs"],
                      "stop_reason": summary["stop_reason"]}, indent=2))
    return 0 if summary["stop_reason"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
