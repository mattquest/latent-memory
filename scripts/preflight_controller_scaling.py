#!/usr/bin/env python3
"""Small real-weight controller mechanics check, without benchmark questions."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.adaptive_experiments import AdaptiveConfig, Evidence, PROLOGUE, Work, decision_text, decision_seed
from scripts.preflight_adaptive import block_digest, compare_logits, expect_cap_error

QUESTION = 'Which sea receives water downstream from Lark Creek?'
DOCUMENTS = [
    {'id': 'creek', 'title': 'Lark Creek', 'text': 'Lark Creek flows into Bracken Stream.'},
    {'id': 'stream', 'title': 'Bracken Stream', 'text': 'Bracken Stream joins River Alder.'},
    {'id': 'river', 'title': 'River Alder', 'text': 'River Alder drains into the North Sea.'},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--memory-limit-gib', type=float, default=32)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / 'preflight.json'
    if destination.exists():
        raise ValueError('Use a fresh preflight output directory')
    from engine.backends_mlx import MLXBackend
    backend = MLXBackend(args.model, revision=args.revision, memory_limit_gb=args.memory_limit_gib)
    result = {'scope': 'synthetic mechanics only; no benchmark questions or performance conclusion',
              'status': 'running', 'model': backend.metadata(), 'checks': [],
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                  'scripts/preflight_controller_scaling.py', 'scripts/preflight_adaptive.py',
                  'engine/backends_mlx.py', 'eval/adaptive_experiments.py')}, 'question': QUESTION,
              'documents': DOCUMENTS}
    started = time.perf_counter()
    try:
        for budget in (0, 8, 256):
            config = AdaptiveConfig('synthetic', 'synthetic', str(args.output_dir), limit=1,
                max_reasoning_tokens=budget, controller_temperature=.6, controller_top_p=.95,
                controller_top_k=20)
            config.validate()
            evidence = {}
            for arm in ('iterative_text', 'incremental_kv_bf16', 'relay_kv_bf16'):
                value = Evidence(backend, config, arm, Work(), backend.encode(PROLOGUE))
                value.initialize()
                value.add([(doc, 1.0) for doc in DOCUMENTS], 18, 3)
                evidence[arm] = value
            suffix = decision_text(QUESTION, [QUESTION], 4, thinking=bool(budget))
            seed = decision_seed(20260922, 'synthetic-controller-preflight', 'decision', 1)
            generated = {}
            for arm, value in evidence.items():
                before = block_digest(backend, value.prefix) if value.native else None
                raw, ids = value.generate(suffix, 32, True, controller_seed=seed)
                stats = dict(value.last_controller_decode)
                unchanged = not value.native or before == block_digest(backend, value.prefix)
                assert unchanged
                assert value.work.counts['internal_decoded_tokens'] == (
                    value.work.counts['controller_reasoning_tokens'] + len(ids))
                assert len(stats['reasoning_output_token_ids']) <= budget and len(ids) <= 32
                assert stats['context_reservation_tokens'] <= 8192
                generated[arm] = {'raw_action': raw, 'action_token_ids': ids,
                    'controller_decode': stats, 'counts': value.work.counts,
                    'prefix_immutable': unchanged}
            # Same seed and causal evidence should preserve the sampled trajectory
            # on this fixed small check; independent relay is not required to match.
            exact = all(generated['iterative_text'][key] == generated['incremental_kv_bf16'][key]
                        for key in ('raw_action', 'action_token_ids'))
            exact_reasoning = (generated['iterative_text']['controller_decode']['reasoning_output_token_ids'] ==
                               generated['incremental_kv_bf16']['controller_decode']['reasoning_output_token_ids'])
            assert exact and exact_reasoning, 'Text and causal-prefix sampled paths diverged in synthetic preflight'
            suffix_ids = backend.encode(suffix)
            import numpy as np
            reference = np.asarray(backend.next_logits(None, evidence['iterative_text'].token_ids + suffix_ids).astype(backend.mx.float32))
            candidate = np.asarray(backend.next_logits(evidence['incremental_kv_bf16'].prefix, suffix_ids).astype(backend.mx.float32))
            logits = compare_logits(reference, candidate)
            assert logits['scale_aware_close'] and logits['argmax_equal']
            result['checks'].append({'reasoning_budget': budget, 'config': asdict(config),
                'outputs': generated, 'sampled_causal_path_equal': exact and exact_reasoning,
                'logits': logits})
            destination.write_text(json.dumps(result, indent=2) + '\n')
            del evidence, generated, value
            backend.clear_cache()
        token = backend.encode('a')[0]
        capped = expect_cap_error(lambda: backend.decode_with_reasoning(None, [token] * 8190,
                                   max_reasoning_tokens=8, max_tokens=32))
        assert capped['rejected']
        result['oversized_reservation'] = capped
        result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        result['elapsed_seconds_excluding_load'] = time.perf_counter() - started
        result['memory'] = backend.memory_stats()
        destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(result['checks']), 'output': str(destination)}))


if __name__ == '__main__':
    main()
