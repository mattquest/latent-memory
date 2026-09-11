#!/usr/bin/env python3
"""Summarize all archived final-system pairs; preserve original pilot headlines."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent

def lines(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]

all_pairs = []
for model in ('8b', '14b'):
    base = root / 'supplement/runs/controller-scaling/final-diagnostic-v1' / model
    summary = json.loads(gzip.decompress((base / 'summary.json.gz').read_bytes()))
    assert summary['status'] == 'complete' and summary['successful_pairs'] == 24
    rows = lines(base / 'results.jsonl.gz')
    calls = lines(base / 'generations.jsonl.gz')
    assert len(rows) == 24 and len(calls) == 48
    assert len({(row['source_result_id'], row['condition']) for row in calls}) == 48
    call_lookup = {(row['source_result_id'], row['condition']): row for row in calls}
    for row in rows:
        for name, value in row['conditions'].items():
            assert call_lookup[row['source_result_id'], name]['output_token_ids'] == value['output_token_ids']
            assert hashlib.sha256(json.dumps(value['prompt_token_ids']).encode()).hexdigest() == value['prompt_token_ids_sha256']
    all_pairs.extend(rows)
assert len(all_pairs) == 48 and len({(row['job_id'], row['source_result_id']) for row in all_pairs}) == 48
metrics = []
for job in ('8b_short', '8b_thinking', '14b_short', '14b_thinking'):
    pilot = {row['id']: row for row in lines(root / f'artifacts/runs/{job}/results.jsonl.gz')}
    rows = [row for row in all_pairs if row['job_id'] == job]
    assert len(rows) == 12 and {row['source_result_id'] for row in rows} == set(pilot)
    for row in rows:
        source, original = pilot[row['source_result_id']], row['conditions']['original']
        assert source['output_token_ids'] == original['output_token_ids']
        assert source['raw_output'] == original['raw_output']
        assert source['scores'] == original['scores']
    for condition in ('original', 'final_only_system'):
        values = [row['conditions'][condition] for row in rows]
        metrics.append({'job_id': job, 'condition': condition, 'n': 12,
            'correct': int(sum(value['scores']['exact_match'] for value in values)),
            'f1': statistics.mean(value['scores']['f1'] for value in values),
            'median_final_synthesis_ms': statistics.median(value['final_synthesis_ms'] for value in values),
            'mean_prefill_tokens': statistics.mean(value['work']['prefill_tokens'] for value in values),
            'mean_generated_tokens': statistics.mean(value['work']['generated_tokens'] for value in values),
            'search_commands': sum(bool(re.match(r'^SEARCH\s*:', value['prediction'], re.I)) for value in values),
            'correct_question_ids': [row['example_id'] for row in rows if row['conditions'][condition]['scores']['exact_match']]})
result = {'scope': 'Posthoc saved-evidence final-system ablation; final-stage timings only; twelve unique questions reused across four configurations.',
          'source_conditions': 48, 'final_generations': 96, 'original_exact_token_and_text_replays': 48,
          'metrics': metrics, 'paired_predictions': [{
              'job_id': row['job_id'], 'example_id': row['example_id'],
              'original': row['conditions']['original']['prediction'],
              'final_only_system': row['conditions']['final_only_system']['prediction'],
              'score_difference': row['score_difference']} for row in all_pairs]}
(root / 'final-diagnostic-summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'pairs': len(all_pairs), 'metrics': metrics}))
