#!/usr/bin/env python3
"""Recompute descriptive pilot diagnostics from archived raw ledgers (CPU only)."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import statistics


def read_json(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.report_dir
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['report_status'] == 'complete' and summary['successful_unique_conditions'] == 48
    matrix = read_json(root / 'artifacts/matrix/controller-scaling-development.json.gz')
    data = [json.loads(line) for line in gzip.decompress((root / 'artifacts/inputs/8b_short/dataset.jsonl.gz').read_bytes()).splitlines()]
    questions = {row['id']: row for row in data}
    result = {'scope': 'Descriptive development diagnostics; no causal inference from support annotations or selected examples.',
              'jobs': [], 'questions': []}
    per_question = {}
    for job in matrix['jobs']:
        job_id = job['id']
        rows = [json.loads(line) for line in gzip.decompress((root / f'artifacts/runs/{job_id}/results.jsonl.gz').read_bytes()).splitlines()]
        assert len(rows) == len({row['example_id'] for row in rows}) == 12
        assert set(questions) == {row['example_id'] for row in rows}
        assert all(row['status'] == 'ok' for row in rows)
        decisions = [event for row in rows for event in row['trace'] if event['event'] == 'decision']
        initial, final, gains = [], [], []
        for row in rows:
            support = set(questions[row['example_id']]['supporting_context_ids'])
            retrievals = [event for event in row['trace'] if event['event'] == 'retrieval']
            exposed_initial = support & set(retrievals[0]['accumulated_document_ids'])
            exposed_final = support & set(row['evidence_ids'])
            initial.append(len(exposed_initial) / len(support))
            final.append(len(exposed_final) / len(support))
            gains.append(len(exposed_final - exposed_initial))
            per_question.setdefault(row['example_id'], {'question': row['question'], 'answers': row['answers'], 'jobs': {}})['jobs'][job_id] = {
                'prediction': row['prediction'], 'scores': row['scores'],
                'rounds': row['executed_retrieval_rounds'], 'stop': row['stop_reason'],
                'initial_support': len(exposed_initial), 'final_support': len(exposed_final), 'total_support': len(support),
                'new_support_ids': sorted(exposed_final - exposed_initial),
                'search_queries': row['search_queries'],
                'raw_actions': [event['raw_action'] for event in row['trace'] if event['event'] == 'decision']}
        guard = read_json(root / f'artifacts/runs/{job_id}/guard-status.json.gz')
        resources = [json.loads(line) for line in gzip.decompress((root / f'artifacts/runs/{job_id}/resources.jsonl.gz').read_bytes()).splitlines()]
        result['jobs'].append({'job_id': job_id, 'n': 12, 'correct': int(sum(row['scores']['exact_match'] for row in rows)),
            'mean_f1': statistics.mean(row['scores']['f1'] for row in rows),
            'controller_decisions': len(decisions),
            'reasoning_cap_decisions': sum(event['controller_decode']['reasoning_cap_reached'] for event in decisions),
            'action_cap_decisions': sum(event['controller_decode']['action_stop_reason'] == 'max_tokens' for event in decisions),
            'final_cap_questions': sum(row['generation_stop'] == 'max_tokens' for row in rows),
            'answer_payload_variants': sum(event['action'].get('format_variant') == 'answer_with_ignored_payload' for event in decisions),
            'invalid_actions': sum(row['invalid_actions'] for row in rows),
            'final_search_commands': sum(bool(re.match(r'^SEARCH\s*:', row['prediction'], re.I)) for row in rows),
            'final_unknown': sum(row['prediction'] == 'UNKNOWN' for row in rows),
            'stop_reasons': dict(Counter(row['stop_reason'] for row in rows)),
            'mean_rounds': statistics.mean(row['executed_retrieval_rounds'] for row in rows),
            'round_counts': dict(Counter(row['executed_retrieval_rounds'] for row in rows)),
            'mean_initial_support_coverage': statistics.mean(initial),
            'mean_final_support_coverage': statistics.mean(final),
            'complete_support_questions': sum(value == 1 for value in final),
            'total_new_support_after_initial': sum(gains),
            'questions_gaining_support': sum(value > 0 for value in gains),
            'guard': guard, 'resource_sample_count': len(resources)})
    result['questions'] = [{'example_id': key, **value} for key, value in per_question.items()]
    (root / 'pilot-diagnostics.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = ['# All twelve development questions', '',
        'Original end-to-end predictions, without replacing answers with controller thoughts or diagnostic replays. Support is annotated document exposure; it does not establish understanding.', '',
        '| Question | Gold aliases | 8B short | 8B reasoning | 14B short | 14B reasoning |',
        '| --- | --- | --- | --- | --- | --- |']
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    for row in result['questions']:
        cells = [row['question'], '; '.join(row['answers'])]
        for job_id in ['8b_short', '8b_thinking', '14b_short', '14b_thinking']:
            entry = row['jobs'][job_id]
            cells.append(f"{entry['prediction']} ({entry['final_support']}/{entry['total_support']} supports; {entry['rounds']} rounds)")
        lines.append('| ' + ' | '.join(map(cell, cells)) + ' |')
    lines += ['', 'Exact question IDs, searches, actions and new supporting IDs are in [pilot-diagnostics.json](pilot-diagnostics.json).', '']
    (root / 'per-question.md').write_text('\n'.join(lines))
    print(json.dumps({'jobs': [{key: value for key, value in row.items() if key != 'guard'} for row in result['jobs']]}))

if __name__ == '__main__':
    main()
