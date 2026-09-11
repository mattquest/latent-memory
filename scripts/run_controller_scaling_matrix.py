#!/usr/bin/env python3
"""Execute the declared controller pilot serially under per-job resource guards."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.adaptive_experiments import AdaptiveConfig, write_json
from scripts.build_controller_scaling_report import read_matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--job-seconds', type=int, default=1800)
    parser.add_argument('--allow-battery', action='store_true')
    args = parser.parse_args()
    if args.job_seconds < 120:
        parser.error('Per-job outer limit must be at least 120 seconds')
    matrix_path = args.matrix.resolve()
    matrix = read_matrix(matrix_path)
    shared = matrix['shared_config']
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Matrix progress belongs in a repository runs directory')
    output.mkdir(parents=True, exist_ok=True)
    for key, expected in [('dataset', 'questions_sha256'), ('corpus', 'corpus_sha256')]:
        if sha(ROOT / matrix[key]) != matrix[expected]:
            raise ValueError(f'Frozen {key} input changed')
    receipt = {'matrix': str(matrix_path.relative_to(ROOT)), 'matrix_sha256': sha(matrix_path),
        'source_sha256': {name: sha(ROOT / name) for name in (
            'engine/backends_mlx.py', 'eval/adaptive_experiments.py', 'eval/real_experiments.py',
            'eval/retrieval.py', 'scripts/run_adaptive.py', 'scripts/run_guarded.py',
            'scripts/run_controller_scaling_matrix.py')},
        'questions_sha256': matrix['questions_sha256'], 'corpus_sha256': matrix['corpus_sha256']}
    frozen = output / 'freeze.json'
    if frozen.exists() and json.loads(frozen.read_text()) != receipt:
        raise ValueError('Existing matrix freeze differs; use a new declared output root')
    write_json(frozen, receipt)
    statuses = []
    cli_fields = ('dataset', 'corpus', 'limit', 'seed', 'initial_top_k', 'documents_per_round',
                  'max_documents', 'max_document_tokens', 'max_evidence_tokens',
                  'max_model_context_tokens', 'max_action_tokens', 'max_answer_tokens',
                  'bridge_ratio', 'controller_temperature', 'controller_top_p', 'controller_top_k')
    for name, expected in [('max_question_tokens', 256), ('rrf_k', 60), ('rerank_candidates', 20)]:
        if shared[name] != expected:
            raise ValueError(f'Runner CLI has no override for {name}; expected {expected}')
    for job in matrix['jobs']:
        for name, expected in receipt['source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f'Generation source changed after freeze: {name}')
        memory = job.get('memory_limit_gib', 32)
        if not 0 < memory <= 48:
            raise ValueError('Invalid declared model memory limit')
        config = AdaptiveConfig(**shared, output_dir=job['run_dir'],
                                max_reasoning_tokens=job['max_reasoning_tokens'])
        config.validate()
        command = [sys.executable, 'scripts/run_guarded.py', '--output', job['run_dir'],
                   '--max-seconds', str(args.job_seconds), '--max-rss-gib', str(memory),
                   '--poll-seconds', '5', '--nice-level', '0', '--min-battery-percent', '20']
        if args.allow_battery:
            command.append('--allow-battery')
        command += ['--', sys.executable, '-u', 'scripts/run_adaptive.py', '--model', job['model_path'],
                    '--revision', job['model_revision'], '--output-dir', job['run_dir'],
                    '--memory-limit-gib', str(memory), '--max-reasoning-tokens', str(job['max_reasoning_tokens']),
                    '--arms', ','.join(shared['arms']), '--rounds', ','.join(map(str, shared['rounds'])),
                    '--max-runtime-seconds', str(args.job_seconds - 60), '--max-case-seconds', '300']
        for key in cli_fields:
            command += ['--' + key.replace('_', '-'), str(shared[key])]
        write_json(output / 'matrix-status.json', {'status': 'running', 'active_job': job['id'],
                   'finished_jobs': statuses, 'updated_at': time.time()})
        print(json.dumps({'starting_job': job['id'], 'command': command}), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        summary_path = ROOT / job['run_dir'] / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        ok = (completed.returncode == 0 and summary.get('stop_reason') == 'complete'
              and summary.get('unresolved_failed_runs') == 0)
        statuses.append({'job': job['id'], 'returncode': completed.returncode,
                         'successful_runs': summary.get('successful_unique_runs'), 'complete': ok})
        if not ok:
            write_json(output / 'matrix-status.json', {'status': 'stopped', 'finished_jobs': statuses})
            return 2
    write_json(output / 'matrix-status.json', {'status': 'complete', 'finished_jobs': statuses,
               'successful_runs': sum(row['successful_runs'] for row in statuses), 'completed_at': time.time()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
