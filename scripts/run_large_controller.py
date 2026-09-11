#!/usr/bin/env python3
"""Run one pinned model's synthetic preflight or complete development screen."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.adaptive_experiments import Corpus, append_record, read_questions, write_json
from eval.controller_screen import PROTOCOL, ScreenConfig, case_id, messages_for, run_case, seed_for
from engine.text_backend_mlx import MLXTextBackend


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def checkpoint_receipt(job):
    directory = ROOT / job['model_path']
    manifest_path = directory / 'download-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('download_complete') or manifest['revision'] != job['model_revision']:
        raise ValueError('Pinned checkpoint download is incomplete or has a different revision')
    records = []
    for entry in manifest['files']:
        if Path(entry['path']).name != entry['path']:
            raise ValueError('Unexpected checkpoint path')
        path = directory / entry['path']
        if path.is_symlink() or path.stat().st_size != entry['size_bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Checkpoint hash mismatch: ' + str(path))
        records.append({'path': entry['path'], 'bytes': entry['size_bytes'], 'sha256': entry['sha256']})
    actual = {p.name for p in directory.glob('*.safetensors')}
    expected = {row['path'] for row in records if row['path'].endswith('.safetensors')}
    if actual != expected:
        raise ValueError('Unrecorded or missing model weights')
    return {'manifest_path': str(manifest_path.relative_to(ROOT)), 'manifest_sha256': digest(manifest_path),
            'repo_id': manifest['repo_id'], 'revision': manifest['revision'], 'verified_files': records,
            'verified_bytes': sum(row['bytes'] for row in records)}


def preflight(backend, output, config):
    """Fictional facts only; no development or test questions are opened here."""
    question = 'Which sea does the water from Lark Creek eventually reach?'
    documents = [{'text': '\n<document>\nLark Creek flows into Bracken Stream. '
                  'Bracken Stream flows into River Alder. River Alder empties into the North Sea.\n</document>\n'}]
    final_messages = messages_for(question, documents, 'final')
    calls = []
    for _ in range(2):
        calls.append({'phase': 'final', 'messages': final_messages,
                      **backend.generate(final_messages, max_tokens=24)})
    decision_messages = messages_for(question, documents, 'decision', [question], 4)
    calls.append({'phase': 'decision', 'messages': decision_messages,
                  **backend.generate(decision_messages, max_tokens=32)})
    for _ in range(2):
        calls.append({'phase': 'sampled_decision', 'messages': decision_messages,
            **backend.generate(decision_messages, max_tokens=config.max_action_tokens,
                temperature=config.controller_temperature, top_p=config.controller_top_p,
                top_k=config.controller_top_k, presence_penalty=config.controller_presence_penalty,
                seed=seed_for(config, 'synthetic-lark-creek', 'decision', 1))})
    from eval.adaptive_experiments import parse_action
    checks = {'greedy_exact_repeat': calls[0]['output_token_ids'] == calls[1]['output_token_ids'],
              'grounded_synthetic_final': all('north sea' in call['raw_output'].lower() for call in calls[:2]),
              'valid_synthetic_decision': parse_action(calls[2]['raw_output'], allow_answer_payload=True)['valid'],
              'valid_sampled_decisions': all(parse_action(call['raw_output'], allow_answer_payload=True)['valid'] for call in calls[3:]),
              'seeded_exact_repeat': calls[3]['output_token_ids'] == calls[4]['output_token_ids'],
              'thinking_disabled': all(call['stats']['thinking'] is False for call in calls),
              'native_cache_present': bool(backend.native_cache_types),
              'token_accounting': all(call['stats']['prefill_tokens'] == len(call['prompt_token_ids']) and
                  call['stats']['generated_tokens'] == len(call['output_token_ids']) and
                  call['stats']['sampled_tokens_including_eos'] == len(call['stats']['sampled_token_ids'])
                  for call in calls)}
    # Oversized output reservation must fail before a forward pass.
    try:
        backend.generate(final_messages, max_tokens=backend.max_context)
    except ValueError as exc:
        checks['context_rejected'] = 'reservation' in str(exc)
    else:
        checks['context_rejected'] = False
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'calls': calls, 'backend': backend.metadata(), 'memory': backend.memory_stats()}
    write_json(output / 'preflight.json', result)
    if result['status'] != 'passed':
        raise RuntimeError('Synthetic preflight failed; preserve output and investigate before benchmark')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=ROOT / 'configs/large-controller-screen.json')
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    job = next(job for job in matrix['jobs'] if job['id'] == args.job_id)
    config = ScreenConfig(**matrix['config'])
    config.validate()
    if matrix['protocol'] != PROTOCOL or matrix['arms'] != ['basic_rag', 'iterative_text'] or matrix['limit'] != 12:
        raise ValueError('Unexpected protocol or condition matrix')
    output = ROOT / (job['preflight_dir'] if args.preflight else job['run_dir'])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('manifest.json', 'results.jsonl', 'preflight.json')):
        raise ValueError('Refusing to overwrite or resume a prior execution')
    started = time.perf_counter()
    started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    source_hashes = {name: digest(ROOT / name) for name in matrix['source_files']}
    for name in matrix['source_files']:
        target = output / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    receipt_started = time.perf_counter()
    receipt = checkpoint_receipt(job)
    checkpoint_verify_ms = (time.perf_counter() - receipt_started) * 1000
    write_json(output / 'checkpoint-receipt.json', receipt)
    backend = MLXTextBackend(ROOT / job['model_path'], revision=job['model_revision'],
        memory_limit_gib=job['memory_limit_gib'], max_context=config.max_context_tokens)
    if args.preflight:
        result = preflight(backend, output, config)
        write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'mode': 'synthetic_preflight',
            'started_at': started_at, 'checkpoint_receipt': receipt, 'source_sha256': source_hashes,
            'matrix_sha256': digest(args.matrix), 'elapsed_seconds': time.perf_counter() - started})
        print(json.dumps({'status': result['status'], 'job': job['id'], 'elapsed_seconds': time.perf_counter() - started}), flush=True)
        return
    preflight_path = ROOT / job['preflight_dir'] / 'preflight.json'
    prior = json.loads(preflight_path.read_text())
    if prior['status'] != 'passed' or prior['backend']['model_revision'] != job['model_revision']:
        raise ValueError('Matching successful synthetic preflight required')
    prior_manifest = json.loads((preflight_path.parent / 'manifest.json').read_text())
    for source in source_hashes:
        if prior_manifest['source_sha256'][source] != source_hashes[source]:
            raise ValueError('Generation source changed after preflight: ' + source)
    for key, hash_key in (('dataset', 'questions_sha256'), ('corpus', 'corpus_sha256')):
        if digest(ROOT / matrix[key]) != matrix[hash_key]:
            raise ValueError('Dataset identity mismatch')
    corpus = Corpus(ROOT / matrix['corpus'])
    questions = read_questions(ROOT / matrix['dataset'], corpus, matrix['limit'] + 1)
    if len(questions) != 12 or any(q['split'] != 'dev' for q in questions):
        raise ValueError('Exactly the twelve development questions are required')
    pairs = []
    for question in questions:
        arms = list(matrix['arms'])
        seed = int(hashlib.sha256(f"{config.seed}:{question['id']}:arm-order".encode()).hexdigest()[:8], 16)
        random.Random(seed).shuffle(arms)
        pairs.extend((question, arm) for arm in arms)
    manifest = {'identity': {'protocol': PROTOCOL, 'config': asdict(config), 'backend': backend.metadata(),
        'model_revision': job['model_revision'], 'model_path': job['model_path'], 'source_sha256': source_hashes,
        'questions_sha256': matrix['questions_sha256'], 'corpus_sha256': matrix['corpus_sha256'],
        'arms': matrix['arms'], 'limit': 12}, 'setup_timings_ms': {'checkpoint_verification': checkpoint_verify_ms,
        'model_load': backend.model_load_ms, 'corpus_load_and_index': corpus.preparation_ms},
        'checkpoint_receipt': receipt, 'started_at': started_at, 'preflight_sha256': digest(preflight_path),
        'matrix_sha256': digest(args.matrix), 'execution_order': [case_id(config, q['id'], arm, job['model_revision']) for q, arm in pairs],
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_json(output / 'manifest.json', manifest)
    completed, errors = 0, 0
    try:
        for question, arm in pairs:
            if time.perf_counter() - started > matrix['max_job_seconds']:
                raise TimeoutError('Whole-job runtime budget exceeded')
            try:
                row = run_case(backend, corpus, question, arm, config, job['model_revision'])
            except Exception as exc:
                errors += 1
                append_record(output / 'errors.jsonl', {'example_id': question['id'], 'arm': arm,
                    'exception': repr(exc), 'traceback': traceback.format_exc()})
                raise
            append_record(output / 'results.jsonl', row)
            completed += 1
            write_json(output / 'summary.json', {'status': 'running', 'planned_conditions': len(pairs),
                'successful_conditions': completed, 'historical_errors': errors,
                'elapsed_seconds': time.perf_counter() - started})
            print(json.dumps({'completed': completed, 'planned': len(pairs), 'example_id': question['id'],
                'arm': arm, 'exact_match': row['scores']['exact_match'], 'seconds': row['cold_end_to_end_ms'] / 1000,
                'stop_reason': row['stop_reason']}), flush=True)
        if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
            raise ValueError('Generation source changed during execution')
    except BaseException:
        write_json(output / 'summary.json', {'status': 'failed', 'planned_conditions': len(pairs),
            'successful_conditions': completed, 'historical_errors': errors,
            'elapsed_seconds': time.perf_counter() - started})
        raise
    write_json(output / 'summary.json', {'status': 'complete', 'planned_conditions': len(pairs),
        'successful_conditions': completed, 'historical_errors': errors,
        'elapsed_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
