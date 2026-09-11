#!/usr/bin/env python3
"""Verify this self-contained report's published bytes and frozen run identities."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def unpack(relative):
    return gzip.decompress((ROOT / relative).read_bytes())


def main():
    publication = json.loads((ROOT / 'publication-manifest.json').read_text())
    expected = set()
    for entry in publication['files']:
        path = (REPO / entry['path']).resolve()
        assert path.is_relative_to(REPO.resolve()) and not path.is_symlink(), entry['path']
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], entry['path']
        if path.is_relative_to(ROOT):
            expected.add(path)
    actual = {path.resolve() for path in ROOT.rglob('*') if path.is_file()}
    assert actual == expected | {ROOT / 'publication-manifest.json'}, 'Unexpected or missing package file'
    digests, archives = set(), 0
    for name in ('artifact-manifest.json', 'supplement-manifest.json'):
        for receipt in json.loads((ROOT / name).read_text())['artifacts']:
            path = (ROOT / receipt['artifact']).resolve()
            assert path.is_relative_to(ROOT) and not path.is_symlink()
            packed = path.read_bytes()
            assert len(packed) == receipt['artifact_bytes'] and sha(packed) == receipt['artifact_sha256'], receipt['artifact']
            raw = gzip.decompress(packed)
            assert len(raw) == receipt['source_bytes'] and sha(raw) == receipt['source_sha256'], receipt['source']
            digests.add(sha(raw))
            archives += 1
    summary = json.loads((ROOT / 'summary.json').read_text())
    assert summary['report_status'] == 'complete' and not summary['verification_issues']
    assert summary['successful_unique_conditions'] == 48
    original = {}
    for job in ('8b_short', '8b_thinking', '14b_short', '14b_thinking'):
        manifest = json.loads(unpack(f'artifacts/runs/{job}/manifest.json.gz'))
        for name, digest in manifest['identity']['source_sha256'].items():
            assert sha(unpack(f'artifacts/source/{job}/{name}.gz')) == digest
        assert sha(unpack(f'artifacts/inputs/{job}/dataset.jsonl.gz')) == manifest['identity']['questions_sha256']
        assert sha(unpack(f'artifacts/inputs/{job}/corpus.jsonl.gz')) == manifest['identity']['corpus_sha256']
        rows = [json.loads(line) for line in unpack(f'artifacts/runs/{job}/results.jsonl.gz').splitlines()]
        assert len(rows) == 12 and len({row['id'] for row in rows}) == 12
        assert all(row['status'] == 'ok' for row in rows)
        for row in rows:
            original[job, row['id']] = row
    replays = set()
    for model in ('8b', '14b'):
        base = f'supplement/runs/controller-scaling/final-diagnostic-v1/{model}'
        manifest = json.loads(unpack(base + '/manifest.json.gz'))
        assert set(manifest['source_sha256'].values()) <= digests, 'Missing executed diagnostic source bytes'
        rows = [json.loads(line) for line in unpack(base + '/results.jsonl.gz').splitlines()]
        calls = [json.loads(line) for line in unpack(base + '/generations.jsonl.gz').splitlines()]
        assert len(rows) == 24 and len(calls) == 48
        for row in rows:
            key = row['job_id'], row['source_result_id']
            assert key in original and key not in replays
            replay = row['conditions']['original']
            assert replay['output_token_ids'] == original[key]['output_token_ids']
            assert replay['raw_output'] == original[key]['raw_output']
            replays.add(key)
    assert len(replays) == len(original) == 48
    print(json.dumps({'status': 'verified', 'published_files': len(publication['files']),
        'gzip_receipts': archives, 'pilot_conditions': 48, 'original_exact_replays': 48,
        'final_diagnostic_generations': 96, 'weights': 'Not redistributed; model receipt hashes only.'}))


if __name__ == '__main__':
    main()
