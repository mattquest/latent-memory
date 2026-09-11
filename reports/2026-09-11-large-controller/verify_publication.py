#!/usr/bin/env python3
"""Verify archived publication bytes; optionally check today's root README too."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def verify_projection(originals, digests):
    directory = ROOT / 'provider-latency'
    raw_workloads = (directory / 'workloads.json').read_bytes()
    raw_profiles = (directory / 'profiles.json').read_bytes()
    workloads = json.loads(raw_workloads)
    projection = json.loads((directory / 'projections.json').read_text())
    profiles = json.loads(raw_profiles)['profiles']
    observations = (directory / 'published-source-observations.json').read_bytes()
    audit = json.loads((directory / 'projection-audit.json').read_text())
    assert audit['status'] == 'pass' and set(audit['file_sha256'].values()) <= digests
    render = json.loads((directory / 'provider-render-receipt.json').read_text())
    assert sha((directory / 'render_provider_report.py').read_bytes()) == render['renderer_sha256']
    for section in ('input_sha256', 'output_sha256'):
        for filename, digest in render[section].items():
            assert sha((directory / filename).read_bytes()) == digest, filename
    assert projection['workloads_sha256'] == sha(raw_workloads)
    assert projection['profiles_sha256'] == sha(raw_profiles)
    assert projection['extractor_sha256'] == workloads['extractor_sha256']
    assert workloads['extractor_sha256'] in digests
    assert workloads['conditions'] == 96 and workloads['distinct_questions'] == 12
    assert len(workloads['runs']) == 6 and len(profiles) == 14
    source_observations = {item['id']: item for item in json.loads(observations)['sources']}
    for item in profiles:
        assert item['snapshot_sha256'] == sha(observations)
        observed = source_observations[item['source_id']]
        assert item['model_id'] == observed['model_id']
        assert any(pair['provider'] == item['provider'] and pair['ttft_s'] == item['ttft_s']
                   and pair['reported_tps'] == item['reported_tps'] for pair in observed['provider_pairs'])
    enriched = [{**item, 'snapshot_verified': True} for item in profiles]
    assert projection['profiles'] == enriched
    lookup = {}
    question_ids = set()
    for run in workloads['runs']:
        assert {item['sha256'] for item in run['source_sha256']} <= digests
        for receipt in run['input_receipts']:
            raw = originals[receipt['path']]
            assert len(raw) == receipt['bytes'] and sha(raw) == receipt['sha256']
        results_receipt = next(item for item in run['input_receipts'] if item['path'].endswith('/results.jsonl'))
        rows = [json.loads(line) for line in originals[results_receipt['path']].splitlines()]
        assert len(rows) == len(run['workloads'])
        for workload, row in zip(run['workloads'], rows):
            assert workload['source_row_sha256'] == sha(canonical(row))
            assert workload['case_id'] == row['id']
            assert workload['observed_prediction'] == row['prediction']
            assert workload['observed_scores_unchanged'] == row['scores']
            key = (run['label'], workload['case_id'])
            assert key not in lookup
            lookup[key] = workload
            question_ids.add(workload['example_id'])
    assert len(lookup) == 96 and len(question_ids) == 12
    assert sum(len(item['calls']) for item in lookup.values()) == 254
    assert len(projection['conditions']) == 408 and not projection['unprofiled_runs']
    profile_lookup = {item['id']: item for item in profiles}
    groups = {}
    seen = set()
    scenarios = ('decode_rate_assumption_s', 'inclusive_rate_proxy_s')
    for result in projection['conditions']:
        profile = profile_lookup[result['profile_id']]
        workload = lookup[result['run_label'], result['case_id']]
        key = (result['profile_id'], result['run_label'], result['case_id'])
        assert key not in seen
        seen.add(key)
        assert result['run_label'] in profile['applies_to']
        assert result['observed_prediction'] == workload['observed_prediction']
        assert result['observed_scores_unchanged'] == workload['observed_scores_unchanged']
        assert result['outer_host_s'] == workload['local_observed_ms']['outer_host_work'] / 1000
        assert len(result['calls']) == len(workload['calls'])
        for call, source in zip(result['calls'], workload['calls']):
            generated = sum(source[k] for k in ('reasoning_generated_tokens', 'action_generated_tokens', 'final_generated_tokens'))
            assert call['generated_tokens_used'] == generated
            assert call[scenarios[0]] == profile['ttft_s'] + max(generated - 1, 0) / profile['reported_tps']
            expected_b = max(profile['ttft_s'], generated / profile['reported_tps']) if profile['metric_semantics'] == 'ambiguous_reported_throughput' else None
            assert call[scenarios[1]] == expected_b
        for scenario in scenarios:
            expected = None if result['calls'][0][scenario] is None else result['outer_host_s'] + sum(call[scenario] for call in result['calls'])
            assert result[scenario] == expected
        groups.setdefault((result['profile_id'], result['run_label'], result['arm']), []).append(result)
    assert len(groups) == len(projection['summaries']) == 34
    for summary in projection['summaries']:
        rows = groups[summary['profile_id'], summary['run_label'], summary['arm']]
        assert len(rows) == summary['conditions'] == 12
        for scenario in scenarios:
            values = sorted(row[scenario] for row in rows if row[scenario] is not None)
            if not values:
                assert summary[scenario] is None
                continue
            position = (len(values) - 1) * .95
            low = math.floor(position)
            p95 = values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)
            for field, expected in [('mean', sum(values) / len(values)), ('p50', statistics.median(values)), ('p95', p95)]:
                assert math.isclose(summary[scenario][field], expected, rel_tol=1e-12)
    assert len(projection['paired_summaries']) == 17
    for paired in projection['paired_summaries']:
        left_run, left_arm = paired['baseline'].rsplit('/', 1)
        right_run, right_arm = paired['comparison'].rsplit('/', 1)
        left = {r['example_id']: r for r in groups[paired['profile_id'], left_run, left_arm]}
        right = {r['example_id']: r for r in groups[paired['profile_id'], right_run, right_arm]}
        assert set(left) == set(right) == set(paired['example_ids']) and paired['n_paired'] == 12
        for scenario in scenarios:
            if next(iter(left.values()))[scenario] is None:
                assert paired[scenario] is None
                continue
            difference = statistics.mean(right[key][scenario] - left[key][scenario] for key in left)
            ratio = statistics.median(right[key][scenario] / left[key][scenario] for key in left)
            assert math.isclose(paired[scenario]['mean_paired_difference_s'], difference, rel_tol=1e-12)
            assert math.isclose(paired[scenario]['median_paired_comparison_over_baseline'], ratio, rel_tol=1e-12)
    return len(projection['conditions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-current-readme', action='store_true')
    args = parser.parse_args()
    publication = json.loads((ROOT / 'publication-manifest.json').read_text())
    expected = set()
    for entry in publication['files']:
        path = ROOT / entry['path']
        assert path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink(), entry['path']
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and sha(raw) == entry['sha256'], entry['path']
        expected.add(path.resolve())
    actual = {path.resolve() for path in ROOT.rglob('*') if path.is_file()}
    assert actual == expected | {ROOT / 'publication-manifest.json'}, 'Unexpected or missing package file'
    readme = publication['readme_at_publication']
    frozen_readme = (ROOT / readme['snapshot']).read_bytes()
    assert len(frozen_readme) == readme['bytes'] and sha(frozen_readme) == readme['sha256']
    if args.check_current_readme:
        assert (REPO / 'README.md').read_bytes() == frozen_readme, 'Root README changed since this report was published'
    originals, digests = {}, set()
    receipts = json.loads((ROOT / 'artifact-manifest.json').read_text())['artifacts']
    for receipt in receipts:
        path = ROOT / receipt['artifact']
        assert path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
        packed = path.read_bytes()
        assert len(packed) == receipt['artifact_bytes'] and sha(packed) == receipt['artifact_sha256']
        raw = gzip.decompress(packed) if receipt['compression'] == 'gzip' else packed
        assert len(raw) == receipt['source_bytes'] and sha(raw) == receipt['source_sha256']
        originals[receipt['source']] = raw
        digests.add(sha(raw))

    def original(relative):
        matches = {raw for source, raw in originals.items() if source.endswith('/' + relative)}
        assert len(matches) == 1, relative
        return next(iter(matches))

    summary = json.loads((ROOT / 'summary.json').read_text())
    assert summary['report_status'] == 'complete' and not summary['verification_issues']
    assert summary['successful_verified_conditions'] == summary['planned_conditions'] == 48
    matrix = json.loads(original('configs/large-controller-screen.json'))
    total = 0
    identities = []
    for job in matrix['jobs']:
        base = job['run_dir']
        manifest = json.loads(original(base + '/manifest.json'))
        identity = manifest['identity']
        identities.append(identity['source_sha256'])
        assert set(identity['source_sha256'].values()) <= digests, 'Missing executed source bytes'
        assert sha(original(matrix['dataset'])) == identity['questions_sha256']
        assert sha(original(matrix['corpus'])) == identity['corpus_sha256']
        rows = [json.loads(line) for line in original(base + '/results.jsonl').splitlines()]
        assert len(rows) == 24 and len({row['id'] for row in rows}) == 24
        assert [row['id'] for row in rows] == manifest['execution_order']
        assert all(row['status'] == 'ok' and row['model_revision'] == job['model_revision'] for row in rows)
        assert {row['arm'] for row in rows} == {'basic_rag', 'iterative_text'}
        assert all(sum(row['arm'] == arm for row in rows) == 12 for arm in matrix['arms'])
        for directory in (base, job['preflight_dir']):
            guard = json.loads(original(directory + '/guard-status.json'))
            assert guard['status'] == 'complete' and guard['returncode'] == 0 and guard['reason'] is None
        assert sha(original(job['preflight_dir'] + '/preflight.json')) == manifest['preflight_sha256']
        total += len(rows)
    assert identities[0] == identities[1]
    assert len(summary['metrics']) == len(summary['paired_comparisons']) == 4
    projected = verify_projection(originals, digests)
    print(json.dumps({'status': 'verified', 'published_files': len(publication['files']),
        'archive_receipts': len(receipts), 'conditions': total,
        'projected_condition_profile_pairs': projected,
        'current_readme_checked': args.check_current_readme,
        'weights': 'Not redistributed; archived publisher and execution hash receipts only.'}))


if __name__ == '__main__':
    main()
