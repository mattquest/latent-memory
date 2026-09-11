"""Report semantics only; these fixtures are not model experiment results."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_report", Path(__file__).parents[1] / "scripts" / "build_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_record(example="example-1", arm="latent", repeat=0, em=1):
    case = {"experiment": "E1", "arm": arm, "hops": 3, "control": "correct",
            "bridge_ratio": .2, "precision": "bf16", "update_policy": "all", "repeat": repeat}
    return {"id": f"{example}-{arm}-{repeat}", "status": "ok", "dataset": "fixture",
            "example_id": example, "case": case, "prediction": "fixture only",
            "scores": {"exact_match": em, "f1": em}, "latency_ms": 100 + repeat,
            "counts": {"host_output_tokens": 2}, "cache": {},
            "report_job": "test", "report_job_status": "complete"}


def test_partial_jsonl_does_not_modify_run_and_archives_exact_bytes(tmp_path):
    source = tmp_path / "results.jsonl"
    raw = b'{"id":"ok"}\n{"id":'
    source.write_bytes(raw)
    rows, snapshot, issues = report.read_jsonl_snapshot(source)
    assert rows == [{"id": "ok"}]
    assert issues[0]["kind"] == "partial_trailing_line"
    receipt = report.archive_bytes(tmp_path / "output", Path("results.jsonl.gz"), snapshot, source, True)
    artifact = tmp_path / "output" / receipt["artifact"]
    assert gzip.decompress(artifact.read_bytes()) == source.read_bytes() == raw
    assert receipt["source_sha256"] == report.digest(raw)


def test_completion_requires_planned_unique_results_not_stale_summary(tmp_path):
    source = tmp_path / "job"
    source.mkdir()
    record = fixture_record()
    (source / "results.jsonl").write_text(json.dumps(record) + "\n")
    (source / "summary.json").write_text(json.dumps({"stop_reason": "complete", "planned_runs": 2}))
    jobs, _, _, _ = report.collect_jobs(source, {}, tmp_path / "out")
    assert jobs[0]["status"] == "incomplete"
    assert jobs[0]["successful_unique_runs"] == 1


def test_repeats_do_not_inflate_accuracy_and_pairing_uses_common_examples():
    rows = [fixture_record(repeat=0), fixture_record(repeat=1),
            fixture_record(arm="text", em=0), fixture_record(example="unpaired", em=1)]
    groups = report.metric_groups(rows)
    latent = next(group for group in groups if group["arm"] == "latent")
    assert latent["n_examples"] == 2 and latent["n_runs"] == 3
    pairs = report.paired_groups(rows)
    pair = next(pair for pair in pairs if pair["comparator"] == "text/bf16/correct")
    assert pair["n_paired_examples"] == 1
    assert pair["exact_match_difference"] == 1
    assert pair["paired_bootstrap_ci95_low"] is None


def test_beam_proxy_has_no_population_interval_and_no_paired_em_claim():
    row = fixture_record()
    row["dataset"] = "beam"
    groups = report.metric_groups([row])
    assert groups[0]["exact_match_ci95_low"] is None
    assert "proxy" in groups[0]["metric_role"]
    assert report.paired_groups([row]) == []


def test_truncation_distinguishes_delivered_evidence_from_unused_candidates():
    rows = [fixture_record(example="unused-truncation"), fixture_record(example="delivered-truncation")]
    rows[0].update(truncated_document_ids=["unused"], evidence_ids=["delivered"])
    rows[1].update(truncated_document_ids=["delivered", "unused"], evidence_ids=["delivered"])
    metrics = report.metric_groups(rows)
    assert metrics[0]["fraction_document_truncation"] == .5
    assert metrics[0]["fraction_any_candidate_truncation"] == 1


def test_e6_pairs_current_minus_all_within_arm_precision_and_category():
    old, current, unpaired, other_precision = [fixture_record(example=name, em=em) for name, em in
                                             [("shared", 0), ("shared", 1), ("unpaired", 1), ("shared", 0)]]
    for row in (old, current, unpaired, other_precision):
        row["case"]["experiment"] = "E6"
        row["category"] = "updated"
    current["case"]["update_policy"] = unpaired["case"]["update_policy"] = "current"
    other_precision["case"]["precision"] = "int8"
    pairs = [p for p in report.paired_groups([old, current, unpaired, other_precision]) if p["contrast_type"] == "update_policy"]
    assert len(pairs) == 2
    assert {p["stratification"] for p in pairs} == {"all", "category"}
    assert all(p["n_paired_examples"] == 1 and p["exact_match_difference"] == 1 for p in pairs)
    assert all(p["baseline"] == "latent/bf16/correct/current" for p in pairs)
    for row in (old, current):
        row["dataset"] = "beam"
    assert report.paired_groups([old, current]) == []


def test_partial_beam_judgments_are_withheld_from_default_comparison_table():
    group = {"job": "beam-job", "job_status": "complete", "category": "all", "experiment": "E1",
             "variant": "latent/bf16", "hops": 1, "n_candidate_answers": 2, "n_valid_judgments": 1,
             "n_invalid_judgments": 0, "n_pending_judgments": 1, "n_conversations": 1,
             "mean_criterion_fraction": .75, "all_criteria_pass_rate": 0}
    snapshot = {"generated_utc": "fixture", "status": "generation_complete", "jobs": [], "metrics": [],
                "paired": [], "judge": {"groups": [group]}, "figures": []}
    default = report.tables(snapshot)
    assert "75.0%" not in default and "Partial valid" in default
    partial = report.tables(snapshot, include_incomplete=True)
    assert "75.0%" in partial and "PARTIAL / INVALID" in partial
    group["n_valid_judgments"], group["n_pending_judgments"] = 2, 0
    assert "75.0%" in report.tables(snapshot)


def test_judge_matches_prediction_digest_and_counts_pending(tmp_path):
    source = fixture_record()
    source["dataset"] = "beam"
    source["example_metadata"] = {"conversation_id": "conversation-1"}
    judge = tmp_path / "judge"
    judge.mkdir()
    stale = {"id": "stale", "source_result_id": source["id"], "candidate_sha256": "wrong",
             "status": "ok", "judgment": {"criterion_fraction": 1, "all_criteria_met": True}}
    (judge / "judgments.jsonl").write_text(json.dumps(stale) + "\n")
    result, _ = report.collect_judge(judge, [source], tmp_path / "output")
    assert result["unmatched_judgment_records"] == 1
    assert result["groups"][0]["n_pending_judgments"] == 1
    assert result["groups"][0]["mean_criterion_fraction"] is None
    assert result["groups"][0]["n_conversations"] == 1


def test_exact_input_archives_check_generation_hash_and_preserve_bytes(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "normalized.jsonl"
    raw = b'{"id":"fixture","metadata":{}}\n'
    source.write_bytes(raw)
    job = {"job": "test", "generation_identity": {"config": {"dataset": str(source)}, "dataset_sha256": report.digest(raw)}}
    receipts, issues = report.collect_input_artifacts([job], data, tmp_path / "output")
    assert not issues
    receipt = next(row for row in receipts if row.get("verified_against_generation_manifest"))
    assert gzip.decompress((tmp_path / "output" / receipt["artifact"]).read_bytes()) == raw
    assert receipt["used_by_jobs"] == ["test"]
    source.write_text('{"changed":true}\n')
    receipts, issues = report.collect_input_artifacts([job], data, tmp_path / "mismatch-output")
    assert issues[0]["kind"] == "executed_input_hash_mismatch"
    assert not any(row.get("verified_against_generation_manifest") for row in receipts)


def git_fixture(tmp_path):
    import subprocess
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
    source = repository / "source.py"
    original = b"# exact executed source\nvalue = 1\n"
    source.write_bytes(original)
    subprocess.run(["git", "add", "source.py"], cwd=repository, check=True)
    subprocess.run(["git", "-c", "user.name=Artifact fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--quiet", "-m", "Fixture source"], cwd=repository, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    return repository, source, original, revision


def test_exact_historical_source_is_recovered_after_working_file_changes(tmp_path):
    repository, source, original, revision = git_fixture(tmp_path)
    source.write_text("# newer maintenance code\nvalue = 2\n")
    expected = report.digest(original)
    jobs = [{"job": name, "generation_identity": {"source_code_sha256": {"source.py": expected}},
             "generation_machine": {"git_head": revision, "code_sha256": {"source.py": expected}}}
            for name in ("first", "later")]
    receipts, issues, versions = report.collect_executable_artifacts(jobs, tmp_path / "release", repository)
    assert not issues and len(receipts) == 1 and len(versions) == 1
    assert (tmp_path / "release" / receipts[0]["artifact"]).read_bytes() == original
    assert receipts[0]["resolution"] == "matching_recorded_git_blob"
    assert receipts[0]["resolved_git_head"] == revision
    assert receipts[0]["used_by_jobs"] == ["first", "later"]
    assert source.read_text() == "# newer maintenance code\nvalue = 2\n"


def test_matching_current_source_and_machine_hash_fallback_need_no_git(tmp_path):
    repository = tmp_path / "non_git_repository"
    repository.mkdir()
    source = repository / "source.py"
    source.write_bytes(b"# exact current source\n")
    jobs = [{"job": "legacy", "generation_identity": {"config": {}},
             "generation_machine": {"git_head": None, "code_sha256": {"source.py": report.digest(source.read_bytes())}}}]
    receipts, issues, versions = report.collect_executable_artifacts(jobs, tmp_path / "release", repository)
    assert not issues and versions[0]["complete"]
    assert receipts[0]["resolution"] == "matching_working_file"


def test_git_blob_mismatch_cannot_be_exported_as_executed_source(tmp_path):
    repository, source, original, revision = git_fixture(tmp_path)
    expected = report.digest(b"# unavailable uncommitted execution\n")
    job = {"job": "missing", "generation_identity": {"source_code_sha256": {"source.py": expected}},
           "generation_machine": {"git_head": revision, "code_sha256": {"source.py": expected}}}
    receipts, issues, versions = report.collect_executable_artifacts([job], tmp_path / "release", repository)
    assert receipts == [] and versions[0]["complete"] is False
    assert issues[0]["kind"] == "executed_source_unavailable"
    assert "Git blob SHA-256 mismatch" in issues[0]["error"]


def test_require_complete_fails_when_executable_bytes_are_unavailable(tmp_path, monkeypatch):
    import sys
    data = tmp_path / "data"
    data.mkdir()
    dataset = data / "fixture.jsonl"
    dataset.write_text('{"id":"fixture","metadata":{}}\n')
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.jsonl").write_text(json.dumps(fixture_record()) + "\n")
    (run / "summary.json").write_text('{"stop_reason":"complete","planned_runs":1}\n')
    (run / "manifest.json").write_text(json.dumps({"identity": {
        "config": {"dataset": str(dataset)}, "dataset_sha256": report.digest(dataset.read_bytes()),
        "source_code_sha256": {"unavailable_fixture_source.py": "0" * 64}},
        "machine": {"git_head": "0" * 40}}))
    output = tmp_path / "release"
    monkeypatch.setattr(sys, "argv", ["build_report.py", "--run-root", str(run), "--no-matrix", "--data-dir", str(data),
        "--judge-dir", str(tmp_path / "absent-judge"), "--output-dir", str(output), "--no-plots", "--require-complete"])
    assert report.main() == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "INCOMPLETE_SNAPSHOT"
    assert any(issue["kind"] == "executed_source_unavailable" for issue in summary["issues"])
