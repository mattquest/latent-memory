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
