"""Artifact-only failure-analysis fixtures; no model or harness is imported."""
import hashlib
import json

import pytest

from scripts.analyze_failure_modes import analyze, markdown


def record(name, example, experiment="E1", arm="latent", precision="bf16", hops=3,
           actual=3, ratio=0.2, selected=None, truncated=(), em=0, f1=None,
           prediction="UNKNOWN", capped=False, repeat=0):
    return {"id": name, "status": "ok", "example_id": example["id"],
            "dataset": example["dataset"], "prediction": prediction,
            "case": {"experiment": experiment, "arm": arm, "precision": precision,
                     "hops": hops, "bridge_ratio": ratio, "control": "correct",
                     "update_policy": "all", "repeat": repeat},
            "scores": {"exact_match": em, "f1": em if f1 is None else f1},
            "evidence_ids": ["gold"] if selected is None else selected,
            "truncated_document_ids": list(truncated),
            "evidence_hops_executed": actual,
            "generation_stop": "max_tokens" if capped else "eos",
            "counts": {"bridge_recomputed_tokens": int(100 * ratio)},
            "cache": {"metadata": {"bridge_ratio_actual": ratio}}}


def make_job(root, name, examples, records):
    job = root / name
    job.mkdir()
    dataset = root.parent / (name + "-dataset.jsonl")
    content = "".join(json.dumps(row) + "\n" for row in examples)
    dataset.write_text(content)
    (job / "manifest.json").write_text(json.dumps({"identity": {
        "config": {"dataset": str(dataset)},
        "dataset_sha256": hashlib.sha256(content.encode()).hexdigest()}}))
    (job / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (job / "summary.json").write_text(json.dumps({"stop_reason": "complete", "planned_runs": len(records)}))
    return job, dataset


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "saved-run"
    root.mkdir()
    qa1 = {"id": "qa-1", "dataset": "hotpotqa", "supporting_context_ids": ["gold"]}
    qa2 = {"id": "qa-2", "dataset": "hotpotqa", "supporting_context_ids": ["gold"]}
    qa_rows = [
        record("l1", qa1, em=1, prediction="answer", truncated=["unused"]),
        record("d1", qa1, arm="direct_text", truncated=["unused"]),
        record("l2", qa2, selected=["wrong"], truncated=["wrong", "gold"], capped=True),
        record("d2", qa2, arm="direct_text", selected=["wrong"], truncated=["wrong"]),
        record("q1", qa1, precision="int8", em=1, prediction="answer"),
        record("repeat", qa1, repeat=1, em=0),
    ]
    qa_job, dataset = make_job(root, "public", [qa1, qa2], qa_rows)
    for ratio, em in ((0, 0), (0.1, 1), (0.2, 0), (1, 1)):
        qa_rows.append(record(f"bridge-{ratio}", qa1, experiment="E4", ratio=ratio, em=em))
    qa_rows.append(record("bridge-incomplete", qa2, experiment="E4", ratio=0, em=1))
    (qa_job / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in qa_rows))
    (qa_job / "summary.json").write_text(json.dumps({"stop_reason": "complete", "planned_runs": len(qa_rows)}))
    syn2 = {"id": "chain-2", "dataset": "synthetic_multihop", "metadata": {"chain_length": 2}}
    syn5 = {"id": "chain-5", "dataset": "synthetic_multihop", "metadata": {"chain_length": 5}}
    make_job(root, "chains", [syn2, syn5], [
        record("s5", syn2, experiment="E2", hops=5, actual=2, em=1),
        record("s20", syn2, experiment="E2", hops=20, actual=2, em=1),
        record("s55", syn5, experiment="E2", hops=5, actual=5, em=0),
    ])
    (root / "guard-status.json").write_text(json.dumps({"status": "complete", "returncode": 0}))
    return root, qa_job, dataset


def test_support_truncation_and_disagreements_use_matched_repeat_zero_records(artifacts):
    root, _, _ = artifacts
    report = analyze(root, require_complete=True)
    assert report["completeness"] == "complete"
    groups = report["public_e1"]
    latent = next(g for g in groups if g["condition"]["arm"] == "latent" and g["condition"]["precision"] == "bf16")
    assert latent["n_examples"] == 2  # repeat one is excluded
    assert latent["exact_match"] == 0.5
    coverage = {s["support_coverage"]: s for s in latent["by_support_coverage"]}
    assert coverage["all_support_documents"]["exact_match"] == 1
    assert coverage["all_support_documents"]["selected_document_truncation_rate"] == 0
    assert coverage["missing_support_documents"]["unknown_rate"] == 1
    assert coverage["missing_support_documents"]["output_cap_rate"] == 1
    assert latent["truncated_examples"] == [{"example_id": "qa-2", "selected_truncated_ids": ["wrong"],
                                              "selected_support_truncated_ids": []}]
    paired = report["public_e1_bf16_latent_vs_direct"][0]
    assert paired["n_matched"] == 2
    assert [x["example_id"] for x in paired["disagreements"] if x["exact_match_disagreement"]] == ["qa-1"]
    assert len([g for g in groups if g["condition"]["precision"] == "int8"]) == 1
    assert "do not establish causal" in " ".join(report["limitations"])


def test_chain_bins_keep_nominal_hops_separate_and_e4_uses_common_questions(artifacts):
    report = analyze(artifacts[0])
    bins = {(g["chain_length"], g["nominal_hops"], g["executed_evidence_hops"]): g for g in report["synthetic_e2"]}
    assert set(bins) == {(2, 5, 2), (2, 20, 2), (5, 5, 5)}
    assert all(g["n_examples"] == 1 for g in bins.values())
    e4 = report["e4_recompute"][0]
    assert e4["n_matched_all_four"] == 1
    assert e4["incomplete_question_ids"] == ["qa-2"]
    zero, tenth = e4["ratios"][:2]
    assert zero["available"]["exact_match"] == 0.5
    assert zero["matched_all_four"]["exact_match"] == 0
    assert tenth["paired_em_difference_vs_zero"] == 1
    assert "Snapshot: COMPLETE" in markdown(report)


def test_partial_snapshot_never_mutates_raw_artifacts(artifacts):
    root, job, _ = artifacts
    path = job / "results.jsonl"
    content = path.read_bytes() + b'{"id":"unfinished'
    path.write_bytes(content)
    report = analyze(root)
    assert report["completeness"] == "incomplete"
    assert next(x for x in report["jobs"] if x["job"] == "public")["results_snapshot"]["ignored_partial_trailing_records"] == 1
    assert path.read_bytes() == content
    with pytest.raises(ValueError, match="incomplete"):
        analyze(root, require_complete=True)


def test_analysis_rejects_datasets_other_than_the_recorded_bytes(artifacts):
    root, _, dataset = artifacts
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        analyze(root)


def amend_completed_fixture(root):
    """A controlled interruption after all explicitly retained jobs finished."""
    guard = {"status": "stopped", "reason": "interrupted", "returncode": -2,
             "elapsed_seconds": 12.5, "command": ["python", "run_matrix.py"]}
    names = sorted(path.parent.name for path in root.glob("*/manifest.json"))
    total = sum(json.loads(path.read_text())["planned_runs"] for path in root.glob("*/summary.json"))
    amendment = {"original_planned_conditions": total + 6, "completed_initial_conditions": total,
                 "canceled_original_remainder": 6, "completed_jobs": names,
                 "guard_exit": dict(guard), "reason": "Declared budget amendment"}
    (root / "guard-status.json").write_text(json.dumps(guard))
    (root / "amendment.json").write_text(json.dumps(amendment))
    return amendment, guard


def test_strict_amendment_reports_only_completed_scope_and_preserves_interrupted_guard(artifacts):
    root, _, _ = artifacts
    amendment, guard = amend_completed_fixture(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = analyze(root, require_complete=True)
    assert report["completeness"] == "complete"
    assert report["completion_scope"]["kind"] == "amended_initial_jobs"
    assert report["completion_scope"]["reported_jobs"] == ["chains", "public"]
    assert report["completion_scope"]["reported_successful_conditions"] == 14
    assert report["completion_scope"]["original_planned_conditions"] == 20
    assert report["completion_scope"]["canceled_original_remainder"] == 6
    assert report["resource_guard"] == guard
    assert report["resource_guard"]["status"] == "stopped"
    assert report["amendment_receipt"]["content"] == amendment
    assert report["amendment_receipt"]["sha256"] == hashlib.sha256(before[root / "amendment.json"]).hexdigest()
    assert "Canceled conditions are not completed results" in markdown(report)
    assert report["completion_errors"] == []
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("damage", [
    "missing_amendment", "missing_guard", "missing_job", "extra_job", "duplicate_job_name",
    "missing_manifest", "extra_unmanifested_results", "wrong_completed_count", "unbalanced_counts",
    "boolean_count", "string_count", "zero_canceled", "altered_preserved_guard", "matching_failed_guard",
    "matching_resource_stop", "matching_wrong_exit_code", "missing_summary", "partial_results",
    "new_failed_condition", "later_failure_overrides_success", "wrong_summary_count",
])
def test_amendment_cannot_mask_missing_mismatched_or_failed_runs(artifacts, damage):
    root, job, _ = artifacts
    amendment, guard = amend_completed_fixture(root)
    if damage == "missing_job":
        amendment["completed_jobs"].append("absent")
    elif damage == "extra_job":
        amendment["completed_jobs"].remove("chains")
    elif damage == "duplicate_job_name":
        amendment["completed_jobs"].append("chains")
    elif damage == "wrong_completed_count":
        amendment["completed_initial_conditions"] += 1
        amendment["original_planned_conditions"] += 1  # arithmetic alone still balances
    elif damage == "unbalanced_counts":
        amendment["original_planned_conditions"] += 1
    elif damage == "boolean_count":
        amendment["canceled_original_remainder"] = True
    elif damage == "string_count":
        amendment["completed_initial_conditions"] = "14"
    elif damage == "zero_canceled":
        amendment["canceled_original_remainder"] = 0
        amendment["original_planned_conditions"] = 14
    elif damage == "altered_preserved_guard":
        amendment["guard_exit"]["elapsed_seconds"] += 1
    elif damage.startswith("matching_"):
        if damage == "matching_failed_guard":
            guard.update(status="failed", reason="child_failed", returncode=1)
        elif damage == "matching_resource_stop":
            guard.update(reason="rss_limit")
        else:
            guard.update(returncode=1)
        amendment["guard_exit"] = dict(guard)
    (root / "guard-status.json").write_text(json.dumps(guard))
    (root / "amendment.json").write_text(json.dumps(amendment))
    if damage == "missing_amendment":
        (root / "amendment.json").unlink()
    elif damage == "missing_guard":
        (root / "guard-status.json").unlink()
    elif damage == "missing_manifest":
        (job / "manifest.json").unlink()
    elif damage == "extra_unmanifested_results":
        extra = root / "unused_partial"
        extra.mkdir()
        (extra / "results.jsonl").write_text('{}\n')
    elif damage == "missing_summary":
        (job / "summary.json").unlink()
    elif damage == "partial_results":
        with (job / "results.jsonl").open("ab") as handle:
            handle.write(b'{"id":"partial')
    elif damage in {"new_failed_condition", "later_failure_overrides_success"}:
        failed = {"id": "l1" if damage == "later_failure_overrides_success" else "failed", "status": "error"}
        with (job / "results.jsonl").open("a") as handle:
            handle.write(json.dumps(failed) + "\n")
    elif damage == "wrong_summary_count":
        path = job / "summary.json"
        summary = json.loads(path.read_text())
        summary["n_successful_runs"] = 999
        path.write_text(json.dumps(summary))
    report = analyze(root)
    assert report["completeness"] == "incomplete"
    assert report["completion_errors"]
    with pytest.raises(ValueError, match="Run is incomplete"):
        analyze(root, require_complete=True)


def test_successful_guard_does_not_hide_latest_failed_attempt(artifacts):
    root, job, _ = artifacts
    with (job / "results.jsonl").open("a") as handle:
        handle.write(json.dumps({"id": "l1", "status": "error"}) + "\n")
    report = analyze(root)
    public = next(x for x in report["jobs"] if x["job"] == "public")
    assert public["unresolved_error_records"] == 1
    assert public["successful_unique_runs"] == 10
    with pytest.raises(ValueError, match="Run is incomplete"):
        analyze(root, require_complete=True)
