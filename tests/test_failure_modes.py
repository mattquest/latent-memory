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
