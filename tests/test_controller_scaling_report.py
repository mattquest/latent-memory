"""Synthetic artifact-integrity fixtures, never model evaluation results."""
import gzip
import json
from pathlib import Path

import pytest

from scripts import build_controller_scaling_report as report
from scripts.build_adaptive_report import expected_case_id


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path, arms=None, sampling=False):
    root = tmp_path / "repo"
    root.mkdir()
    dataset, corpus, source = root / "questions.jsonl", root / "corpus.jsonl", root / "source.py"
    questions = [{"id": f"q{i}", "question": f"Where {i}?", "answers": ["FOUND"],
                  "category": "2hop", "split": "development", "supporting_context_ids": ["d1", "d2"]} for i in range(2)]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in questions))
    corpus.write_text(''.join(json.dumps({"id": f"d{i}", "title": "T", "text": "Text"}) + "\n" for i in range(1, 4)))
    source.write_text("# frozen fixture implementation\n")
    config = {"dataset": "questions.jsonl", "corpus": "corpus.jsonl", "limit": 2, "seed": 21,
              "arms": arms or ["basic_rag", "iterative_text", "incremental_kv_bf16", "relay_kv_bf16"],
              "rounds": [2], "initial_top_k": 1, "documents_per_round": 1, "max_documents": 3,
              "max_action_tokens": 32, "max_answer_tokens": 48, "max_model_context_tokens": 8192,
              "controller_temperature": .6 if sampling else 0.0, "controller_top_p": .95 if sampling else 1.0,
              "controller_top_k": 20 if sampling else 0,
              "max_evidence_tokens": 6144, "max_document_tokens": 300, "bridge_ratio": .2}
    jobs = []
    for model in ("8B", "14B"):
        for reasoning in (0, 256):
            job = {"id": f"{model}-{reasoning}", "model_label": model, "max_reasoning_tokens": reasoning,
                   "model_path": f"models/{model}", "model_revision": f"revision-{model}", "run_dir": f"runs/{model}-{reasoning}"}
            jobs.append(job)
            identity = {"config": {**config, "max_reasoning_tokens": reasoning, "max_case_seconds": 180},
                        "protocol": report.expected_protocol(reasoning, config), "questions_sha256": report.sha(dataset.read_bytes()),
                        "corpus_sha256": report.sha(corpus.read_bytes()), "source_sha256": {"source.py": report.sha(source.read_bytes())},
                        "backend": {"model_path": str(root / job["model_path"]), "model_revision": job["model_revision"]}}
            run_dir = root / job["run_dir"]
            write_json(run_dir / "manifest.json", {"identity": identity, "setup_timings_ms": {"generator_load_and_model_validation_ms": 1}})
            rows = []
            for question in questions:
                for arm in config["arms"]:
                    rounds = 1 if arm == "basic_rag" else 2
                    case = {"arm": arm, "max_rounds": rounds}
                    trace = [{"event": "retrieval", "round": 1, "new_document_ids": ["d1"], "accumulated_document_ids": ["d1"],
                              "hit_ids_and_scores": [["d1", 2]], "evidence_tokens": 5}]
                    n_reasoning = 2 if reasoning and rounds == 2 else 0
                    n_action = 1 if rounds == 2 else 0
                    forced = 1 if reasoning and rounds == 2 else 0
                    if rounds == 2:
                        trace.extend([{"event": "decision", "round": 1, "action_token_ids": [42],
                            "controller_seed": int(report.sha(json.dumps([21, question["id"], "decision", 1], sort_keys=True).encode())[:8], 16),
                            "action": {"valid": True, "kind": "search", "query": "next"},
                            "controller_decode": {"reasoning_output_token_ids": [40, 41] if reasoning else [],
                                "controller_seed": int(report.sha(json.dumps([21, question["id"], "decision", 1], sort_keys=True).encode())[:8], 16),
                                "context_reservation_tokens": 30,
                                "sampling": {"temperature": config["controller_temperature"], "top_p": config["controller_top_p"],
                                    "top_k": config["controller_top_k"], "seed": int(report.sha(json.dumps([21, question["id"], "decision", 1], sort_keys=True).encode())[:8], 16)},
                                "reasoning_generated_tokens": n_reasoning, "action_token_ids": [42], "action_generated_tokens": 1,
                                "reasoning_sampled_token_ids": [40, 41] if reasoning else [],
                                "action_sampled_token_ids": [42, 99], "sampled_tokens": n_reasoning + 2,
                                "forced_control_token_ids": [50] if reasoning else [], "forced_control_tokens": forced,
                                "reasoning_stop_reason": "end_think" if reasoning else "disabled", "action_stop_reason": "eos"}},
                            {"event": "retrieval", "round": 2, "new_document_ids": ["d2"], "accumulated_document_ids": ["d1", "d2"],
                             "hit_ids_and_scores": [["d2", 1]], "evidence_tokens": 10}])
                    rows.append({"id": expected_case_id(identity["protocol"], question["id"], case, reasoning,
                        config["controller_temperature"], config["controller_top_p"], config["controller_top_k"]),
                        "status": "ok", "protocol_version": identity["protocol"], "example_id": question["id"], "case": case,
                        "question": question["question"], "answers": question["answers"], "prediction": "FOUND",
                        "scores": {"exact_match": 1, "f1": 1}, "cold_end_to_end_ms": 10 + reasoning,
                        "component_ms": {"controller_ms": reasoning, "final_generation_ms": 10},
                        "counts": {"host_output_tokens": 1, "internal_decoded_tokens": n_reasoning + n_action,
                            "controller_reasoning_tokens": n_reasoning, "controller_action_tokens": n_action,
                            "controller_forced_tokens": forced, "controller_calls": rounds - 1, "retrieval_calls": rounds,
                            "controller_sampled_tokens_including_eos": n_reasoning + n_action + (rounds - 1),
                            "controller_prefill_tokens": 5 * (rounds - 1), "final_prefill_tokens": 10,
                            "cache_prefill_tokens": 0, "bridge_recomputed_tokens": 0, "reranker_input_tokens": 0,
                            "total_prefill_tokens": 5 * (rounds - 1) + 10 + forced,
                            "primary_model_prefill_tokens": 5 * (rounds - 1) + 10 + forced,
                            "all_models_prefill_tokens": 5 * (rounds - 1) + 10 + forced,
                            "retrieved_documents": rounds, "retrieved_evidence_tokens": rounds * 5,
                            "max_model_context_tokens": 30}, "executed_retrieval_rounds": rounds,
                        "invalid_actions": 0, "generation_stop": "eos", "stop_reason": "round_budget",
                        "support_annotation_coverage": rounds / 2, "controller_config": {"max_reasoning_tokens": reasoning,
                            "max_action_tokens": 32, "max_answer_tokens": 48, "controller_temperature": config["controller_temperature"],
                            "controller_top_p": config["controller_top_p"], "controller_top_k": config["controller_top_k"]},
                        "category": "2hop", "split": "development",
                        "evidence_ids": ["d1"] if rounds == 1 else ["d1", "d2"], "output_token_ids": [22],
                        "search_queries": [question["question"]] + (["next"] if rounds == 2 else []), "trace": trace})
            (run_dir / "results.jsonl").write_text(''.join(json.dumps(row) + "\n" for row in rows))
            write_json(run_dir / "summary.json", {"stop_reason": "complete", "planned_runs": len(rows)})
    provenance = root / "provenance.json"
    write_json(provenance, {"outputs": [{"path": path.name, "sha256": report.sha(path.read_bytes())} for path in (dataset, corpus)]})
    matrix = {"schema_version": report.MATRIX_VERSION, "cohort": "development", "dataset": dataset.name,
              "corpus": corpus.name, "data_manifest": provenance.name,
              "questions_sha256": report.sha(dataset.read_bytes()), "corpus_sha256": report.sha(corpus.read_bytes()),
              "shared_config": config, "jobs": jobs}
    path = root / "matrix.json"
    write_json(path, matrix)
    return root, path, matrix


def mutate_first_row(root, matrix, change):
    path = root / matrix["jobs"][0]["run_dir"] / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text(''.join(json.dumps(row) + "\n" for row in rows))


def test_complete_matrix_has_paired_productivity_and_exact_archives(tmp_path):
    root, path, matrix = fixture(tmp_path)
    output = tmp_path / "report"
    summary = report.build(path, output, root, no_plots=True)
    assert summary["report_status"] == "complete", summary["verification_issues"]
    assert summary["successful_unique_conditions"] == 32
    iterative = next(row for row in summary["metrics"] if row["job_id"] == "8B-256" and row["arm"] == "iterative_text" and row["category"] == "all")
    assert iterative["mean_new_support_after_initial"] == 1
    assert iterative["mean_counts"]["controller_reasoning_tokens"] == 2
    pair = next(row for row in summary["paired_comparisons"] if row["comparison_family"] == "reasoning_budget")
    assert pair["n_paired"] == 2
    assert pair["paired_example_ids"] == ["q0", "q1"]
    assert pair["mean_cold_latency_difference_ms"] == 256
    trajectory = next(row for row in summary["trajectories"] if row["job_id"] == "8B-256" and row["arm"] == "iterative_text" and row["round"] == 2)
    assert trajectory["new_support_count"] == 2 and trajectory["mean_cumulative_support_coverage"] == 1
    manifest = json.loads((output / "artifact-manifest.json").read_text())
    for receipt in manifest["artifacts"]:
        packed = (output / receipt["artifact"]).read_bytes()
        assert report.sha(packed) == receipt["artifact_sha256"]
        raw = gzip.decompress(packed)
        assert report.sha(raw) == receipt["source_sha256"]
        assert raw == Path(receipt["source"]).read_bytes()


def test_iterative_only_sampled_pilot_matches_generation_id_recipe(tmp_path):
    from eval.adaptive_experiments import AdaptiveCase, case_id, protocol_for
    root, path, matrix = fixture(tmp_path, arms=["iterative_text"], sampling=True)
    summary = report.build(path, tmp_path / "report", root, no_plots=True)
    assert summary["report_status"] == "complete", summary["verification_issues"]
    assert summary["successful_unique_conditions"] == 8
    assert len(summary["paired_comparisons"]) == 6
    for job in matrix["jobs"]:
        row = json.loads((root / job["run_dir"] / "results.jsonl").read_text().splitlines()[0])
        policy = dict(max_reasoning_tokens=job["max_reasoning_tokens"], controller_temperature=.6, controller_top_p=.95, controller_top_k=20)
        assert row["protocol_version"] == protocol_for(**policy)
        assert row["id"] == case_id(row["example_id"], AdaptiveCase(**row["case"]), **policy)


@pytest.mark.parametrize("mutation,kind", [
    (lambda rows: rows.append(rows[0]), "duplicate_result_rows"),
    (lambda rows: rows.pop(), "incomplete_or_unverified_job"),
    (lambda rows: rows[0].update(id="fake"), "unexpected_result_ids"),
    (lambda rows: rows[0]["scores"].update(exact_match=0), "result_score_mismatch"),
    (lambda rows: rows[0].update(support_annotation_coverage=1), "trace_or_budget_verification_failed"),
    (lambda rows: rows[1]["trace"][-1].update(new_document_ids=["d1"]), "trace_or_budget_verification_failed"),
    (lambda rows: rows[0]["counts"].update(controller_reasoning_tokens=1), "trace_or_budget_verification_failed"),
    (lambda rows: rows[0]["counts"].update(max_model_context_tokens=10000), "trace_or_budget_verification_failed"),
    (lambda rows: rows[0]["component_ms"].update(controller_ms=5), "trace_or_budget_verification_failed"),
    (lambda rows: rows[0]["controller_config"].update(max_action_tokens=64), "trace_or_budget_verification_failed"),
])
def test_bad_rows_withhold_all_headline_results(tmp_path, mutation, kind):
    root, path, matrix = fixture(tmp_path)
    mutate_first_row(root, matrix, mutation)
    summary = report.build(path, tmp_path / "report", root, no_plots=True)
    assert summary["report_status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert any(issue["kind"] == kind for issue in summary["verification_issues"])
    assert summary["metrics"] == [] and summary["paired_comparisons"] == [] and summary["figures"] == []


@pytest.mark.parametrize("field,value,kind", [
    ("max_answer_tokens", 49, "shared_generation_configuration_mismatch"),
    ("max_reasoning_tokens", 256, "reasoning_budget_mismatch"),
])
def test_job_configuration_changes_are_rejected(tmp_path, field, value, kind):
    root, path, matrix = fixture(tmp_path)
    manifest = root / matrix["jobs"][0]["run_dir"] / "manifest.json"
    data = json.loads(manifest.read_text())
    data["identity"]["config"][field] = value
    write_json(manifest, data)
    summary = report.build(path, tmp_path / "report", root, no_plots=True)
    assert any(issue["kind"] == kind for issue in summary["verification_issues"])


def test_changed_data_provenance_source_or_checkpoint_is_not_complete(tmp_path):
    root, path, matrix = fixture(tmp_path)
    (root / "source.py").write_text("# changed\n")
    summary = report.build(path, tmp_path / "report", root, no_plots=True)
    assert any(issue["kind"] == "source_verification_failed" for issue in summary["verification_issues"])
    (root / "provenance.json").write_text('{"outputs": []}')
    summary = report.build(path, tmp_path / "report", root, no_plots=True)
    assert any(issue["kind"] == "data_selection_provenance_unverified" for issue in summary["verification_issues"])


def test_matrix_requires_full_factorial_and_safe_output(tmp_path):
    root, path, matrix = fixture(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        report.build(path, root, root, no_plots=True)
    output = tmp_path / "report"
    output.mkdir()
    (output / "artifacts").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        report.build(path, output, root, no_plots=True)
    matrix["jobs"][0]["max_reasoning_tokens"] = 256
    write_json(path, matrix)
    with pytest.raises(ValueError, match="budgets"):
        report.read_matrix(path, root)


def test_early_stop_carries_coverage_without_claiming_new_evidence(tmp_path):
    root, path, matrix = fixture(tmp_path)
    parsed = report.read_matrix(path, root)
    inspections, productivity, issues = report.inspect_matrix(parsed, root)
    assert not issues
    # Analysis-only fixture: a valid trajectory ending after step one still has
    # an originally assigned two-round cap; later support exposure remains flat.
    job = matrix["jobs"][0]["id"]
    row = next(row for row in inspections[job]["records"] if row["case"]["arm"] == "iterative_text")
    detail = productivity[job][row["id"]]
    detail["steps"] = detail["steps"][:1]
    metrics, trajectories, _, _ = report.compute_results(parsed, inspections, productivity)
    later = next(item for item in trajectories if item["job_id"] == job and item["arm"] == "iterative_text" and item["round"] == 2)
    assert later["n_questions"] == 2 and later["n_reached_round"] == 1
    assert later["mean_new_support_per_original_question"] == .5
    assert later["mean_new_support_per_reached_question"] == 1
    assert later["mean_cumulative_support_coverage"] == .75
