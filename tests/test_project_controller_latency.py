"""CPU receipts/accounting fixtures, never provider performance measurements."""
from copy import deepcopy
import json

import pytest

from scripts.project_controller_latency import (
    LEGACY, NATIVE, canonical, digest, extract_condition, extract_run, load_profiles,
    project_condition, project_workloads, validate_profile, verify_sources,
)


def native_row():
    def generation(phase, output):
        return {"phase": phase, "round": 1, "messages": [{"role": "user", "content": phase}],
            "prompt_token_ids": [1, 2, 3], "output_token_ids": output,
            "raw_output": "ANSWER" if phase == "decision" else "answer",
            "stats": {"prefill_tokens": 3, "generated_tokens": len(output),
                "sampled_token_ids": output + [99], "sampled_tokens_including_eos": len(output) + 1,
                "stop_reason": "eos", "prefill_ms": 10, "generation_ms": 5,
                "model_inference_ms": 15, "elapsed_ms": 16, "max_output_tokens": 48,
                "sampling": {"temperature": 0}}}
    return {"status": "ok", "protocol": NATIVE, "id": "case", "example_id": "q", "arm": "iterative_text",
        "category": "2hop", "question": "Question?", "answers": ["answer"], "prediction": "answer",
        "raw_output": "answer", "output_token_ids": [4], "scores": {"exact_match": 1.0, "f1": 1.0},
        "search_queries": ["Question?"], "evidence_ids": ["a"], "stop_reason": "controller_answer",
        "generations": [generation("decision", [5, 6]), generation("final", [4])],
        "counts": {"controller_calls": 1, "controller_prefill_tokens": 3, "total_prefill_tokens": 6,
            "controller_generated_tokens": 2, "final_generated_tokens": 1, "total_generated_tokens": 3,
            "controller_sampled_tokens_including_eos": 3, "final_sampled_tokens_including_eos": 2},
        "cold_end_to_end_ms": 55, "component_ms": {"controller_ms": 30, "final_ms": 20, "other_ms": 4, "retrieval_ms": 1}}


def legacy_row():
    stats = {"prefix_tokens": 0, "prefill_tokens": 100, "reasoning_output_token_ids": [5, 6, 7],
        "reasoning_token_ids": [5, 6], "reasoning_sampled_token_ids": [5, 6, 7],
        "action_token_ids": [8], "action_sampled_token_ids": [8, 99], "forced_control_token_ids": [10],
        "reasoning_stop_reason": "end_think", "action_stop_reason": "eos", "stop_reason": "eos",
        "reasoning_generated_tokens": 3, "action_generated_tokens": 1, "forced_control_tokens": 1,
        "generated_tokens": 4, "sampled_tokens": 5, "max_reasoning_tokens": 256, "max_action_tokens": 32,
        "sampling": {"temperature": .6}, "phase_ms": {"initial_prefill": 8, "reasoning": 10,
                                                             "forced_control_prefill": 2, "action": 8}}
    return {"status": "ok", "protocol_version": LEGACY, "id": "legacy-case", "example_id": "q",
        "case": {"arm": "iterative_text", "max_rounds": 5}, "category": "2hop", "question": "Question?",
        "answers": ["answer"], "prediction": "answer", "raw_output": "answer", "output_token_ids": [4],
        "scores": {"exact_match": 1.0, "f1": 1.0}, "search_queries": ["Question?"], "evidence_ids": ["a"],
        "stop_reason": "controller_answer", "generation_stop": "eos", "executed_retrieval_rounds": 1,
        "controller_config": {"max_answer_tokens": 48},
        "trace": [{"event": "decision", "round": 1, "controller_decode": stats, "action_token_ids": [8]}],
        "counts": {"controller_calls": 1, "controller_prefill_tokens": 100, "final_prefill_tokens": 80,
            "total_prefill_tokens": 181, "controller_reasoning_tokens": 3, "controller_action_tokens": 1,
            "controller_forced_tokens": 1, "host_output_tokens": 1, "internal_decoded_tokens": 4,
            "controller_sampled_tokens_including_eos": 5, "cache_prefill_tokens": 0,
            "bridge_recomputed_tokens": 0, "reranker_input_tokens": 0},
        "cold_end_to_end_ms": 55, "component_ms": {"controller_ms": 30, "final_generation_ms": 20,
            "retrieval_ms": 2, "tokenization_ms": 1, "other_query_ms": 2, "cache_construction_ms": 0, "reranking_ms": 0}}


def profile(**changes):
    return {"id": "fixture", "model_id": "Qwen/Qwen3-14B", "provider": "fixture-only",
        "date": "2026-09-11", "source_url": "https://example.invalid/fixture",
        "snapshot_sha256": "a" * 64, "ttft_s": 2, "reported_tps": 10,
        "metric_semantics": "ambiguous_reported_throughput", "applies_to": ["native/14b"], **changes}


def workload(kind="native"):
    row = extract_condition(native_row() if kind == "native" else legacy_row(), kind)
    row["run_label"] = "native/14b" if kind == "native" else "prior/8b_thinking"
    row["model_path"] = "/fixture/Qwen3-14B" if kind == "native" else "/fixture/Qwen3-8B"
    return row


def test_native_counts_eos_and_outer_host_work_are_distinct():
    result = workload()
    assert result["totals"]["logical_model_calls"] == 2
    assert result["totals"]["generated_tokens"] == 3
    assert result["totals"]["sampled_tokens_including_eos"] == 5
    assert result["totals"]["initial_prefill_tokens"] == 6
    assert result["local_observed_ms"]["outer_host_work"] == 5
    assert result["local_observed_ms"]["native_within_call_wrapper_work"] == 20
    assert result["calls"][0]["prompt_token_ids_sha256"] == digest(canonical([1, 2, 3]))


def test_reasoning_generated_delimiter_and_forced_inputs_are_not_doubled():
    result = workload("legacy")
    assert result["totals"]["logical_model_calls"] == 2  # closure is not a third request
    assert result["totals"]["reasoning_generated_tokens"] == 3
    assert result["totals"]["reasoning_content_tokens"] == 2
    assert result["totals"]["generated_tokens"] == 5
    assert result["totals"]["initial_prefill_tokens"] == 180
    assert result["totals"]["forced_control_prefill_tokens"] == 1
    assert result["totals"]["sampled_tokens_including_eos"] == 7
    assert result["calls"][-1]["sampled_token_ids_sha256"] is None
    assert result["calls"][-1]["sample_count_basis"].startswith("derived:")
    assert result["local_observed_ms"]["native_within_call_wrapper_work"] is None


def test_early_reasoning_eos_aborts_action_without_inventing_forced_work():
    row = legacy_row()
    stats = row["trace"][0]["controller_decode"]
    stats.update(reasoning_output_token_ids=[5], reasoning_token_ids=[5], reasoning_sampled_token_ids=[5, 99],
        action_token_ids=[], action_sampled_token_ids=[], forced_control_token_ids=[], reasoning_stop_reason="eos",
        action_stop_reason="not_started", stop_reason="reasoning_eos", reasoning_generated_tokens=1,
        action_generated_tokens=0, forced_control_tokens=0, generated_tokens=1, sampled_tokens=2)
    row["trace"][0]["action_token_ids"] = []
    row["counts"].update(total_prefill_tokens=180, controller_reasoning_tokens=1, controller_action_tokens=0,
        controller_forced_tokens=0, internal_decoded_tokens=1, controller_sampled_tokens_including_eos=2)
    result = extract_condition(row, "legacy")
    assert result["totals"]["generated_tokens"] == 2 and result["totals"]["sampled_tokens_including_eos"] == 4


@pytest.mark.parametrize("mutation", ["bad_sample", "bad_total", "bad_timing", "bad_final", "negative_token"])
def test_inconsistent_saved_workloads_fail_closed(mutation):
    row = native_row()
    if mutation == "bad_sample": row["generations"][0]["stats"]["sampled_token_ids"][0] = 777
    if mutation == "bad_total": row["counts"]["total_prefill_tokens"] += 1
    if mutation == "bad_timing": row["cold_end_to_end_ms"] += 1
    if mutation == "bad_final": row["output_token_ids"] = [9]
    if mutation == "negative_token": row["generations"][0]["prompt_token_ids"][0] = -1
    with pytest.raises(ValueError): extract_condition(row, "native")


def test_cache_relay_is_explicitly_unsupported():
    row = legacy_row()
    row["case"]["arm"] = "relay_kv_bf16"
    with pytest.raises(ValueError, match="unsupported"):
        extract_condition(row, "legacy")


def test_projection_uses_generated_tokens_only_and_never_adds_prefill_rate():
    source = workload("legacy")
    before = deepcopy(source)
    p = profile(applies_to=["prior/8b_thinking"], model_id="Qwen/Qwen3-8B")
    result = project_condition(source, p)
    assert source == before
    assert result["decode_rate_assumption_s"] == pytest.approx(.005 + 2 + 3/10 + 2)
    assert result["inclusive_rate_proxy_s"] == pytest.approx(.005 + 2 + 2)
    assert [c["generated_tokens_used"] for c in result["calls"]] == [4, 1]
    source["calls"][0]["initial_prefill_tokens"] = 100000
    source["calls"][0]["forced_control_prefill_tokens"] = 100000
    same = project_condition(source, p)
    assert same["decode_rate_assumption_s"] == result["decode_rate_assumption_s"]


def test_known_tpot_does_not_emit_inclusive_proxy_or_false_interval():
    w = workload()
    known = profile(metric_semantics="decode_tps_from_tpot")
    result = project_condition(w, known)
    assert result["inclusive_rate_proxy_s"] is None
    output = project_workloads({"runs": [{"label": "native/14b", "workloads": [w]}]}, [known])
    assert output["summaries"][0]["inclusive_rate_proxy_s"] is None
    assert output["summaries"][0]["decode_rate_assumption_s"]["n"] == 1


def test_projection_pairs_same_questions_without_transferring_accuracy():
    base, enhanced = workload(), workload()
    base["arm"], base["case_id"] = "basic_rag", "basic"
    enhanced["observed_scores_unchanged"] = {"exact_match": 0.0, "f1": 0.0}
    result = project_workloads({"runs": [{"label": "native/14b", "workloads": [base, enhanced]}]}, [profile()])
    assert result["paired_summaries"][0]["n_paired"] == 1
    assert sorted(r["observed_correct_unchanged"] for r in result["summaries"]) == [0.0, 1.0]
    assert result["paired_summaries"][0]["decode_rate_assumption_s"]["mean_paired_difference_s"] == 0


@pytest.mark.parametrize("field,value", [("ttft_s", 0), ("ttft_s", float("nan")), ("reported_tps", float("inf")),
    ("reported_tps", -1), ("reported_tps", True), ("reported_tps", None), ("reported_tps", 0),
    ("date", "yesterday"), ("source_url", "file:///secret"),
    ("snapshot_sha256", "unknown"), ("metric_semantics", "guess"), ("applies_to", [])])
def test_profile_requires_explicit_valid_source_and_positive_finite_rates(field, value):
    with pytest.raises(ValueError): validate_profile(profile(**{field: value}))


def test_profile_cannot_cross_model_families_or_hide_wrong_checkpoint():
    with pytest.raises(ValueError, match="model_id"):
        validate_profile(profile(applies_to=["native/122b"]))
    with pytest.raises(ValueError, match="does not apply"):
        project_condition(workload("legacy"), profile())
    wrong = workload()
    wrong["model_path"] = "/fixture/Qwen3.5-122B-A10B-5bit"
    with pytest.raises(ValueError, match="checkpoint"):
        project_condition(wrong, profile())
    correct = workload()
    correct.update(run_label="native/122b", model_path="/fixture/Qwen3.5-122B-A10B-5bit")
    assert project_condition(correct, profile(model_id="Qwen/Qwen3.5-122B-A10B", applies_to=["native/122b"]))


def test_projection_applies_nonlinear_terms_per_call_including_zero_and_one_tokens():
    w = workload()
    w["calls"] = [{"call_index": index, "phase": "final" if index == 2 else "controller",
        "reasoning_generated_tokens": 0, "action_generated_tokens": 0,
        "final_generated_tokens": generated} for index, generated in enumerate([0, 1, 41])]
    result = project_condition(w, profile())
    assert len(result["calls"]) == 3
    assert result["decode_rate_assumption_s"] == pytest.approx(.005 + 2 + 2 + 6)
    assert result["inclusive_rate_proxy_s"] == pytest.approx(.005 + 2 + 2 + 4.1)
    assert result["inclusive_rate_proxy_s"] != pytest.approx(.005 + max(3 * 2, 42 / 10))


def test_optional_snapshot_is_hashed_confined_and_not_embedded(tmp_path):
    source = tmp_path / "source.html"
    source.write_text("Internal fixture source; do not redistribute")
    p = profile(snapshot_path="source.html", snapshot_sha256=digest(source.read_bytes()))
    target = tmp_path / "profiles.json"
    target.write_text(json.dumps({"profiles": [p]}))
    loaded, _ = load_profiles(target)
    assert loaded[0]["snapshot_verified"] is True
    assert source.read_text() not in json.dumps(loaded)
    source.write_text("changed")
    with pytest.raises(ValueError, match="snapshot hash"): load_profiles(target)
    for unsafe in ("../source.html", "/tmp/outside-source.html"):
        target.write_text(json.dumps({"profiles": [profile(snapshot_path=unsafe)]}))
        with pytest.raises(ValueError): load_profiles(target)
    link = tmp_path / "link.html"
    link.symlink_to(source)
    target.write_text(json.dumps({"profiles": [profile(snapshot_path="link.html")]}))
    with pytest.raises(ValueError, match="symlink"): load_profiles(target)


def test_repository_relative_snapshot_is_still_confined(tmp_path, monkeypatch):
    import scripts.project_controller_latency as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "runs/provider-latency"
    directory.mkdir(parents=True)
    source = directory / "source.json"
    source.write_text('{"fixture": true}')
    target = directory / "profiles.json"
    target.write_text(json.dumps({"profiles": [profile(snapshot_path="runs/provider-latency/source.json",
        snapshot_sha256=digest(source.read_bytes()))]}))
    loaded, _ = load_profiles(target)
    assert loaded[0]["snapshot_verified"] is True


def test_manifest_data_source_and_scores_are_verified_before_extraction(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "source.py"
    source.write_text("# frozen fixture\n")
    hashes = {"questions": "qhash", "corpus": "chash"}
    row = native_row()
    manifest = {"identity": {"source_sha256": {"source.py": digest(source.read_bytes())},
        "questions_sha256": "qhash", "corpus_sha256": "chash", "arms": ["iterative_text"], "config": {},
        "backend": {"model_revision": "fixture", "model_path": "fixture"}}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "summary.json").write_text(json.dumps({"status": "complete", "successful_conditions": 1, "planned_conditions": 1}))
    (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n")
    questions = {"q": {"question": "Question?", "answers": ["answer"]}}
    result = extract_run(run_dir, "fixture", "native", questions, hashes, tmp_path)
    assert len(result["workloads"]) == 1
    with pytest.raises(ValueError, match="questions hash"):
        extract_run(run_dir, "fixture", "native", questions, {**hashes, "questions": "changed"}, tmp_path)
    row["scores"]["exact_match"] = 0.0
    (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unchanged scores"):
        extract_run(run_dir, "fixture", "native", questions, hashes, tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source unavailable"):
        verify_sources(run_dir, manifest, tmp_path)
