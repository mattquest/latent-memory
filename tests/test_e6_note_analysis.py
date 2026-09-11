import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("e6_notes", Path(__file__).resolve().parents[1] /
                                           "scripts/analyze_e6_notes.py")
notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notes)


def example():
    return {"id": "sample", "question": "Current code for record K123456?", "answers": ["654321"],
            "category": "knowledge_update", "metadata": {"current_answer": "654321", "old_answer": "111111"},
            "evidence": [{"id": "v2", "text": "Record K123456 now uses 654321."}]}


def result():
    ex = example()
    return {"id": "condition", "example_id": ex["id"], "question": ex["question"], "answers": ex["answers"],
            "category": ex["category"], "status": "ok", "case": {"experiment": "E6", "arm": "text",
                "precision": "bf16", "update_policy": "current", "repeat": 0},
            "evidence_ids": ["v2"], "prediction": "UNKNOWN", "scores": {"exact_match": 0},
            "output_token_ids": [3], "generation_stop": "eos", "trace": [
                {"hop": 1, "evidence_ids": ["v2"], "decoded_tokens": 16,
                 "relay_text": '{"facts":["record | code | 654321"]}'},
                {"hop": 2, "evidence_ids": [], "skipped": "no_new_evidence"}]}


def test_last_generated_note_skips_empty_hops_and_keeps_numeric_vs_entity_distinct():
    receipt = notes.note_receipt(result(), example(), 192, 16, 7, "line-hash")
    assert receipt["final_note"]["source_trace_index"] == 0
    assert not receipt["final_note"]["target_id_present"]
    assert receipt["final_note"]["current_answer_present"]
    assert not receipt["final_note"]["reached_configured_token_cap"]
    assert receipt["all_selected_evidence_contains_target_id"]
    assert receipt["final_unknown"]
    assert not notes.exact_token_present("K654321 6543210 X654321", "654321")
    assert notes.exact_token_present("code=654321; record k123456", "K123456")


def test_conflict_stratum_and_repeated_policies_have_correct_denominators():
    update = notes.note_receipt(result(), example(), 192, 16, 1, "h")
    all_policy = dict(update, result_id="other-condition", update_policy="all")
    ex = example()
    ex.update(id="conflict", category="contradiction_resolution", answers=["CONFLICT"],
              metadata={"conflicting_values": ["654321", "111111"]})
    row = result()
    row.update(id="conflict-condition", example_id=ex["id"], category=ex["category"], answers=ex["answers"],
               prediction="CONFLICT", scores={"exact_match": 1})
    row["trace"] = [{"hop": 1, "evidence_ids": ["v2"], "decoded_tokens": 192,
                     "relay_text": '{"facts":["K123456 | conflicting codes | 654321 and 111111"]}'}]
    conflict = notes.note_receipt(row, ex, 192, 16, 3, "c")
    groups = {(g["category"], g["update_policy"]): g for g in notes.summarize([update, all_policy, conflict])}
    combined = groups[("knowledge_update", "both_policies")]
    assert combined["conditions"] == 2 and combined["distinct_examples"] == 1
    assert combined["final_target_id_omitted"] == combined["final_current_answer_retained"] == combined["unknown"] == 2
    conflicts = groups[("contradiction_resolution", "current")]
    assert conflicts["current_answer_applicable"] == 0
    assert conflicts["final_both_conflict_values_retained"] == conflicts["final_target_id_retained"] == 1
    assert conflicts["final_note_at_cap"] == 1


def test_exact_hashes_and_completion_required_without_source_mutation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(json.dumps(example()) + "\n")
    manifest = {"identity": {"config": {"dataset": str(dataset), "max_relay_tokens": 192, "max_answer_tokens": 16},
                             "dataset_sha256": notes.sha(dataset.read_bytes()), "source_code_sha256": {"engine": "historical"}}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text(json.dumps({"stop_reason": "complete", "planned_runs": 1, "n_failed_runs": 0}))
    (run / "results.jsonl").write_text(json.dumps(result()) + "\n")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    analysis, receipts = notes.analyze(run)
    notes.write_report(analysis, receipts, tmp_path / "report")
    assert analysis["source"]["results_sha256"] == notes.sha(before["results.jsonl"])
    assert analysis["scope"] == "posthoc_descriptive_not_causal"
    assert analysis["source"]["executed_identity"]["source_code_sha256"] == {"engine": "historical"}
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    (run / "summary.json").write_text(json.dumps({"stop_reason": "interrupted", "planned_runs": 1}))
    with pytest.raises(ValueError, match="complete successful"):
        notes.analyze(run)
    dataset.write_text("{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        notes.analyze(run)
