"""CPU-only protocol/metric checks; no fake model accuracy claims."""
import dataclasses
import json
from pathlib import Path

import pytest

from eval.real_experiments import (
    Case, ExperimentConfig, answer_scores, cases_for, clean_prediction,
    load_records, normalize_answer, schedule_evidence, score_example, summarize, wilson_interval,
)


def test_answer_metrics():
    assert answer_scores("The Blue-Aster.", ["blueaster"])["exact_match"] == 1
    assert answer_scores("red blue", ["red green"])["f1"] == 0.5
    assert answer_scores("no", ["yes"])["f1"] == 0
    assert answer_scores("", [""])["f1"] == 1
    assert answer_scores("Wrong. The answer is Oslo.", ["Oslo"])["exact_match"] == 0
    assert clean_prediction("<think>x</think>Blue<|im_end|>") == "Blue"


def test_wilson_does_not_claim_certain_ceiling():
    low, high = wilson_interval(10, 10)
    assert 0.7 < low < 0.8
    assert high == pytest.approx(1)
    assert wilson_interval(0, 10)[1] > 0.2


def test_opaque_synthetic_codes_are_strict():
    example = {"dataset": "synthetic_private_fact", "answers": ["012345"]}
    assert score_example("012345", example)["exact_match"] == 1
    for invalid in ("-012345", "012,345", "012345.", "12345", "The code is 012345"):
        assert score_example(invalid, example)["exact_match"] == 0
    conflict = {"dataset": "synthetic_updates", "answers": ["CONFLICT"]}
    assert score_example("conflict", conflict)["exact_match"] == 1
    assert score_example("Conflict?", conflict)["exact_match"] == 0


def test_schedule_never_uses_gold_labels():
    example = {"question": "Who owns alpha?", "answers": ["secret"],
               "evidence": [{"id": "a", "text": "alpha owner one"},
                            {"id": "b", "text": "unrelated secret"}],
               "supporting_context_ids": ["b"]}
    cfg = ExperimentConfig("unused", "unused", evidence_per_hop=1)
    groups = schedule_evidence(example, 3, cfg)
    assert [[d["id"] for d in g] for g in groups] == [["a"], ["b"], []]
    example["answers"] = ["something else"]
    example["supporting_context_ids"] = ["a"]
    assert schedule_evidence(example, 3, cfg) == groups


def test_bm25_ignores_annotation_like_ids_and_retains_zero_scores():
    from eval.retrieval import BM25Index
    docs = [{"id": "support:first", "text": "alpha same"},
            {"id": "distractor:second", "text": "alpha else"},
            {"id": "support:third", "text": "unmatched words"}]
    before = [doc["text"] for doc, _ in BM25Index(docs).search("alpha", k=3)]
    docs[0]["id"], docs[1]["id"] = docs[1]["id"], docs[0]["id"]
    after = [doc["text"] for doc, _ in BM25Index(docs).search("alpha", k=3)]
    assert before == after == ["alpha same", "alpha else", "unmatched words"]
    assert [doc for doc, _ in BM25Index(docs).search("alpha", k=3)][:2] == docs[:2]
    assert BM25Index(docs).search("nohits", k=3) == [(doc, 0.0) for doc in docs]


def test_update_policy_uses_explicit_metadata():
    example = {"question": "code?", "evidence": [
        {"id": "old", "text": "code RED", "superseded_by": "new"},
        {"id": "new", "text": "code BLUE"}]}
    cfg = ExperimentConfig("unused", "unused")
    assert [x["id"] for x in schedule_evidence(example, 1, cfg, "current")[0]] == ["new"]
    assert len(schedule_evidence(example, 1, cfg, "all")[0]) == 2


def test_oracle_requires_explicit_schedule():
    cfg = ExperimentConfig("unused", "unused", schedule="oracle")
    with pytest.raises(ValueError, match="hop_context_ids"):
        schedule_evidence({"evidence": []}, 1, cfg)


def test_case_grid_and_repeat_identity():
    cfg = ExperimentConfig("unused", "unused", experiments=("E3", "E4", "E5"), repeats=2)
    cases = cases_for(cfg, {})
    assert len(cases) == 24
    assert {c.control for c in cases if c.experiment == "E3"} == {
        "correct", "mismatched", "zero", "random", "no_context"}
    assert {c.bridge_ratio for c in cases if c.experiment == "E4"} == {0, .1, .2, 1}


def test_multiple_latent_precisions_and_beam_policy():
    cfg = ExperimentConfig("unused", "unused", experiments=("E1", "E6"),
                           latent_precisions=("bf16", "int8"))
    cases = cases_for(cfg, {"category": "knowledge_update", "evidence": [{"text": "raw message"}]})
    e1 = [case for case in cases if case.experiment == "E1"]
    assert [(case.arm, case.precision) for case in e1] == [
        ("text", "bf16"), ("latent", "bf16"), ("latent", "int8"), ("direct_text", "bf16")]
    assert {case.update_policy for case in cases if case.experiment == "E6"} == {"all"}


def test_resume_discards_only_incomplete_last_record(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":')
    assert load_records(path) == [{"id": "one"}]
    assert path.read_bytes() == b'{"id":"one"}\n'
    path.write_bytes(b'bad\n{"id":"one"}\n')
    with pytest.raises(json.JSONDecodeError):
        load_records(path)


def test_repeats_do_not_inflate_accuracy_sample_count():
    from eval.real_experiments import Counts
    rows = []
    for repeat in (0, 1):
        rows.append({"status": "ok", "dataset": "test", "example_id": "a",
                     "case": dataclasses.asdict(Case("E1", "latent", repeat=repeat)),
                     "scores": {"exact_match": 1.0, "f1": 1.0}, "latency_ms": 10 + repeat,
                     "cold_ingest_ms": 2, "counts": dataclasses.asdict(Counts()),
                     "evidence_hops_executed": 1, "truncated_document_ids": []})
    group = summarize(rows)["groups"][0]
    assert group["n_examples"] == 1
    assert group["n_runs"] == 2
    assert group["exact_match_ci95"][0] < 0.3
