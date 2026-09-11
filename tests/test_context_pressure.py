"""Mechanics fixtures only; no model accuracy or latency evidence."""
import copy
from dataclasses import dataclass
import json

import pytest

from eval.context_pressure import (ARMS, PressureConfig, aggregate, prepare_replay, rank_retention,
                                   retain, run_pressure_case, run_pressure_suite, sha)
from eval.adaptive_experiments import PROLOGUE, document_text


@dataclass
class FakeBlock:
    token_ids: tuple
    quant: str = "bf16"
    metadata: dict = None

    @property
    def nbytes(self):
        return len(self.token_ids) * (1 if self.quant == "int8" else 2)


class FixtureBackend:
    name = "mock-fixture"
    max_context = 8192

    def __init__(self):
        self.calls = []
        self.last_decode_stats = {}

    def encode(self, text):
        return list(text.encode())

    def decode_tokens(self, tokens):
        return bytes(tokens).decode(errors="replace")

    def synchronize(self):
        pass

    def prefill(self, tokens):
        self.calls.append(("prefill", len(tokens)))
        return FakeBlock(tuple(tokens), metadata={})

    def quantize(self, block, bits):
        assert bits == 8
        return FakeBlock(block.token_ids, "int8", {})

    def concat(self, blocks, bridge_ratio):
        return FakeBlock(tuple(token for block in blocks for token in block.token_ids), metadata={"bridge_tokens_recomputed": 0})

    def decode(self, prefix, prompt, max_tokens):
        self.calls.append(("decode", (prefix.token_ids if prefix else ()) + tuple(prompt), max_tokens))
        output = b"alpha supports beta"[:max_tokens]
        self.last_decode_stats = {"stop_reason": "eos"}
        return list(output)

    def memory_stats(self):
        return {"fixture": True}


def fixture():
    backend = FixtureBackend()
    corpus = {f"doc-{index}": {"id": f"doc-{index}", "title": f"Title {index}",
                               "text": ("alpha relation " if index % 2 == 0 else "unrelated note ") * 12}
              for index in range(6)}
    source_config = {"max_document_tokens": 120, "bridge_ratio": .2}
    trace, accumulated, tokens = [], [], backend.encode(PROLOGUE)
    for number, new in enumerate((list(corpus)[:3], list(corpus)[3:]), 1):
        accumulated.extend(new)
        for key in new:
            tokens.extend(backend.encode(document_text(corpus[key]))[:120])
        trace.append({"event": "retrieval", "round": number, "new_document_ids": new,
                      "accumulated_document_ids": accumulated.copy(), "evidence_tokens": len(accumulated) * 120,
                      "evidence_token_ids_sha256": sha(json.dumps(tokens).encode())})
    source = {"id": "source-row", "trace": trace, "evidence_ids": accumulated,
              "component_ms": {"controller_ms": 10, "retrieval_ms": 2}}
    example = {"id": "question", "question": "alpha relation", "answers": ["alpha supports beta"],
               "supporting_context_ids": ["doc-0"], "category": "2hop"}
    config = PressureConfig("fixture", "fixture", "fixture", "fixture", evidence_budget=240, summary_budget=100,
                            expected_questions=1, max_answer_tokens=32)
    return backend, corpus, source_config, source, example, config


def test_reconstruction_verifies_source_tokens_and_rejects_mutation():
    backend, corpus, source_config, source, _, _ = fixture()
    fragments, rounds = prepare_replay(backend, corpus, source, source_config)
    assert len(fragments) == 6 and len(rounds) == 2
    assert all(row["truncated"] for row in fragments.values())
    source["trace"][0]["evidence_token_ids_sha256"] = "wrong"
    with pytest.raises(ValueError, match="token hash mismatch"):
        prepare_replay(backend, corpus, source, source_config)


def test_lexical_retention_uses_visible_text_only_and_deterministic_ties():
    fragments = {"b": {"tokens": [1]*20, "visible_text": "alpha relation"},
                 "a": {"tokens": [2]*20, "visible_text": "alpha relation"},
                 "c": {"tokens": [3]*20, "visible_text": "irrelevant"}}
    ranking, scores = rank_retention("alpha", fragments)
    assert ranking[:2] == ["a", "b"]
    assert retain("alpha", fragments, 20)[0] == ["a"]
    fragments["c"]["supporting"] = True
    fragments["c"]["answers"] = ["alpha"]
    assert rank_retention("alpha", fragments) == (ranking, scores)


def test_text_and_native_variants_retain_identical_documents_and_tokens():
    backend, corpus, source_config, source, example, config = fixture()
    rows = [run_pressure_case(backend, corpus, example, source, source_config, arm, config) for arm in ARMS[:3]]
    assert len({tuple(row["retained_document_ids"]) for row in rows}) == 1
    assert len({row["retained_evidence_token_ids_sha256"] for row in rows}) == 1
    assert all(row["retention_trace"] == rows[0]["retention_trace"] for row in rows)
    assert all(row["retained_evidence_tokens"] <= config.evidence_budget for row in rows)
    assert all(row["trace_overflow"] for row in rows)
    assert rows[2]["cache"]["stored_precision"] == "int8"
    assert all(row["replay_plus_source_controller_and_retrieval_ms"] == row["replay_cold_ms"] + 12 for row in rows)


def test_rolling_summary_never_reads_unbounded_history_and_reserves_its_slot():
    backend, corpus, source_config, source, example, config = fixture()
    row = run_pressure_case(backend, corpus, example, source, source_config, "rolling_summary", config)
    assert row["summary_trace"]
    assert all(call["evidence_input_tokens"] <= config.evidence_budget for call in row["model_calls"])
    assert row["retained_evidence_tokens"] <= config.evidence_budget
    assert all(state["summary_tokens_with_wrappers"] <= config.summary_budget for state in row["retention_trace"])
    assert len(row["summarized_source_document_ids"]) == 5
    assert row["counts"]["summary_generation_generated_tokens"] > 0
    assert "summary_generation_ms" in row["component_ms"]


def test_answer_gold_never_changes_retention_or_model_prompts():
    backend, corpus, source_config, source, example, config = fixture()
    altered = copy.deepcopy(example)
    altered["answers"] = ["entirely different gold"]
    altered["supporting_context_ids"] = ["doc-5"]
    first = run_pressure_case(backend, corpus, example, source, source_config, "retained_text", config)
    first_calls = backend.calls.copy()
    backend.calls.clear()
    second = run_pressure_case(backend, corpus, altered, source, source_config, "retained_text", config)
    assert backend.calls == first_calls
    assert first["retention_trace"] == second["retention_trace"]
    assert first["retained_evidence_token_ids_sha256"] == second["retained_evidence_token_ids_sha256"]
    assert first["prediction"] == second["prediction"]
    assert first["scores"] != second["scores"]


def test_aggregate_includes_all_questions_and_reports_overflow_separately():
    backend, corpus, source_config, source, example, config = fixture()
    rows = [run_pressure_case(backend, corpus, example, source, source_config, arm, config) for arm in ARMS]
    result = aggregate(rows, [], config, "complete", 4)
    assert result["status"] == "complete"
    assert len(result["metrics"]) == 8
    assert len(result["paired"]) == 6
    assert aggregate(rows[:3], [], config, "complete", 4)["status"] == "INCOMPLETE"
    for row in rows:
        row["retained_original_support_id_coverage"] = None
    assert all(row["mean_retained_original_support_id_coverage"] is None
               for row in aggregate(rows, [], config, "complete", 4)["metrics"])


def test_source_requires_basic_reference_for_every_question_before_inference(tmp_path):
    backend, corpus, source_config, source, example, _ = fixture()
    backend.name = "unit-reference-validation"
    backend.metadata = lambda: {"fixture": True}
    source.update(status="ok", example_id=example["id"], question=example["question"], answers=example["answers"],
                  case={"arm": "iterative_text", "max_rounds": 5}, scores={"exact_match": 0, "f1": 0},
                  cold_end_to_end_ms=1, counts={"retrieved_evidence_tokens": 720})
    question_path, corpus_path = tmp_path / "questions.jsonl", tmp_path / "corpus.jsonl"
    question_path.write_text(json.dumps(example) + "\n")
    corpus_path.write_text("\n".join(json.dumps(row) for row in corpus.values()) + "\n")
    source_config["max_answer_tokens"] = 48
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": {"config": source_config,
        "questions_sha256": sha(question_path.read_bytes()), "corpus_sha256": sha(corpus_path.read_bytes()),
        "backend": backend.metadata()}}))
    (tmp_path / "summary.json").write_text(json.dumps({"stop_reason": "complete"}))
    (tmp_path / "results.jsonl").write_text(json.dumps(source) + "\n")
    config = PressureConfig(str(question_path), str(corpus_path), str(tmp_path), str(tmp_path / "out"), expected_questions=1)
    with pytest.raises(ValueError, match="Basic RAG reference"):
        run_pressure_suite(backend, config)
    assert not backend.calls
