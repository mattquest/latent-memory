"""CPU protocol fixtures only; fake outputs are never experiment evidence."""
from dataclasses import dataclass, field, replace
import json
import math

import pytest

from eval.adaptive_experiments import (
    AdaptiveCase, AdaptiveConfig, Corpus, Evidence, Work, aggregate, case_id, cases_for, parse_action,
    PROTOCOL, CONTROLLER_PROTOCOL, controller_policy, decision_seed,
    read_questions, read_records, rrf_merge, run_adaptive_case,
)


@dataclass(frozen=True)
class Block:
    token_ids: tuple
    metadata: dict = field(default_factory=dict)
    quant: str = "bf16"

    @property
    def seq_len(self):
        return len(self.token_ids)

    @property
    def nbytes(self):
        return self.seq_len * (2 if self.quant == "bf16" else 1)


class RecordingBackend:
    name = "cpu-protocol-mock-not-for-experiments"
    max_context = 8192

    def __init__(self, actions=("SEARCH: beta location", "ANSWER"), answer="FOUND"):
        self.actions = list(actions)
        self.answer = answer
        self.inputs, self.prefixes, self.prefills, self.appends, self.concats = [], [], [], [], []
        self.quantizations = 0
        self.last_decode_stats = {}

    def encode(self, text):
        return list(text.encode())

    def decode_tokens(self, ids):
        return bytes(ids).decode()

    def synchronize(self):
        pass

    def prefill(self, ids):
        self.prefills.append(tuple(ids))
        return Block(tuple(ids))

    def append(self, prefix, ids):
        self.appends.append(tuple(ids))
        return Block(prefix.token_ids + tuple(ids))

    def concat(self, blocks, bridge_ratio=0):
        self.concats.append(tuple(block.token_ids for block in blocks))
        ids = tuple(token for block in blocks for token in block.token_ids)
        return Block(ids, {"bridge_tokens_recomputed": math.ceil(len(ids) * bridge_ratio)})

    def quantize(self, block, bits=8):
        self.quantizations += 1
        return replace(block, quant=f"int{bits}")

    def decode(self, prefix, ids, max_tokens=48):
        full = (list(prefix.token_ids) if prefix else []) + list(ids)
        self.inputs.append(tuple(full))
        self.prefixes.append(prefix.token_ids if prefix else None)
        if "Task: Give only the short answer" in self.decode_tokens(ids):
            text = self.answer
        else:
            text = self.actions.pop(0)
        output = self.encode(text)[:max_tokens]
        self.last_decode_stats = {"stop_reason": "eos" if len(output) < max_tokens else "max_tokens"}
        return output

    def memory_stats(self):
        return {"fixture_only": True}


@pytest.fixture
def setup_case(tmp_path):
    path = tmp_path / "corpus.jsonl"
    documents = [{"id": "a", "title": "Alpha", "text": "Alpha points to beta."},
                 {"id": "b", "title": "Beta", "text": "Beta location is FOUND."},
                 {"id": "c", "title": "Unrelated", "text": "Other facts."}]
    path.write_text("".join(json.dumps(document) + "\n" for document in documents))
    corpus = Corpus(path)
    example = {"id": "q", "dataset": "fixture", "split": "test", "question": "Alpha destination?",
               "answers": ["FOUND"], "supporting_context_ids": ["a", "b"], "category": "2hop"}
    config = AdaptiveConfig("unused", str(path), str(tmp_path / "out"), initial_top_k=1,
                            documents_per_round=1, max_documents=3, max_document_tokens=300,
                            max_evidence_tokens=1024)
    return corpus, example, config


def test_exact_incremental_matches_fulltext_inputs_without_retaining_actions(setup_case):
    corpus, example, config = setup_case
    text_backend, native_backend = RecordingBackend(), RecordingBackend()
    text = run_adaptive_case(text_backend, corpus, example, AdaptiveCase("iterative_text", 5), config)
    native = run_adaptive_case(native_backend, corpus, example, AdaptiveCase("incremental_kv_bf16", 5), config)
    assert text_backend.inputs == native_backend.inputs
    assert text["search_queries"] == native["search_queries"] == ["Alpha destination?", "beta location"]
    assert text["evidence_ids"] == native["evidence_ids"] == ["a", "b"]
    assert native["executed_retrieval_rounds"] == 2
    for prefix in native_backend.prefixes:
        assert b"Previous searches" not in bytes(prefix)
        assert b"SEARCH: beta location" not in bytes(prefix)
        assert b"Task:" not in bytes(prefix)
    assert native["counts"]["internal_decoded_tokens"] == len("SEARCH: beta locationANSWER")
    assert native["counts"]["cache_prefill_tokens"] == len(native_backend.prefixes[-1])
    assert native["counts"]["controller_prefill_tokens"] < text["counts"]["controller_prefill_tokens"]
    assert sum(native["component_ms"].values()) == pytest.approx(native["cold_end_to_end_ms"])


def test_initial_evidence_same_for_basic_and_iterative_and_gold_never_enters_input(setup_case):
    corpus, example, config = setup_case
    basic_backend = RecordingBackend(actions=())
    basic = run_adaptive_case(basic_backend, corpus, example, AdaptiveCase("basic_rag", 1), config)
    changed = {**example, "answers": ["SECRET_GOLD"], "supporting_context_ids": ["c"],
               "metadata": {"decomposition": "SECRET_GOLD"}, "evidence": [{"text": "SECRET_GOLD"}]}
    first, second = RecordingBackend(), RecordingBackend()
    a = run_adaptive_case(first, corpus, example, AdaptiveCase("iterative_text", 5), config)
    b = run_adaptive_case(second, corpus, changed, AdaptiveCase("iterative_text", 5), config)
    assert first.inputs == second.inputs
    assert all(b"SECRET_GOLD" not in bytes(ids) for ids in second.inputs)
    assert a["trace"][0]["new_document_ids"] == basic["evidence_ids"]
    assert a["prediction"] == b["prediction"]
    assert a["scores"] != b["scores"]  # gold changes scoring only


def test_independent_relay_cold_prefill_quantization_and_action_counts(setup_case):
    corpus, example, config = setup_case
    backend = RecordingBackend()
    row = run_adaptive_case(backend, corpus, example, AdaptiveCase("relay_kv_int8", 5), config)
    assert len(backend.prefills) == 3  # prologue, then two independent documents
    assert len(backend.concats) == backend.quantizations == 2
    assert row["counts"]["cache_prefill_tokens"] == sum(map(len, backend.prefills))
    assert row["counts"]["bridge_recomputed_tokens"] > 0
    assert row["counts"]["internal_decoded_tokens"] > 0
    assert row["cache"]["quant"] == "int8"
    assert row["component_ms"]["cache_construction_ms"] > 0


def test_complete_context_output_reservation_is_enforced(setup_case):
    corpus, example, config = setup_case
    tiny = replace(config, max_model_context_tokens=64)
    with pytest.raises(ValueError, match="Complete model input/output reservation"):
        run_adaptive_case(RecordingBackend(), corpus, example, AdaptiveCase("basic_rag", 1), tiny)


def test_invalid_and_repeated_actions_are_visible_fallbacks(setup_case):
    corpus, example, config = setup_case
    invalid = run_adaptive_case(RecordingBackend(actions=("some explanation",)), corpus, example,
                                AdaptiveCase("iterative_text", 5), config)
    assert invalid["invalid_actions"] == 1
    assert invalid["stop_reason"] == "invalid_action_fallback_to_final"
    repeated = run_adaptive_case(RecordingBackend(actions=("SEARCH: Alpha destination?",)), corpus, example,
                                 AdaptiveCase("iterative_text", 5), config)
    assert repeated["stop_reason"] == "repeated_query_fallback_to_final"
    assert repeated["counts"]["retrieval_calls"] == 1


def test_expansion_is_question_only_and_keeps_baseline_document_budget(setup_case):
    corpus, example, config = setup_case
    backend = RecordingBackend(actions=("SEARCH: beta location",))
    row = run_adaptive_case(backend, corpus, example, AdaptiveCase("expanded_rag", 1), config)
    assert b"Alpha points to beta" not in bytes(backend.inputs[0])
    assert row["counts"]["retrieval_calls"] == 2
    assert row["counts"]["retrieved_documents"] == config.initial_top_k
    assert row["document_budget"] == config.initial_top_k


def test_action_parser_and_six_arm_default():
    assert parse_action("ANSWER")["kind"] == "answer"
    assert parse_action("SEARCH: a name")["query"] == "a name"
    assert not parse_action("ANSWER: invented fact")["valid"]
    assert not parse_action("SEARCH: first\nSEARCH: second")["valid"]
    assert not parse_action("SEARCH: !!!")["valid"]
    assert not parse_action("SEARCH: the who should")["valid"]
    config = AdaptiveConfig("q", "c", "out")
    assert len(cases_for(config)) == 6
    assert config.initial_top_k == 6 and config.documents_per_round == 3 and config.max_documents == 18


def test_corpus_digest_and_artifact_reader_are_strict(setup_case, tmp_path):
    corpus, example, config = setup_case
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({**example, "metadata": {"corpus_sha256": "wrong"}}) + "\n")
    with pytest.raises(ValueError, match="corpus hash"):
        read_questions(path, corpus, 1)
    ledger = tmp_path / "partial.jsonl"
    raw = b'{"id":"one"}\n{"id":'
    ledger.write_bytes(raw)
    assert read_records(ledger) == [{"id": "one"}]
    assert ledger.read_bytes() == raw  # artifact reporting cannot truncate a running ledger


def test_conditional_recovery_pairs_only_common_basic_failures(setup_case):
    corpus, example, config = setup_case
    basic = run_adaptive_case(RecordingBackend(actions=(), answer="WRONG"), corpus, example,
                              AdaptiveCase("basic_rag", 1), config)
    enhanced = run_adaptive_case(RecordingBackend(), corpus, example, AdaptiveCase("iterative_text", 5), config)
    unpaired = {**enhanced, "id": "unpaired", "example_id": "other"}
    summary = aggregate([basic, enhanced, unpaired, enhanced])
    assert summary["successful_unique_runs"] == 3
    failure = next(row for row in summary["paired_vs_basic"] if row["subset"] == "basic_rag_exact_match_failures")
    assert failure["n_paired"] == failure["enhanced_correct"] == 1
    assert failure["paired_example_ids"] == ["q"]


def test_reranker_charges_all_candidates_but_delivers_baseline_budget(setup_case):
    corpus, example, config = setup_case
    class Reranker:
        last_stats = {"input_tokens": 777, "documents": 3}
        def score(self, question, documents):
            assert question == example["question"]
            assert len(documents) == 3
            return [float(document["id"] == "b") for document in documents]
    row = run_adaptive_case(RecordingBackend(actions=()), corpus, example,
                            AdaptiveCase("reranked_rag", 1), config, reranker=Reranker())
    assert row["evidence_ids"] == ["b"]
    assert row["counts"]["reranker_input_tokens"] == 777
    assert row["counts"]["reranker_scored_documents"] == 3
    assert row["counts"]["all_models_prefill_tokens"] == row["counts"]["primary_model_prefill_tokens"] + 777
    assert row["component_ms"]["reranking_ms"] > 0


def test_cache_comparison_includes_same_question_f1_and_latency_pairs(setup_case):
    corpus, example, config = setup_case
    text = run_adaptive_case(RecordingBackend(), corpus, example, AdaptiveCase("iterative_text", 5), config)
    native = run_adaptive_case(RecordingBackend(), corpus, example, AdaptiveCase("incremental_kv_bf16", 5), config)
    relay = run_adaptive_case(RecordingBackend(), corpus, example, AdaptiveCase("relay_kv_bf16", 5), config)
    text["cold_end_to_end_ms"], native["cold_end_to_end_ms"], relay["cold_end_to_end_ms"] = 100, 50, 200
    pairs = aggregate([text, native, relay])["paired_core_comparisons"]
    exact = next(pair for pair in pairs if pair["target_arm"] == "incremental_kv_bf16")
    assert exact["n_paired"] == 1
    assert exact["f1_difference"] == 0
    assert exact["exact_match_paired_bootstrap_ci95"] is None
    assert exact["median_paired_cold_latency_ratio"] == .5
    assert exact["mean_cold_latency_difference_ms"] == -50
    assert any(pair["target_arm"] == "relay_kv_bf16" and pair["comparator_arm"] == "incremental_kv_bf16" for pair in pairs)


class ThinkingRecordingBackend(RecordingBackend):
    def __init__(self, *args, reasoning_eos=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_eos = reasoning_eos
        self.reasoning_calls = []

    def decode_with_reasoning(self, prefix, ids, *, max_reasoning_tokens, max_tokens, controls, **sampling):
        assert bytes(ids).endswith(b"<think>\n")
        assert b"</think>" not in bytes(ids)
        self.reasoning_calls.append((prefix, tuple(ids), max_reasoning_tokens, max_tokens, controls, sampling))
        output = [] if self.reasoning_eos else super().decode(prefix, ids, max_tokens=max_tokens)
        thought = self.encode("transient reasoning")[:max_reasoning_tokens]
        forced = [] if self.reasoning_eos else controls["forced_closure"]
        self.last_decode_stats = {
            "thinking": "bounded_transient", "reasoning_token_ids": thought,
            "reasoning_output_token_ids": thought, "raw_reasoning": self.decode_tokens(thought),
            "action_token_ids": output, "forced_control_token_ids": forced,
            "reasoning_generated_tokens": len(thought), "action_generated_tokens": len(output),
            "forced_control_tokens": len(forced), "generated_tokens": len(thought) + len(output),
            "stop_reason": "reasoning_eos" if self.reasoning_eos else "eos",
            "reasoning_stop_reason": "eos" if self.reasoning_eos else "max_reasoning_tokens",
            "action_stop_reason": "not_started" if self.reasoning_eos else "eos",
            "reasoning_cap_reached": not self.reasoning_eos, "sampling": sampling}
        return output


def test_thinking_controller_is_matched_across_evidence_methods_and_transient(setup_case):
    corpus, example, base = setup_case
    config = replace(base, max_reasoning_tokens=4)
    rows, backends = [], []
    for arm in ("iterative_text", "incremental_kv_bf16", "relay_kv_bf16", "relay_kv_int8"):
        backend = ThinkingRecordingBackend()
        row = run_adaptive_case(backend, corpus, example, AdaptiveCase(arm, 5), config)
        rows.append(row)
        backends.append(backend)
        decisions = [step for step in row["trace"] if step["event"] == "decision"]
        assert len(decisions) == len(backend.reasoning_calls) == 2
        assert row["counts"]["controller_reasoning_tokens"] == 8
        assert row["counts"]["controller_action_tokens"] == len("SEARCH: beta locationANSWER")
        assert row["counts"]["controller_forced_tokens"] == len("\n</think>\n\n") * 2
        assert row["counts"]["internal_decoded_tokens"] == 8 + len("SEARCH: beta locationANSWER")
        assert all(step["controller_decode"]["raw_reasoning"] == "tran" for step in decisions)
        assert all(step["controller_decode"]["action_token_ids"] == step["action_token_ids"] for step in decisions)
        assert row["protocol_version"] == CONTROLLER_PROTOCOL
        assert row["controller_config"] == {**controller_policy(config), "max_action_tokens": 32, "max_answer_tokens": 48}
        assert b"<think>\n\n</think>\n\n" in bytes(backend.inputs[-1])  # final remains nonthinking
        for prefix in backend.prefixes:
            if prefix is not None:
                assert b"transient" not in bytes(prefix)
                assert b"SEARCH: beta location" not in bytes(prefix)
                assert b"Previous searches" not in bytes(prefix)
        assert row["counts"]["total_prefill_tokens"] == sum(row["counts"][key] for key in (
            "controller_prefill_tokens", "final_prefill_tokens", "cache_prefill_tokens",
            "bridge_recomputed_tokens", "controller_forced_tokens"))
    assert all(backend.inputs == backends[0].inputs for backend in backends)
    assert all(row["search_queries"] == rows[0]["search_queries"] for row in rows)
    assert aggregate(rows)["protocol_version"] == CONTROLLER_PROTOCOL


@pytest.mark.parametrize("kwargs", [{"reasoning_eos": True}, {"actions": ("invalid action",)}])
def test_thinking_invalid_or_eos_action_explicitly_falls_back_to_nonthinking_final(setup_case, kwargs):
    corpus, example, base = setup_case
    backend = ThinkingRecordingBackend(**kwargs)
    row = run_adaptive_case(backend, corpus, example, AdaptiveCase("iterative_text", 5),
                            replace(base, max_reasoning_tokens=4))
    assert row["stop_reason"] == "invalid_action_fallback_to_final"
    assert row["invalid_actions"] == 1
    assert row["prediction"] == "FOUND"
    decision = next(step for step in row["trace"] if step["event"] == "decision")
    if kwargs.get("reasoning_eos"):
        assert decision["raw_action"] == ""
        assert decision["controller_decode"]["stop_reason"] == "reasoning_eos"
        assert row["counts"]["controller_forced_tokens"] == 0


def test_reasoning_reservation_charges_forced_closure_before_backend_call(setup_case):
    _corpus, _example, base = setup_case
    backend = ThinkingRecordingBackend()
    config = replace(base, max_reasoning_tokens=2, max_model_context_tokens=19)
    evidence = Evidence(backend, config, "iterative_text", Work(), [1, 2, 3])
    with pytest.raises(ValueError, match="reservation 20 exceeds 19"):
        evidence.generate("p", max_tokens=3, controller=True)
    assert backend.inputs == backend.reasoning_calls == []


def test_default_protocol_ids_preserved_and_thinking_budget_is_distinct():
    import hashlib
    case = AdaptiveCase("iterative_text", 5)
    old_payload = [PROTOCOL, "q", {"arm": "iterative_text", "max_rounds": 5}]
    expected = hashlib.sha256(json.dumps(old_payload, sort_keys=True).encode()).hexdigest()[:24]
    assert case_id("q", case) == case_id("q", case, 0) == expected
    assert len({case_id("q", case, budget) for budget in (0, 128, 256)}) == 3
    with pytest.raises(ValueError, match="Reasoning token budget must be nonnegative"):
        replace(AdaptiveConfig("q", "c", "out"), max_reasoning_tokens=-1).validate()


def test_sampling_policy_is_common_across_models_arms_and_budgets_with_greedy_final(setup_case):
    corpus, example, base = setup_case
    class SampledBackend(ThinkingRecordingBackend):
        def __init__(self):
            super().__init__()
            self.short_sampling = []
        def decode(self, prefix, ids, max_tokens=48, **sampling):
            final = "Task: Give only the short answer" in self.decode_tokens(ids)
            if final:
                assert sampling == {}
            else:
                self.short_sampling.append(sampling)
            return super().decode(prefix, ids, max_tokens=max_tokens)
    traces = []
    for reasoning in (0, 256):
        for arm in ("iterative_text", "incremental_kv_bf16", "relay_kv_bf16"):
            config = replace(base, max_reasoning_tokens=reasoning, controller_temperature=.6,
                             controller_top_p=.95, controller_top_k=20)
            backend = SampledBackend()
            row = run_adaptive_case(backend, corpus, example, AdaptiveCase(arm, 5), config)
            decisions = [step for step in row["trace"] if step["event"] == "decision"]
            traces.append([step["controller_seed"] for step in decisions])
            expected = [{"temperature": .6, "top_p": .95, "top_k": 20,
                         "seed": decision_seed(config.seed, example["id"], "decision", round_number)}
                        for round_number in (1, 2)]
            if reasoning:
                assert [call[-1] for call in backend.reasoning_calls] == expected
            else:
                assert backend.short_sampling == expected
            assert row["protocol_version"] == CONTROLLER_PROTOCOL
            assert row["controller_config"]["controller_temperature"] == .6
    assert all(trace == traces[0] for trace in traces)
    assert traces[0][0] != traces[0][1]
    case = AdaptiveCase("iterative_text", 5)
    assert case_id("q", case) != case_id("q", case, controller_temperature=.6,
                                        controller_top_p=.95, controller_top_k=20)


def test_aggregate_rejects_different_controller_budgets_even_with_same_protocol(setup_case):
    corpus, example, base = setup_case
    rows = [run_adaptive_case(ThinkingRecordingBackend(), corpus, example,
                              AdaptiveCase("iterative_text", 5), replace(base, max_reasoning_tokens=budget))
            for budget in (128, 256)]
    with pytest.raises(ValueError, match="different controller configurations"):
        aggregate(rows)
