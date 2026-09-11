"""CPU protocol fixtures; these fabricated outputs are never benchmark evidence."""
from copy import deepcopy
from dataclasses import replace

import pytest

from eval import controller_screen as screen


class RecordingBackend:
    max_context = 8192

    def __init__(self, actions=("SEARCH: Beta location", "ANSWER"), answer="FOUND"):
        self.actions = list(actions)
        self.answer = answer
        self.calls = []

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode_tokens(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")

    def chat_tokens(self, messages):
        return self.encode("".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n")

    def clear_cache(self):
        pass

    def synchronize(self):
        pass

    def memory_stats(self):
        return {"cpu_fixture_only": True}

    def generate(self, messages, max_tokens, **sampling):
        self.calls.append({"messages": deepcopy(messages), "max_tokens": max_tokens, "sampling": sampling})
        is_final = "Task: Give only the short answer" in messages[-1]["content"]
        raw = self.answer if is_final else self.actions.pop(0)
        output = self.encode(raw)[:max_tokens]
        return {"raw_output": self.decode_tokens(output), "output_token_ids": output,
                "prompt_token_ids": self.chat_tokens(messages),
                "stats": {"sampled_tokens_including_eos": len(output) + int(len(output) < max_tokens)}}


class RecordingCorpus:
    def __init__(self, documents, rankings=None):
        self.documents = documents
        self.rankings = rankings or {}
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        docs = self.rankings.get(query, self.documents)
        return [(doc, 10.0 - index) for index, doc in enumerate(docs[:k])]


@pytest.fixture
def fixture_case():
    documents = [
        {"id": "a", "title": "Alpha", "text": "Alpha links to Beta."},
        {"id": "b", "title": "Beta", "text": "Beta location is FOUND."},
        {"id": "c", "title": "Distractor", "text": "Other records."},
    ]
    question = {"id": "q1", "question": "Where is Alpha?", "answers": ["FOUND"],
                "supporting_context_ids": ["a", "b"], "category": "2hop", "split": "dev"}
    config = screen.ScreenConfig(initial_top_k=1, documents_per_round=1, max_documents=3)
    return documents, question, config


def run(documents, question, config, backend=None, arm="iterative_text", corpus=None):
    backend = backend or RecordingBackend()
    corpus = corpus or RecordingCorpus(documents)
    return screen.run_case(backend, corpus, question, arm, config, "fixture-revision")


def test_labels_only_change_scoring_not_retrieval_prompts_or_sample_seeds(fixture_case):
    documents, question, config = fixture_case
    changed = {**question, "answers": ["SECRET_GOLD_ANSWER"],
               "supporting_context_ids": ["c"], "category": "SECRET_CATEGORY", "split": "SECRET_SPLIT",
               "metadata": {"decomposition": "SECRET_CHAIN"}, "evidence": [{"text": "SECRET_EVIDENCE"}]}
    second_docs = [{**doc, "is_supporting": True, "answer": "SECRET_DOCUMENT_LABEL"} for doc in documents]
    first, second = RecordingBackend(), RecordingBackend()
    left_corpus, right_corpus = RecordingCorpus(documents), RecordingCorpus(second_docs)
    left = run(documents, question, config, first, corpus=left_corpus)
    right = run(second_docs, changed, config, second, corpus=right_corpus)
    assert left_corpus.calls == right_corpus.calls
    assert first.calls == second.calls
    assert "SECRET_" not in str(second.calls)
    assert left["trace"] == right["trace"]
    assert left["prediction"] == right["prediction"] == "FOUND"
    assert left["scores"]["exact_match"] == 1 and right["scores"]["exact_match"] == 0
    assert left["support_annotation_coverage"] == 1 and right["support_annotation_coverage"] == 0


def test_common_initial_retrieval_and_same_final_prompt_at_same_evidence(fixture_case):
    documents, question, config = fixture_case
    basic_backend = RecordingBackend(actions=())
    iterative_backend = RecordingBackend(actions=("ANSWER",))
    basic_corpus, iterative_corpus = RecordingCorpus(documents), RecordingCorpus(documents)
    basic = run(documents, question, config, basic_backend, "basic_rag", basic_corpus)
    iterative = run(documents, question, config, iterative_backend, corpus=iterative_corpus)
    assert basic_corpus.calls == iterative_corpus.calls
    assert basic["trace"][0] == iterative["trace"][0]
    assert basic["evidence_ids"] == iterative["evidence_ids"] == ["a"]
    assert basic_backend.calls[-1] == iterative_backend.calls[-1]
    assert basic["counts"]["controller_calls"] == 0
    assert iterative["counts"]["controller_calls"] == 1


@pytest.mark.parametrize("action,stop", [
    ("SEARCH: WHERE IS ALPHA!", "repeated_query_fallback_to_final"),
    ("Here is an explanation", "invalid_action_fallback_to_final"),
    ("ANSWER: SECRET_IGNORED", "controller_answer"),
])
def test_stopping_always_uses_separate_greedy_final_generation(fixture_case, action, stop):
    documents, question, config = fixture_case
    backend = RecordingBackend(actions=(action,))
    corpus = RecordingCorpus(documents)
    row = run(documents, question, config, backend, corpus=corpus)
    assert row["stop_reason"] == stop
    assert row["executed_retrieval_rounds"] == len(corpus.calls) == 1
    assert [generation["phase"] for generation in row["generations"]] == ["decision", "final"]
    assert row["prediction"] == "FOUND"
    assert backend.calls[0]["sampling"] == {
        "temperature": config.controller_temperature, "top_p": config.controller_top_p,
        "top_k": config.controller_top_k, "presence_penalty": config.controller_presence_penalty,
        "seed": screen.seed_for(config, question["id"], "decision", 1),
    }
    assert backend.calls[1]["sampling"] == {}
    assert backend.calls[0]["max_tokens"] == config.max_action_tokens
    assert backend.calls[1]["max_tokens"] == config.max_answer_tokens
    assert "Previous searches" not in backend.calls[-1]["messages"][-1]["content"]
    assert "SECRET_IGNORED" not in str(backend.calls[-1])
    assert row["invalid_actions"] == int(stop.startswith("invalid"))


def test_new_queries_append_unique_evidence_and_keep_accounting_separate(fixture_case):
    documents, question, config = fixture_case
    backend = RecordingBackend()
    row = run(documents, question, config, backend)
    assert row["search_queries"] == [question["question"], "Beta location"]
    assert row["evidence_ids"] == ["a", "b"]
    assert row["executed_retrieval_rounds"] == 2
    assert row["counts"]["controller_calls"] == 2
    assert row["counts"]["controller_generated_tokens"] == len("SEARCH: Beta locationANSWER")
    assert row["counts"]["final_generated_tokens"] == len("FOUND")
    assert row["counts"]["controller_sampled_tokens_including_eos"] == len("SEARCH: Beta locationANSWER") + 2
    assert row["counts"]["total_prefill_tokens"] == sum(len(g["prompt_token_ids"]) for g in row["generations"])
    assert row["counts"]["total_generated_tokens"] == sum(len(g["output_token_ids"]) for g in row["generations"])
    assert sum(row["component_ms"].values()) == pytest.approx(row["cold_end_to_end_ms"])


@pytest.mark.parametrize("constraint,stop", [("rounds", "round_budget"), ("documents", "document_budget"),
                                             ("exhaustion", "no_new_evidence")])
def test_round_document_and_exhaustion_stops_do_not_generate_unused_actions(fixture_case, constraint, stop):
    documents, question, config = fixture_case
    if constraint == "rounds":
        config = replace(config, max_rounds=2)
    elif constraint == "documents":
        config = replace(config, max_documents=2)
    corpus = RecordingCorpus(documents, {"Beta location": documents[:1]} if constraint == "exhaustion" else None)
    backend = RecordingBackend(actions=("SEARCH: Beta location",))
    row = run(documents, question, config, backend, corpus=corpus)
    assert row["stop_reason"] == stop
    assert row["counts"]["controller_calls"] == 1
    assert row["executed_retrieval_rounds"] == 2
    assert [g["phase"] for g in row["generations"]] == ["decision", "final"]


def test_document_and_joined_evidence_caps_skip_overflow_without_using_annotations(fixture_case):
    _, question, config = fixture_case
    docs = [{"id": str(i), "title": "Long", "text": "x" * 200} for i in range(3)]
    config = replace(config, initial_top_k=3, max_document_tokens=60, max_evidence_tokens=120)
    backend = RecordingBackend(actions=())
    row = run(docs, question, config, backend, "basic_rag")
    assert row["evidence_ids"] == ["0", "1"]
    assert row["skipped_evidence_budget_ids"] == ["2"]
    assert row["counts"]["retrieved_evidence_tokens"] == 120
    assert all(len(f["token_ids"]) == 60 and f["truncated"] for f in row["delivered_fragments"])
    assert row["trace"][0]["evidence_tokens"] == 120


def test_unicode_fragment_roundtrip_cannot_expand_past_cap():
    backend = RecordingBackend()
    document = {"id": "unicode", "title": "", "text": "éééé"}
    # The byte tokenizer turns a cut two-byte character into a three-byte
    # replacement on decoding. bounded_fragment must retreat before returning.
    prefix = "\n<document>\n\n"
    cap = len(prefix.encode()) + 3
    result = screen.bounded_fragment(backend, document, cap)
    assert result["truncated"]
    assert result["token_ids"] == backend.encode(result["text"])
    assert len(result["token_ids"]) <= cap
    assert "�" not in result["text"]


def test_question_cap_rejects_before_search_or_generation(fixture_case):
    documents, question, config = fixture_case
    backend, corpus = RecordingBackend(), RecordingCorpus(documents)
    with pytest.raises(ValueError, match="question-token cap"):
        run(documents, {**question, "question": "x" * 257}, config, backend, corpus=corpus)
    assert corpus.calls == backend.calls == []


def test_complete_native_template_reservation_checked_before_inference(fixture_case):
    documents, question, config = fixture_case
    baseline = run(documents, question, config, RecordingBackend(actions=()), "basic_rag")
    exact = len(baseline["generations"][0]["prompt_token_ids"]) + config.max_answer_tokens
    accepted = run(documents, question, replace(config, max_context_tokens=exact), RecordingBackend(actions=()), "basic_rag")
    assert accepted["counts"]["max_model_context_tokens"] == exact
    for budget, backend_cap in [(exact - 1, 8192), (8192, exact - 1)]:
        backend = RecordingBackend(actions=())
        backend.max_context = backend_cap
        with pytest.raises(ValueError, match="reservation exceeds"):
            run(documents, question, replace(config, max_context_tokens=budget), backend, "basic_rag")
        assert backend.calls == []


def test_backend_cannot_silently_change_checked_template_or_exceed_output(fixture_case):
    documents, question, config = fixture_case
    class BadBackend(RecordingBackend):
        violation = "prompt"
        def generate(self, messages, max_tokens, **sampling):
            result = super().generate(messages, max_tokens, **sampling)
            if self.violation == "prompt":
                result["prompt_token_ids"] = [999] + result["prompt_token_ids"]
            else:
                result["output_token_ids"] = [1] * (max_tokens + 1)
            return result
    for violation, message in [("prompt", "checked native template"), ("output", "output-token budget")]:
        backend = BadBackend(actions=())
        backend.violation = violation
        with pytest.raises(ValueError, match=message):
            run(documents, question, config, backend, "basic_rag")


def test_overdue_final_call_is_not_reported_as_success(fixture_case, monkeypatch):
    documents, question, config = fixture_case
    clock = [0.0]
    monkeypatch.setattr(screen.time, "perf_counter", lambda: clock[0])
    class SlowBackend(RecordingBackend):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            clock[0] += 2
            return result
    with pytest.raises(TimeoutError, match="runtime budget"):
        run(documents, question, replace(config, max_case_seconds=1), SlowBackend(actions=()), "basic_rag")


def test_cold_timer_includes_pre_case_cache_cleanup(fixture_case, monkeypatch):
    documents, question, config = fixture_case
    clock = [0.0]
    monkeypatch.setattr(screen.time, "perf_counter", lambda: clock[0])
    class TimedBackend(RecordingBackend):
        def clear_cache(self):
            clock[0] += 0.25
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            clock[0] += 0.5
            return result
    row = run(documents, question, config, TimedBackend(actions=()), "basic_rag")
    assert row["cold_end_to_end_ms"] == 750
    assert row["component_ms"]["final_ms"] == 500
    assert row["component_ms"]["other_ms"] == 250


def test_empty_annotations_are_unavailable_coverage_not_zero_or_error(fixture_case):
    documents, question, config = fixture_case
    row = run(documents, {**question, "supporting_context_ids": []}, config)
    assert row["support_annotation_coverage"] is None
    assert row["scores"]["exact_match"] == 1


@pytest.mark.parametrize("field,value", [
    ("initial_top_k", 0), ("max_rounds", 6), ("max_documents", 19), ("max_context_tokens", 8193),
    ("max_answer_tokens", True), ("seed", -1), ("seed", 2**32), ("seed", True),
    ("controller_temperature", float("nan")), ("controller_temperature", float("inf")),
    ("controller_temperature", -0.1), ("controller_temperature", True),
    ("controller_top_p", float("nan")), ("controller_top_p", 0), ("controller_top_p", 1.1),
    ("controller_top_k", -1), ("controller_top_k", 1.5), ("controller_top_k", True),
    ("controller_presence_penalty", float("inf")), ("controller_presence_penalty", -2.1),
    ("controller_presence_penalty", 2.1), ("max_case_seconds", float("nan")), ("max_case_seconds", 601),
])
def test_invalid_configuration_rejected_without_backend_work(field, value):
    with pytest.raises(ValueError):
        replace(screen.ScreenConfig(), **{field: value}).validate()


def test_per_round_document_limits_and_sampler_boundary_values():
    for kwargs in ({"initial_top_k": 4, "max_documents": 3},
                   {"documents_per_round": 4, "initial_top_k": 1, "max_documents": 3}):
        with pytest.raises(ValueError, match="document budget"):
            replace(screen.ScreenConfig(), **kwargs).validate()
    screen.ScreenConfig(seed=0, controller_temperature=0, controller_top_p=1,
                        controller_top_k=0, controller_presence_penalty=-2).validate()
    screen.ScreenConfig(seed=2**32 - 1, controller_presence_penalty=2).validate()
