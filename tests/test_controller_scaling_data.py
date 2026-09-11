"""CPU checks for fresh split isolation, identity-only selection, and source safety."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("controller_scaling_data", Path(__file__).parents[1] / "scripts/prepare_controller_scaling_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def row(hop, index):
    return {"id": f"{hop}hop__{index}", "question": f"Where is subject {hop}/{index}?",
            "question_decomposition": [{}] * hop}


def test_normalization_catches_unicode_case_and_whitespace_without_removing_words():
    assert prepare.normalized_question("  ＷＨＥＲＥ\u00a0is\tStraße? ") == "where is strasse?"
    assert prepare.normalized_question("The tower?") != prepare.normalized_question("Tower?")


def test_selection_excludes_prior_id_and_normalized_question_alias_and_is_order_independent():
    rows = [row(hop, index) for hop in (2, 3, 4) for index in range(12)]
    excluded = [prepare.identity(rows[0], "old"),
                prepare.identity({"id": "another-id", "question": rows[12]["question"].upper() + "  "}, "pilot")]
    first, eligible, blocked = prepare.select_questions(rows, excluded, 123, 2, 3)
    second, _, _ = prepare.select_questions(list(reversed(rows)), list(reversed(excluded)), 123, 2, 3)
    assert first == second
    assert eligible == {"2": 11, "3": 11, "4": 12}
    assert {item["id"] for item in blocked} == {rows[0]["id"], rows[12]["id"]}
    assert blocked[1]["id_match_sources"] == []
    assert blocked[1]["normalized_question_match_sources"] == ["pilot"]
    all_chosen = first["dev"] + first["test"]
    assert len({item["id"] for item in all_chosen}) == 15
    assert len({prepare.normalized_question(item["question"]) for item in all_chosen}) == 15
    for split, count in (("dev", 2), ("test", 3)):
        assert all(sum(prepare.structural_hops(item) == hop for item in first[split]) == count for hop in (2, 3, 4))


def test_selection_suppresses_normalized_duplicates_even_across_hop_groups():
    rows = [row(hop, index) for hop in (2, 3, 4) for index in range(12)]
    rows[12]["question"] = rows[0]["question"].upper()
    chosen, _, _ = prepare.select_questions(rows, [], 42, 5, 6)
    selected = chosen["dev"] + chosen["test"]
    assert len({prepare.normalized_question(item["question"]) for item in selected}) == 33


def test_identity_audit_loader_ignores_outcome_fields():
    bad = object()
    audit = {"observed_model_pilots": [{"source": "pilot", "examples": [row(2, 0)],
                                         "scores": bad, "successful_records": bad}],
             "cohorts": {"original": {"examples": [row(3, 0)], "metrics": bad}}, "predictions": bad}
    identities = prepare.identities_from_audit(audit)
    assert [item["source"] for item in identities] == ["pilot:pilot", "cohort:original"]
    assert all(set(item) == {"id", "question", "normalized_question", "source"} for item in identities)


def test_selection_does_not_depend_on_scores_answers_or_sizes():
    rows = [row(hop, index) for hop in (2, 3, 4) for index in range(8)]
    first, _, _ = prepare.select_questions(rows, [], 4, 1, 2)
    for index, item in enumerate(rows):
        item.update(answer="other", score=index, support_tokens=10000 - index)
    second, _, _ = prepare.select_questions(rows, [], 4, 1, 2)
    assert {split: [item["id"] for item in values] for split, values in first.items()} == {
        split: [item["id"] for item in values] for split, values in second.items()}


@pytest.mark.parametrize("kind", ["duplicate_id", "insufficient", "bad_count"])
def test_invalid_selection_fails_closed(kind):
    rows = [row(hop, index) for hop in (2, 3, 4) for index in range(3)]
    if kind == "duplicate_id":
        rows.append(rows[0])
    with pytest.raises(ValueError):
        prepare.select_questions(rows, [], 1, dev_per_hop=0 if kind == "bad_count" else 2, test_per_hop=2)


def test_corpus_rejects_hidden_gold_metadata_and_content_changes():
    corpus = [{"id": "hash", "title": "Title", "text": "Content"}]
    assert prepare.verify_corpus(json.dumps(corpus[0]).encode(), corpus) == corpus
    with pytest.raises(ValueError, match="forbidden metadata"):
        prepare.verify_corpus(json.dumps({**corpus[0], "answer": "gold"}).encode(), corpus)
    with pytest.raises(ValueError, match="official archive"):
        prepare.verify_corpus(json.dumps({**corpus[0], "text": "changed"}).encode(), corpus)


def test_existing_frozen_output_is_not_overwritten(tmp_path):
    output = tmp_path / "frozen"
    output.mkdir()
    (output / "dev.jsonl").write_text("original")
    with pytest.raises(ValueError, match="never overwritten"):
        prepare.prepare(tmp_path, output)
    assert (output / "dev.jsonl").read_text() == "original"


def test_support_sizes_are_separate_and_use_exact_wrapper_before_cap():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=list(text))
    corpus = [{"id": "doc", "title": "T", "text": "Some text"}]
    question = {"id": "q", "split": "test", "category": "2hop", "supporting_context_ids": ["doc"]}
    before = dict(question)
    sizes = prepare.support_sizes([question], corpus, Tokenizer(), per_document_cap=10)
    assert question == before
    assert sizes[0]["full_support_fragment_tokens"] == len("\n<document>\nT\nSome text\n</document>\n")
    assert sizes[0]["capped_support_fragment_tokens"] == 10
