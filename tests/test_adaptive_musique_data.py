"""CPU-only tests of global corpus and disjoint sampling semantics."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_adaptive_musique", Path(__file__).parents[1] / "scripts" / "prepare_adaptive_musique.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def question(hops, index):
    return {"id": f"{hops}hop__{index}", "question": f"Question {hops}/{index}?",
            "question_decomposition": [{}] * hops}


def test_exact_title_text_dedup_and_content_order_remove_source_grouping():
    rows = [{"id": "first", "paragraphs": [{"idx": 0, "title": "A", "paragraph_text": "same", "is_supporting": True},
                                            {"idx": 1, "title": "A", "paragraph_text": "variant", "is_supporting": False}]},
            {"id": "second", "paragraphs": [{"idx": 0, "title": "A", "paragraph_text": "same", "is_supporting": False},
                                             {"idx": 1, "title": "B", "paragraph_text": "same", "is_supporting": True}]}]
    corpus, provenance, mapping = prepare.build_corpus(rows)
    other, _, _ = prepare.build_corpus(list(reversed(rows)))
    assert corpus == other and len(corpus) == 3
    assert all(set(row) == {"id", "title", "text"} for row in corpus)
    assert mapping[("first", 0)] == mapping[("second", 0)]
    assert mapping[("first", 0)] != mapping[("second", 1)]
    assert sum(len(row["source_occurrences"]) for row in provenance) == 4


def test_balanced_selection_excludes_previous_ids_texts_and_is_order_independent():
    rows = [question(hops, index) for hops in (2, 3, 4) for index in range(8)]
    excluded = [rows[0], rows[8], rows[16]]
    first, _ = prepare.select_questions(rows, excluded, 123, test_per_hop=2, dev_per_hop=1)
    second, _ = prepare.select_questions(list(reversed(rows)), excluded, 123, test_per_hop=2, dev_per_hop=1)
    assert first == second
    dev_ids, test_ids = [{row["id"] for row in first[split]} for split in ("dev", "test")]
    assert not dev_ids & test_ids
    assert not (dev_ids | test_ids) & {row["id"] for row in excluded}
    assert len(dev_ids) == 3 and len(test_ids) == 6
    for split, expected in (("dev", 1), ("test", 2)):
        assert all(sum(prepare.structural_hops(row) == hops for row in first[split]) == expected for hops in (2, 3, 4))
