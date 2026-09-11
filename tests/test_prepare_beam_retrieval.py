"""Timing isolation fixtures; never loads a model or rewrites real datasets."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_beam", Path(__file__).parents[1] / "scripts" / "prepare_beam_retrieval.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class WordTokenizer:
    def encode(self, text, **_kwargs):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_clock_changes_only_sidecar_not_semantic_jsonl(tmp_path):
    corpus = [{"id": "message1", "title": "Message 1", "text": "Amber registry code is 123456",
               "source_message_id": 1, "role": "user", "chronological_index": 0}]
    probes = [{"id": "probe1", "dataset": "beam", "question": "Amber registry code?",
               "answers": ["123456"], "category": "information_extraction", "metadata": {}, "evidence": []}]
    prepare.write_jsonl(tmp_path / "beam_1m_1_corpus.jsonl", corpus)
    prepare.write_jsonl(tmp_path / "beam_1m_1_probes.jsonl", probes)
    path = tmp_path / "beam_1m_1_retrieved.jsonl"
    sidecar = tmp_path / "beam_1m_1_retrieved.timings.json"
    ticks = iter([0., 1., 2., 3.])
    first = prepare.prepare_conversation(1, tmp_path, WordTokenizer(), clock=lambda: next(ticks))
    semantic_before, timing_before = path.read_bytes(), sidecar.read_bytes()
    ticks = iter([10., 14., 20., 27.])
    second = prepare.prepare_conversation(1, tmp_path, WordTokenizer(), clock=lambda: next(ticks))
    assert path.read_bytes() == semantic_before
    assert sidecar.read_bytes() != timing_before
    assert first["sha256"] == second["sha256"]
    assert first["timing_sidecar_sha256"] != second["timing_sidecar_sha256"]
    row = json.loads(semantic_before)
    assert row["metadata"]["timing_sidecar"] == sidecar.name
    assert "retrieval_seconds" not in row["metadata"]
    assert "index_preparation_seconds" not in row["metadata"]
    timing = json.loads(sidecar.read_text())
    assert timing["semantic_dataset_sha256"] == prepare.sha256_file(path)
    assert timing["queries"]["probe1"]["semantic_example_sha256"] == prepare.semantic_example_sha256(row)
    assert timing["queries"]["probe1"]["retrieval_seconds"] == 7
    assert timing["index_preparation_seconds"] == 4
    # Copying rows into combined/subsample inputs retains the same safe reference.
    prepare.write_jsonl(tmp_path / "combined.jsonl", [row])
    combined_row = json.loads((tmp_path / "combined.jsonl").read_text())
    assert combined_row["metadata"]["timing_sidecar"] == sidecar.name
    assert prepare.semantic_example_sha256(combined_row) == timing["queries"]["probe1"]["semantic_example_sha256"]
