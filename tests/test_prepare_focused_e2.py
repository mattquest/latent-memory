"""Structural selection fixtures; no benchmark outcomes are used."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("focused_e2", Path(__file__).parents[1] / "scripts/prepare_focused_e2.py")
focused = importlib.util.module_from_spec(spec)
spec.loader.exec_module(focused)


def test_same_first16_structural_groups_preserve_records_and_remove_only_duplicates(tmp_path):
    rows = [{"id": f"q{i}", "evidence": [{"id": f"d{j}"} for j in (range([6, 7, 10, 20, 40][i % 5]))],
             "answers": [str(i)], "irrelevant_outcome": i % 2} for i in range(20)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = focused.prepare(source, tmp_path / "out")
    assert result["planned_conditions"] == 212
    assert result["omitted_repeated_schedule_conditions"] == 108
    assert result["selected_question_ids_in_source_order"] == [f"q{i}" for i in range(16)]
    restored = {row["id"]: row for group in result["outputs"] for row in
                (json.loads(line) for line in Path(group["path"]).read_text().splitlines())}
    assert restored == {row["id"]: row for row in rows[:16]}
    for group in result["outputs"]:
        assert len(group["executed_evidence_hops"]) == len(set(group["executed_evidence_hops"]))
        assert group["sha256"] == focused.sha(Path(group["path"]).read_bytes())
    assert focused.prepare(source, tmp_path / "out") == result
