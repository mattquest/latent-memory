"""CPU-only integrity tests for optional cost receipts; no model runtime import."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import pytest


def load_attach():
    source_path = Path(os.environ.get("TIMING_SIDECAR_TEST_SOURCE", str(Path(__file__).parents[1] / "eval" / "real_experiments.py")))
    source = source_path.read_text()
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_attach_timing_sidecars")
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_attach_timing_sidecars"]


@pytest.fixture
def attach():
    return load_attach()


@pytest.fixture
def receipt_fixture(tmp_path):
    row = {"id": "fixture", "question": "Which code?", "answers": ["123456"],
           "metadata": {"timing_sidecar": "conversation.timings.json"}}
    source = tmp_path / "conversation.jsonl"
    source.write_text(json.dumps(row, sort_keys=True) + "\n")
    receipt = {"schema_version": 1, "semantic_dataset": source.name,
               "semantic_dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "index_preparation_seconds": 2.5,
               "queries": {row["id"]: {"semantic_example_sha256": hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                                         "retrieval_seconds": .125}}}
    sidecar = tmp_path / row["metadata"]["timing_sidecar"]
    sidecar.write_text(json.dumps(receipt))
    combined = tmp_path / "combined.jsonl"
    combined.write_bytes(source.read_bytes())
    return row, source, sidecar, receipt, combined


def test_valid_hydration_uses_cost_whitelist_and_preserves_semantic_file(attach, receipt_fixture):
    row, source, sidecar, receipt, combined = receipt_fixture
    before = copy.deepcopy(row)
    source_bytes, combined_bytes = source.read_bytes(), combined.read_bytes()
    receipt["question"] = "not a question override"
    receipt["queries"]["fixture"]["answers"] = ["wrong"]
    sidecar.write_text(json.dumps(receipt))
    attach([row], combined)
    assert row["question"] == before["question"] and row["answers"] == before["answers"]
    assert row["metadata"]["retrieval_seconds"] == .125
    assert row["metadata"]["index_preparation_seconds"] == 2.5
    assert row["metadata"]["timing_provenance"]["sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert source.read_bytes() == source_bytes and combined.read_bytes() == combined_bytes


def test_absent_optional_sidecar_is_marked_without_inventing_cost(attach, receipt_fixture):
    row, source, sidecar, receipt, combined = receipt_fixture
    sidecar.unlink()
    attach([row], combined)
    assert row["metadata"]["timing_receipt_status"] == "missing_optional_sidecar"
    assert "retrieval_seconds" not in row["metadata"]


def test_changed_semantic_source_hash_rejected(attach, receipt_fixture):
    row, source, sidecar, receipt, combined = receipt_fixture
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        attach([row], combined)


def test_changed_semantic_example_hash_rejected(attach, receipt_fixture):
    row, source, sidecar, receipt, combined = receipt_fixture
    row["question"] = "Different semantic question"
    with pytest.raises(ValueError, match="example hash mismatch"):
        attach([row], combined)


@pytest.mark.parametrize("name", ["../outside.json", "folder/file.json", "folder\\file.json", "/outside.json", "..", ".", "", 123])
def test_unsafe_sidecar_basename_rejected(attach, receipt_fixture, name):
    row, source, sidecar, receipt, combined = receipt_fixture
    row["metadata"]["timing_sidecar"] = name
    with pytest.raises(ValueError, match="safe basenames"):
        attach([row], combined)


@pytest.mark.parametrize("target", ["receipt", "semantic"])
def test_symlink_escaping_data_directory_rejected(attach, receipt_fixture, tmp_path, target):
    row, source, sidecar, receipt, combined = receipt_fixture
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    selected = sidecar if target == "receipt" else source
    outside.write_bytes(selected.read_bytes())
    selected.unlink()
    selected.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes dataset directory"):
        attach([row], combined)


@pytest.mark.parametrize("field", ["retrieval_seconds", "index_preparation_seconds"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_nonfinite_negative_or_nonnumeric_cost_rejected(attach, receipt_fixture, field, value):
    row, source, sidecar, receipt, combined = receipt_fixture
    target = receipt if field == "index_preparation_seconds" else receipt["queries"]["fixture"]
    target[field] = value
    sidecar.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="finite nonnegative"):
        attach([row], combined)


def test_missing_query_rejected(attach, receipt_fixture):
    row, source, sidecar, receipt, combined = receipt_fixture
    receipt["queries"].clear()
    sidecar.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="lacks the requested example"):
        attach([row], combined)
