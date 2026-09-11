import gzip
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("validation_history", Path(__file__).resolve().parents[1] /
                                           "scripts/archive_validation_history.py")
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def test_preserves_failed_stopped_partial_and_historical_source_identity(tmp_path):
    runs, output = tmp_path / "runs", tmp_path / "report"
    put(runs, "native-validation/guard-status.json", {"status": "failed", "returncode": 1})
    put(runs, "native-validation/process.log", b'{"model":{"model_revision":"old-revision"}}\nAssertionError\n')
    put(runs, "development/guard-status.json", {"status": "stopped", "reason": "interrupted"})
    put(runs, "development/multihop/manifest.json", {"identity": {"source_code_sha256": {"engine.py": "old-source"}},
        "machine": {"git_head": "old-commit", "git_dirty": [" M engine.py"]}})
    partial = b'{"id":"complete","status":"ok"}\n{"id":"partial"'
    put(runs, "development/multihop/results.jsonl", partial)
    put(runs, "development/model.safetensors", b"never archive weights")
    put(runs, "overnight/results.jsonl", b"never archive main metrics")
    before = {str(path.relative_to(runs)): path.read_bytes() for path in runs.rglob("*") if path.is_file()}
    manifest = history.archive_history(runs, output)
    groups = {group["source"]: group for group in manifest["groups"]}
    assert groups["native-validation"]["observed_status"] == "failed"
    assert groups["development"]["observed_status"] == "stopped"
    assert not manifest["eligible_for_main_metrics"]
    assert all(not item["eligible_for_main_metrics"] for item in manifest["artifacts"])
    identity = groups["development"]["source_identity_receipts"][0]["reported_identity"]
    assert identity["identity"]["source_code_sha256"] == {"engine.py": "old-source"}
    assert identity["machine_recorded_source"]["git_head"] == "old-commit"
    sources = {item["source"] for item in manifest["artifacts"]}
    assert "overnight/results.jsonl" not in sources and "development/model.safetensors" not in sources
    assert gzip.decompress((output / "artifacts/development/multihop/results.jsonl.gz").read_bytes()) == partial
    assert before == {str(path.relative_to(runs)): path.read_bytes() for path in runs.rglob("*") if path.is_file()}


def test_archives_are_deterministic_and_existing_gzip_is_preserved_exactly(tmp_path):
    runs = tmp_path / "runs"
    ledger = gzip.compress(b'{"ranks":[1,2]}', mtime=12345)
    put(runs, "bm25_hashseed_ledgers.json.gz", ledger)
    put(runs, "reranker_prompt_audit.json", {"helper_sha256": "earlier-helper", "checks": [True]})
    put(runs, "reranker_prompt_audit_final.json", {"helper_sha256": "final-helper", "checks": [True]})
    first, second = tmp_path / "one", tmp_path / "two"
    manifest = history.archive_history(runs, first)
    assert history.archive_history(runs, first) == manifest
    assert history.archive_history(runs, second) == manifest
    assert {str(p.relative_to(first)): p.read_bytes() for p in first.rglob("*") if p.is_file()} == {
        str(p.relative_to(second)): p.read_bytes() for p in second.rglob("*") if p.is_file()}
    archived = (first / "artifacts/bm25_hashseed_ledgers.json.gz.gz").read_bytes()
    assert gzip.decompress(archived) == ledger
    assert gzip.decompress(gzip.decompress(archived)) == b'{"ranks":[1,2]}'
    assert len(manifest["groups"]) == 3
    for entry in manifest["artifacts"]:
        raw, packed = (runs / entry["source"]).read_bytes(), (first / entry["artifact"]).read_bytes()
        assert history.sha(raw) == entry["source_sha256"]
        assert history.sha(packed) == entry["artifact_sha256"]
    before = (first / "manifest.json").read_bytes()
    put(runs, "reranker_prompt_audit_final.json", {"helper_sha256": "different-helper"})
    with pytest.raises(ValueError, match="Existing archive differs"):
        history.archive_history(runs, first)
    assert (first / "manifest.json").read_bytes() == before


def test_size_and_path_bounds_fail_before_any_archive_is_written(tmp_path, monkeypatch):
    runs, output = tmp_path / "runs", tmp_path / "report"
    put(runs, "native-validation/process.log", b"too large")
    monkeypatch.setattr(history, "MAX_FILE_BYTES", 2)
    with pytest.raises(ValueError, match="size bound"):
        history.archive_history(runs, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="outside"):
        history.archive_history(runs, runs / "nested-report")
    monkeypatch.setattr(history, "MAX_FILE_BYTES", 1024)
    (runs / "native-validation/process.log").unlink()
    private = put(tmp_path, "outside-secret.json", {"not": "task evidence"})
    (runs / "native-validation/process.log").symlink_to(private)
    with pytest.raises(ValueError, match="symlink"):
        history.archive_history(runs, output)
    assert not output.exists()


def test_extension_adds_annotation_receipts_but_cannot_replace_history(tmp_path):
    runs, output = tmp_path / "runs", tmp_path / "report"
    put(runs, "bm25_hashseed_audit.json", {"retrieval_source_sha256": "original"})
    first = history.archive_history(runs, output)
    put(runs, "adaptive-initial-retrieval-audit.json", {"all_support_ids_at_top6": 11, "n": 96})
    second = history.archive_history(runs, output, extend=True)
    assert second["artifact_count"] == first["artifact_count"] + 1
    added = next(group for group in second["groups"] if group["source"].startswith("adaptive-initial"))
    assert added["role"] == "pre_inference_retrieval_annotation_diagnostic"
    assert not added["eligible_for_main_metrics"]
    put(runs, "bm25_hashseed_audit.json", {"retrieval_source_sha256": "changed"})
    with pytest.raises(ValueError, match="cannot change or remove"):
        history.archive_history(runs, output, extend=True)


def test_future_development_directories_extend_archive_with_only_explicit_logits(tmp_path):
    runs, output = tmp_path / "runs", tmp_path / "report"
    put(runs, "bm25_hashseed_audit.json", {"retrieval_source_sha256": "original"})
    initial = history.archive_history(runs, output)
    assert not set(history.OPTIONAL_DIRECTORIES) & set(initial["missing_expected_sources"])
    logits = b"PK\x03\x04small-npz-fixture"
    put(runs, "adaptive-preflight/logits.npz", logits)
    put(runs, "adaptive-preflight/preflight.json", {"status": "passed", "backend_identity": {"revision": "pin"}})
    put(runs, "adaptive-dev/guard-status.json", {"status": "complete"})
    put(runs, "adaptive-dev/logits.npz", b"not-allowlisted-here")
    put(runs, "context-pressure-dev/guard-status.json", {"status": "stopped", "reason": "interrupted"})
    put(runs, "adaptive-final/logits.npz", b"not-development")
    result = history.archive_history(runs, output, extend=True)
    groups = {group["source"]: group for group in result["groups"]}
    assert groups["adaptive-preflight"]["observed_status"] == "passed"
    assert groups["adaptive-preflight"]["source_identity_receipts"][0]["reported_identity"]["backend_identity"] == {"revision": "pin"}
    assert groups["adaptive-dev"]["observed_status"] == "complete"
    assert groups["context-pressure-dev"]["observed_status"] == "stopped"
    assert all(not group["eligible_for_main_metrics"] for group in groups.values())
    sources = {item["source"] for item in result["artifacts"]}
    assert "adaptive-dev/logits.npz" not in sources and "adaptive-final/logits.npz" not in sources
    assert gzip.decompress((output / "artifacts/adaptive-preflight/logits.npz.gz").read_bytes()) == logits
