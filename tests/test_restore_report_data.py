"""Archive restoration integrity and preservation of existing data."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("restore_report_data", Path(__file__).parents[1] / "scripts" / "restore_report_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def archive(root, name="input.jsonl", raw=b'{"fixture":true}\n'):
    packed = gzip.compress(raw, mtime=0)
    relative = "artifacts/data-inputs/fixture_" + name + ".gz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed)
    receipt = {"artifact": relative, "original_basename": name, "compression": "gzip",
               "artifact_bytes": len(packed), "artifact_sha256": module.sha256(packed),
               "source_bytes": len(raw), "source_sha256": module.sha256(raw)}
    (root / "artifact-manifest.json").write_text(json.dumps({"report_version": "artifact-report-v1", "artifacts": [receipt]}))
    return receipt


def test_restore_exact_bytes_and_reuse(tmp_path):
    report, data = tmp_path / "report", tmp_path / "data"
    archive(report)
    first = module.restore(report, data)
    assert first[0]["status"] == "restored"
    assert (data / "input.jsonl").read_bytes() == b'{"fixture":true}\n'
    assert module.restore(report, data)[0]["status"] == "reused"


def test_conflicting_existing_data_is_preserved(tmp_path):
    report, data = tmp_path / "report", tmp_path / "data"
    archive(report)
    data.mkdir()
    (data / "input.jsonl").write_bytes(b"existing\n")
    with pytest.raises(ValueError, match="Existing input differs"):
        module.restore(report, data)
    assert (data / "input.jsonl").read_bytes() == b"existing\n"


def test_corrupt_archive_does_not_create_output(tmp_path):
    report, data = tmp_path / "report", tmp_path / "data"
    receipt = archive(report)
    (report / receipt["artifact"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum or size mismatch"):
        module.restore(report, data)
    assert not data.exists()


@pytest.mark.parametrize("name,relative", [("../escape.jsonl", "artifacts/data-inputs/x.gz"),
                                          ("input.jsonl", "../escape.gz")])
def test_paths_cannot_escape_report_or_destination(tmp_path, name, relative):
    report, data = tmp_path / "report", tmp_path / "data"
    receipt = archive(report)
    receipt.update(original_basename=name, artifact=relative)
    (report / "artifact-manifest.json").write_text(json.dumps({"report_version": "artifact-report-v1", "artifacts": [receipt]}))
    with pytest.raises(ValueError):
        module.restore(report, data)
    assert not data.exists()


def write_manifest(root, receipts, version="adaptive-artifact-report-v1"):
    (root / "artifact-manifest.json").write_text(json.dumps({"report_version": version, "artifacts": receipts}))


@pytest.mark.parametrize("exporter_name", ["build_report", "build_adaptive_report", "build_pressure_report"])
def test_actual_exporter_receipts_restore_original_input_names(tmp_path, exporter_name):
    # Exercise the exporters' real serialization functions without model calls.
    import importlib
    exporter = importlib.import_module("scripts." + exporter_name)
    output, data = tmp_path / "report", tmp_path / "data"
    raw = b'{"id":"exact-normalized-fixture"}\n'
    source = Path("data/adaptive_musique_test.jsonl")
    if exporter_name == "build_report":
        receipt = exporter.archive_bytes(output, Path("data-inputs/pinned.jsonl.gz"), raw, source, True)
        receipt["original_basename"] = source.name
        version = exporter.REPORT_VERSION
    else:
        receipt = exporter.archive(output, Path("inputs/dataset.jsonl.gz"), raw, source,
                                   compress=True, original_basename=source.name)
        version = exporter.VERSION
    write_manifest(output, [receipt], version)
    restored = module.restore(output, data)
    assert restored[0]["original_basename"] == source.name
    assert restored[0]["report_version"] == version
    assert (data / source.name).read_bytes() == raw
    assert sorted(p.name for p in data.iterdir()) == [source.name]


def adaptive_receipt(root, relative, name="adaptive_musique_dev.jsonl", raw=b'{"id":"fixture"}\n'):
    from scripts.build_adaptive_report import archive as emit
    return emit(root, Path(relative), raw, "data/" + name, compress=True, original_basename=name)


def test_adaptive_dev_duplicate_is_verified_and_provenance_is_restored(tmp_path):
    root, data = tmp_path / "report", tmp_path / "data"
    receipts = [adaptive_receipt(root, "inputs/dataset.jsonl.gz"),
                adaptive_receipt(root, "provenance/adaptive_musique_dev.jsonl.gz"),
                adaptive_receipt(root, "provenance/adaptive_musique_corpus_provenance.jsonl.gz",
                                 "adaptive_musique_corpus_provenance.jsonl", b'{"source":"fixture"}\n')]
    write_manifest(root, receipts)
    result = module.restore(root, data)
    assert len(result) == 2 and len(result[0]["source_artifacts"]) == 2
    assert {row["original_basename"] for row in result} == {
        "adaptive_musique_dev.jsonl", "adaptive_musique_corpus_provenance.jsonl"}
    assert all(row["status"] == "reused" for row in module.restore(root, data))


@pytest.mark.parametrize("corruption", ["different_bytes", "bad_checksum"])
def test_conflicting_or_corrupt_duplicate_never_restores_any_file(tmp_path, corruption):
    root, data = tmp_path / "report", tmp_path / "data"
    receipts = [adaptive_receipt(root, "inputs/dataset.jsonl.gz"),
                adaptive_receipt(root, "provenance/adaptive_musique_dev.jsonl.gz",
                                 raw=b'{"different":true}\n')]
    if corruption == "bad_checksum":
        receipts[1]["artifact_sha256"] = "0" * 64
    write_manifest(root, receipts)
    with pytest.raises(ValueError, match="Conflicting archives|checksum or size mismatch"):
        module.restore(root, data)
    assert not data.exists()


def test_unpacked_hash_is_verified_independently(tmp_path):
    root, data = tmp_path / "report", tmp_path / "data"
    receipt = archive(root)
    receipt["source_sha256"] = "0" * 64
    write_manifest(root, [receipt], "artifact-report-v1")
    with pytest.raises(ValueError, match="Input checksum or size mismatch"):
        module.restore(root, data)
    assert not data.exists()


def test_bounded_decompression_rejects_dishonest_small_size(tmp_path, monkeypatch):
    root, data = tmp_path / "report", tmp_path / "data"
    receipt = archive(root, raw=b"x" * 1024)
    receipt["source_bytes"] = 128
    write_manifest(root, [receipt], "artifact-report-v1")
    monkeypatch.setattr(module, "MAX_INPUT_BYTES", 512)
    with pytest.raises(ValueError, match="Uncompressed inputs exceed"):
        module.restore(root, data)
    assert not data.exists()


def test_aggregate_decompression_limit_is_checked_before_writes(tmp_path, monkeypatch):
    root, data = tmp_path / "report", tmp_path / "data"
    receipts = [adaptive_receipt(root, "inputs/first.jsonl.gz", "first.jsonl", b"a" * 100),
                adaptive_receipt(root, "inputs/second.jsonl.gz", "second.jsonl", b"b" * 100)]
    write_manifest(root, receipts)
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 150)
    with pytest.raises(ValueError, match="Uncompressed inputs exceed"):
        module.restore(root, data)
    assert not data.exists()


@pytest.mark.parametrize("kind", ["archive_file", "archive_directory", "destination"])
def test_symlink_paths_cannot_redirect_read_or_write(tmp_path, kind):
    root, data = tmp_path / "report", tmp_path / "data"
    receipt = archive(root)
    if kind == "archive_file":
        source = root / receipt["artifact"]
        moved = source.with_name("actual.gz")
        source.rename(moved)
        source.symlink_to(moved)
    elif kind == "archive_directory":
        source = root / "artifacts/data-inputs"
        moved = tmp_path / "outside"
        source.rename(moved)
        source.symlink_to(moved, target_is_directory=True)
    else:
        data.mkdir()
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes(b"untouched")
        (data / "input.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        module.restore(root, data)
    if kind == "destination":
        assert outside.read_bytes() == b"untouched"
    else:
        assert not data.exists()


@pytest.mark.parametrize("version", ["pressure-artifact-report-v1", "artifact-report-v1", "unknown-version"])
def test_only_known_schema_input_directories_are_accepted(tmp_path, version):
    root, data = tmp_path / "report", tmp_path / "data"
    receipt = adaptive_receipt(root, "provenance/adaptive_musique_dev.jsonl.gz")
    write_manifest(root, [receipt], version)
    with pytest.raises(ValueError, match="Unsupported report version|escapes the report data directory"):
        module.restore(root, data)
    assert not data.exists()
