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
    (root / "artifact-manifest.json").write_text(json.dumps({"artifacts": [receipt]}))
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
    (report / "artifact-manifest.json").write_text(json.dumps({"artifacts": [receipt]}))
    with pytest.raises(ValueError):
        module.restore(report, data)
    assert not data.exists()
