"""Integrity checks at the checkpoint-to-execution boundary."""
import hashlib
import json

import pytest

from scripts import run_large_controller as runner


def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    directory = tmp_path / 'model'
    directory.mkdir()
    weights = b'fixture weights'
    (directory / 'model.safetensors').write_bytes(weights)
    (directory / 'download-manifest.json').write_text(json.dumps({
        'download_complete': True, 'revision': 'pinned', 'repo_id': 'fixture/local',
        'files': [{'path': 'model.safetensors', 'size_bytes': len(weights),
                   'sha256': hashlib.sha256(weights).hexdigest()}]}))
    return directory, {'model_path': 'model', 'model_revision': 'pinned'}


def test_receipt_rehashes_weights_and_rejects_same_size_corruption(tmp_path, monkeypatch):
    directory, job = checkpoint(tmp_path, monkeypatch)
    receipt = runner.checkpoint_receipt(job)
    assert receipt['verified_bytes'] == len(b'fixture weights')
    (directory / 'model.safetensors').write_bytes(b'corrupt weights')
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.checkpoint_receipt(job)


def test_receipt_rejects_unrecorded_weights_and_wrong_revision(tmp_path, monkeypatch):
    directory, job = checkpoint(tmp_path, monkeypatch)
    (directory / 'extra.safetensors').write_bytes(b'extra')
    with pytest.raises(ValueError, match='Unrecorded'):
        runner.checkpoint_receipt(job)
    with pytest.raises(ValueError, match='different revision'):
        runner.checkpoint_receipt({**job, 'model_revision': 'different'})
