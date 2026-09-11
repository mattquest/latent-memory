#!/usr/bin/env python3
"""Fetch and verify pinned official Qwen3-14B BF16 files with bounded workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS', '4')
from huggingface_hub import HfApi, snapshot_download

MODEL_ID = 'Qwen/Qwen3-14B'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
MAX_BYTES = 32 * 1024**3
RESERVE_BYTES = 40 * 1024**3


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def download(output, metadata_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Official API did not resolve the pinned revision')
    files = sorted((entry for entry in info.siblings if entry.rfilename.endswith(
        ('.safetensors', '.json', '.txt', '.jinja')) or entry.rfilename in {'README.md', 'LICENSE'}),
        key=lambda entry: entry.rfilename)
    total = sum(entry.size or 0 for entry in files)
    if not files or any(not entry.size for entry in files) or total > MAX_BYTES:
        raise ValueError('Invalid file metadata or checkpoint exceeds the 32 GiB download bound')
    if any(Path(entry.rfilename).name != entry.rfilename for entry in files):
        raise ValueError('Unexpected nested checkpoint path')
    selected = [entry for entry in files if not metadata_only or not entry.rfilename.endswith('.safetensors')]
    remaining = sum(entry.size for entry in selected if not (output / entry.rfilename).exists())
    if shutil.disk_usage(output).free < remaining + RESERVE_BYTES:
        raise OSError('Checkpoint would leave less than 40 GiB of free disk')
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=output,
                      allow_patterns=[entry.rfilename for entry in selected], max_workers=2)
    records = []
    for entry in selected:
        path = output / entry.rfilename
        if path.is_symlink() or path.stat().st_size != entry.size:
            raise ValueError(f'Unexpected checkpoint file geometry: {entry.rfilename}')
        digest = digest_file(path)
        if entry.lfs and digest != entry.lfs.sha256:
            raise ValueError(f'Official checkpoint checksum mismatch: {entry.rfilename}')
        if not entry.lfs and entry.blob_id:
            raw = path.read_bytes()
            git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
            if git_blob != entry.blob_id:
                raise ValueError(f'Official metadata blob mismatch: {entry.rfilename}')
        records.append({'path': entry.rfilename, 'size_bytes': entry.size, 'sha256': digest,
                        'official_sha256': entry.lfs.sha256 if entry.lfs else None,
                        'official_git_blob': entry.blob_id})
        print(f'Verified {entry.rfilename}: {entry.size} bytes', flush=True)
    manifest = {'repo_id': MODEL_ID, 'revision': REVISION, 'license': 'Apache-2.0',
                'weights_dtype': 'bfloat16', 'download_complete': not metadata_only,
                'total_checkpoint_bytes': total, 'verified_bytes': sum(row['size_bytes'] for row in records),
                'files': records, 'max_workers': 2, 'minimum_disk_reserve_bytes': RESERVE_BYTES}
    name = 'metadata-manifest.json' if metadata_only else 'download-manifest.json'
    temporary = output / (name + '.part')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(output / name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/Qwen3-14B'))
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    manifest = download(args.output, args.metadata_only)
    print(json.dumps({key: manifest[key] for key in ('repo_id', 'revision', 'download_complete',
                                                   'total_checkpoint_bytes', 'verified_bytes')}))


if __name__ == '__main__':
    main()
