#!/usr/bin/env python3
"""Download and hash-verify a pinned MLX-community checkpoint with bounded I/O."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('HF_XET_NUM_CONCURRENT_RANGE_GETS', '4')
from huggingface_hub import HfApi, snapshot_download

MODEL_ID = 'mlx-community/Qwen3.5-122B-A10B-5bit'
REVISION = '958b33bf6418f8c462a5cbec59366ef763cc3069'
MAX_BYTES = 100 * 1024**3
RESERVE_BYTES = 40 * 1024**3


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download(output, metadata_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Checkpoint revision did not resolve exactly')
    files = sorted((f for f in info.siblings if f.rfilename.endswith(
        ('.safetensors', '.json', '.txt', '.jinja')) or f.rfilename in {'README.md', 'LICENSE'}), key=lambda f:f.rfilename)
    if any(Path(f.rfilename).name != f.rfilename for f in files):
        raise ValueError('Unexpected nested checkpoint file')
    total = sum(f.size or 0 for f in files)
    if not files or any(not f.size for f in files) or total > MAX_BYTES:
        raise ValueError('Missing sizes or checkpoint exceeds100GiB limit')
    chosen = [f for f in files if not metadata_only or not f.rfilename.endswith('.safetensors')]
    remaining = sum(f.size for f in chosen if not (output/f.rfilename).exists() or (output/f.rfilename).stat().st_size != f.size)
    if shutil.disk_usage(output).free < remaining + RESERVE_BYTES:
        raise OSError('Download would leave less than40GiB free disk')
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=output,
                      allow_patterns=[f.rfilename for f in chosen], max_workers=2)
    config = json.loads((output/'config.json').read_text())
    if config.get('model_file') or config.get('model_type') != 'qwen3_5_moe':
        raise ValueError('Unexpected model architecture or custom model code')
    if config.get('quantization',{}).get('bits') != 5:
        raise ValueError('Expected5-bit checkpoint')
    records=[]
    for f in chosen:
        path=output/f.rfilename
        if path.is_symlink() or path.stat().st_size != f.size:
            raise ValueError('Checkpoint geometry mismatch: '+f.rfilename)
        digest=digest_file(path)
        if f.lfs and digest != f.lfs.sha256:
            raise ValueError('Publisher SHA-256 mismatch: '+f.rfilename)
        if not f.lfs and f.blob_id:
            raw=path.read_bytes()
            if hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=f.blob_id:
                raise ValueError('Publisher Git blob mismatch: '+f.rfilename)
        records.append({'path':f.rfilename,'size_bytes':f.size,'sha256':digest,
                        'publisher_sha256':f.lfs.sha256 if f.lfs else None,'publisher_git_blob':f.blob_id})
        print(json.dumps({'verified_file':f.rfilename,'bytes':f.size}),flush=True)
    manifest={'repo_id':MODEL_ID,'revision':REVISION,'base_model':'Qwen/Qwen3.5-122B-A10B',
              'provenance_note':'Pinned community5-bit conversion; publisherfilehashverified, not a claimof lossless equivalencetoofficialBF16weights.',
              'license':'Apache-2.0','quantization':config['quantization'],'download_complete':not metadata_only,
              'total_checkpoint_bytes':total,'verified_bytes':sum(f['size_bytes'] for f in records),'files':records,
              'max_workers':2,'minimum_disk_reserve_bytes':RESERVE_BYTES}
    name='metadata-manifest.json' if metadata_only else 'download-manifest.json'
    temporary=output/(name+'.part');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(output/name)
    print(json.dumps({'status':'complete','metadata_only':metadata_only,'verified_bytes':manifest['verified_bytes']}),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('models/Qwen3.5-122B-A10B-5bit'))
    parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args()
    download(args.output,args.metadata_only)
