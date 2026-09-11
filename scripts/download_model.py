#!/usr/bin/env python3
"""Fetch the exact official BF16 model used by the experiments (about 16.4 GB)."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
from huggingface_hub import HfApi, snapshot_download

REPO = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Qwen3-8B"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    allowed = [s for s in info.siblings if s.rfilename.endswith(
        (".safetensors", ".json", ".txt", ".jinja"))]
    total = sum(s.size or 0 for s in allowed)
    if total > 20 * 1024**3:
        raise RuntimeError("Model exceeds the 20 GiB download allowance")
    if shutil.disk_usage(args.output).free < total + 40 * 1024**3:
        raise RuntimeError("Insufficient disk space with a 40 GiB reserve")
    manifest = {"repo_id": REPO, "revision": info.sha, "license": "Apache-2.0",
                "weights_dtype": "bfloat16", "download_bytes": total,
                "files": [{"name": s.rfilename, "bytes": s.size,
                           "sha256": s.lfs.sha256 if s.lfs else None}
                          for s in allowed]}
    snapshot_download(REPO, revision=REVISION, local_dir=args.output,
                      allow_patterns=[s.rfilename for s in allowed], max_workers=2)
    (args.output.parent / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
