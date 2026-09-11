#!/usr/bin/env python3
"""Download the pinned official BF16 reranker; no model import or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.reranker_mlx import MODEL_ID, REVISION, _verify_local_checkpoint

FILES = {"README.md", "chat_template.jinja", "config.json", "generation_config.json",
         "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
RESERVE_BYTES = 40 * 1024**3
MAX_DOWNLOAD_BYTES = 2 * 1024**3


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def download(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "download-manifest.json").exists():
        return _verify_local_checkpoint(output, REVISION)
    endpoint = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(endpoint, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("Official model API did not resolve the pinned revision")
    files = sorted((item for item in metadata["siblings"] if item["rfilename"] in FILES),
                   key=lambda item: item["rfilename"])
    if {item["rfilename"] for item in files} != FILES:
        raise ValueError("Official revision does not contain all expected checkpoint files")
    total = sum(item["size"] for item in files)
    if total >= MAX_DOWNLOAD_BYTES:
        raise ValueError("Reranker download exceeds the authorized 2 GiB bound")
    if shutil.disk_usage(output).free - total < RESERVE_BYTES:
        raise OSError("Reranker download would leave less than 40 GiB free disk")
    records = []
    for info in files:
        path = output / info["rfilename"]
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            received, digest = 0, hashlib.sha256()
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{info['rfilename']}"
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                while block := response.read(8 * 1024**2):
                    received += len(block)
                    if received > info["size"] or shutil.disk_usage(output).free - len(block) < RESERVE_BYTES:
                        raise OSError("Reranker download exceeded a declared size or free-disk bound")
                    stream.write(block)
                    digest.update(block)
            if received != info["size"]:
                raise ValueError(f"Incomplete official checkpoint file: {path.name}")
            if info.get("lfs") and digest.hexdigest() != info["lfs"]["sha256"]:
                raise ValueError(f"Official checkpoint checksum mismatch: {path.name}")
            partial.replace(path)
        digest = digest_file(path)
        if path.stat().st_size != info["size"]:
            raise ValueError(f"Existing file size differs from official revision: {path}")
        if info.get("lfs") and digest != info["lfs"]["sha256"]:
            raise ValueError(f"Existing file checksum differs from official revision: {path}")
        records.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest})
        print(f"Verified {path.name}: {path.stat().st_size} bytes", flush=True)
    manifest = {"repo_id": MODEL_ID, "revision": REVISION, "total_bytes": total, "files": records}
    temporary = output / "download-manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "download-manifest.json")
    return _verify_local_checkpoint(output, REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Qwen3-Reranker-0.6B"))
    parser.add_argument("--verify-only", action="store_true", help="Verify local hashes without network access")
    args = parser.parse_args()
    manifest = _verify_local_checkpoint(args.output, REVISION) if args.verify_only else download(args.output)
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION,
                      "total_bytes": manifest["total_bytes"], "manifest": str(args.output / "download-manifest.json")}))


if __name__ == "__main__":
    main()
