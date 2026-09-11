#!/usr/bin/env python3
"""Restore exact normalized inputs from a checked experiment report archive.

Existing identical files are reused. Conflicting files are never overwritten.
This restores data only; it does not load weights, run inference, or extract tar.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

MAX_INPUT_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 1024**3


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def restore(report_dir: Path, output_dir: Path) -> list[dict]:
    root = report_dir.resolve()
    target = output_dir.resolve()
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    pending, seen, total = [], set(), 0
    for receipt in manifest["artifacts"]:
        name = receipt.get("original_basename")
        if name is None:
            continue
        if (not isinstance(name, str) or not name.endswith(".jsonl") or
                Path(name).name != name or "/" in name or "\\" in name):
            raise ValueError("Input names must be JSONL basenames")
        if name in seen:
            raise ValueError(f"Duplicate input basename: {name}")
        seen.add(name)
        artifact = (root / receipt["artifact"]).resolve()
        if not artifact.is_relative_to(root / "artifacts" / "data-inputs"):
            raise ValueError("Input archive escapes the report data directory")
        if receipt.get("compression") != "gzip":
            raise ValueError("Normalized input archives must use gzip")
        if artifact.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("Compressed input exceeds the size limit")
        packed = artifact.read_bytes()
        if len(packed) != receipt["artifact_bytes"] or sha256(packed) != receipt["artifact_sha256"]:
            raise ValueError(f"Archive checksum or size mismatch: {name}")
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        total += len(raw)
        if len(raw) > MAX_INPUT_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("Uncompressed inputs exceed the size limit")
        if len(raw) != receipt["source_bytes"] or sha256(raw) != receipt["source_sha256"]:
            raise ValueError(f"Input checksum or size mismatch: {name}")
        destination = target / name
        if destination.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {name}")
        exists = destination.exists()
        if exists and (not destination.is_file() or destination.read_bytes() != raw):
            raise ValueError(f"Existing input differs: {destination}; choose a fresh output directory")
        pending.append((destination, raw, exists))
    if not pending:
        raise ValueError("Report contains no normalized input archives")
    # Verify every archive and destination before restoring any file.
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for destination, raw, exists in pending:
        if not exists:
            # Exclusive creation also protects files created since preflight.
            with destination.open("xb") as stream:
                stream.write(raw)
        results.append({"path": str(destination), "status": "reused" if exists else "restored",
                        "bytes": len(raw), "sha256": sha256(raw)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/2026-09-11"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for result in restore(args.report_dir, args.output_dir):
        print(json.dumps(result))


if __name__ == "__main__":
    main()
