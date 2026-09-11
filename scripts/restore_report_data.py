#!/usr/bin/env python3
"""Restore exact normalized inputs from initial, adaptive, or pressure reports.

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
import re

MAX_INPUT_BYTES = 128 * 1024**2
MAX_TOTAL_BYTES = 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
REPORT_INPUT_DIRECTORIES = {
    "artifact-report-v1": ("data-inputs",),
    "adaptive-artifact-report-v1": ("inputs", "provenance"),
    "pressure-artifact-report-v1": ("inputs",),
}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def restore(report_dir: Path, output_dir: Path) -> list[dict]:
    root = report_dir.resolve()
    target = output_dir.resolve()
    with (root / "artifact-manifest.json").open("rb") as stream:
        manifest_raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(manifest_raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Report manifest exceeds the size limit")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise ValueError("Report manifest must be an object")
    version = manifest.get("report_version")
    if not isinstance(version, str) or version not in REPORT_INPUT_DIRECTORIES:
        raise ValueError(f"Unsupported report version: {version}")
    if not isinstance(manifest.get("artifacts"), list):
        raise ValueError("Report artifacts must be a list")
    allowed = REPORT_INPUT_DIRECTORIES[version]
    pending, total = {}, 0
    for receipt in manifest["artifacts"]:
        if not isinstance(receipt, dict):
            raise ValueError("Artifact receipts must be objects")
        name = receipt.get("original_basename")
        if name is None:
            continue
        if (not isinstance(name, str) or not name.endswith(".jsonl") or
                Path(name).name != name or "/" in name or "\\" in name or "\0" in name):
            raise ValueError("Input names must be JSONL basenames")
        relative = receipt.get("artifact")
        if not isinstance(relative, str) or "\\" in relative or "\0" in relative:
            raise ValueError("Input archives require safe relative paths")
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 3 or
                path.parts[0] != "artifacts" or path.parts[1] not in allowed):
            raise ValueError("Input archive escapes the report data directory")
        # Reject symlinked files/directories even if they currently resolve inside
        # the report: a receipt names the actual archived file, not an alias.
        if any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            raise ValueError("Input archives cannot use symlink paths")
        artifact = (root / path).resolve()
        if not artifact.is_relative_to(root / "artifacts" / path.parts[1]):
            raise ValueError("Input archive escapes the report data directory")
        if receipt.get("compression") != "gzip" or path.suffix != ".gz":
            raise ValueError("Normalized input archives must use gzip")
        for key in ("artifact_bytes", "source_bytes"):
            if type(receipt.get(key)) is not int or not 0 <= receipt[key] <= MAX_INPUT_BYTES:
                raise ValueError(f"Invalid or oversized input byte count: {key}")
        for key in ("artifact_sha256", "source_sha256"):
            if not isinstance(receipt.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[key]):
                raise ValueError(f"Invalid input checksum: {key}")
        with artifact.open("rb") as stream:
            packed = stream.read(MAX_INPUT_BYTES + 1)
        if len(packed) > MAX_INPUT_BYTES:
            raise ValueError("Compressed input exceeds the size limit")
        if len(packed) != receipt["artifact_bytes"] or sha256(packed) != receipt["artifact_sha256"]:
            raise ValueError(f"Archive checksum or size mismatch: {name}")
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        total += len(raw)
        if len(raw) > MAX_INPUT_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("Uncompressed inputs exceed the size limit")
        if len(raw) != receipt["source_bytes"] or sha256(raw) != receipt["source_sha256"]:
            raise ValueError(f"Input checksum or size mismatch: {name}")
        if name in pending:
            if pending[name][1] != raw:
                raise ValueError(f"Conflicting archives for input basename: {name}")
            pending[name][3].append(relative)
            continue  # Adaptive dev inputs are also archived as data provenance.
        destination = target / name
        if destination.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {name}")
        exists = destination.exists()
        if exists:
            if not destination.is_file() or destination.stat().st_size != len(raw):
                raise ValueError(f"Existing input differs: {destination}; choose a fresh output directory")
            with destination.open("rb") as stream:
                identical = stream.read(len(raw) + 1) == raw
            if not identical:
                raise ValueError(f"Existing input differs: {destination}; choose a fresh output directory")
        pending[name] = (destination, raw, exists, [relative])
    if not pending:
        raise ValueError("Report contains no normalized input archives")
    # Verify every archive and destination before restoring any file.
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (destination, raw, exists, artifacts) in pending.items():
        if not exists:
            # Exclusive creation also protects files created since preflight.
            with destination.open("xb") as stream:
                stream.write(raw)
        results.append({"path": str(destination), "status": "reused" if exists else "restored",
                        "original_basename": name, "source_artifacts": artifacts,
                        "report_version": version, "bytes": len(raw), "sha256": sha256(raw)})
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
