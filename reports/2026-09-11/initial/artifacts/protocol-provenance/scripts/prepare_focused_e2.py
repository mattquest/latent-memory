#!/usr/bin/env python3
"""Partition the same first 16 E2 questions by available document count.

Selection reads only source order, question IDs and document counts. Answers,
predictions, scores and model outcomes never affect membership or hop budgets.
Each group retains the first original hop cap producing each distinct number
of executed evidence hops. This removes repeated exhausted-evidence schedules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VERSION = 1
ORIGINAL_HOPS = (1, 3, 5, 10, 20)
GROUP_SIZES = {6: 4, 7: 3, 10: 3, 20: 3, 40: 3}
ARMS_PER_HOP = 4  # JSON notes, BF16 relay, int8 relay, direct full text


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_hop_caps(document_count):
    if document_count <= 0:
        raise ValueError("Document count must be positive")
    maximum = (document_count + 1) // 2
    seen, result = set(), []
    for cap in ORIGINAL_HOPS:
        executed = min(cap, maximum)
        if executed not in seen:
            result.append(cap)
            seen.add(executed)
    return result


def write_verified(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Refusing to replace different existing grouped data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def prepare(source, output_dir):
    raw = source.read_bytes()
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) < 16:
        raise ValueError("The amendment requires the same first 16 source questions")
    selected = [(line, json.loads(line)) for line in lines[:16]]
    if len({row["id"] for _, row in selected}) != 16:
        raise ValueError("The first16 source IDs must be unique")
    groups = {}
    for source_index, (line, row) in enumerate(selected):
        count = len(row["evidence"])
        groups.setdefault(count, []).append((source_index, line, row["id"]))
    if {count: len(values) for count, values in groups.items()} != GROUP_SIZES:
        raise ValueError("Source structure differs from the predeclared 4/3/3/3/3 grouping")
    outputs = []
    for count, values in sorted(groups.items()):
        path = output_dir / f"focused_e2_docs{count}.jsonl"
        payload = b"".join(line + b"\n" for _, line, _ in values)
        hops = unique_hop_caps(count)
        write_verified(path, payload)
        outputs.append({"path": str(path), "sha256": sha(payload), "bytes": len(payload),
                        "document_count": count, "examples": len(values),
                        "source_indices_zero_based": [index for index, _, _ in values],
                        "question_ids": [identifier for _, _, identifier in values],
                        "nominal_hop_caps": hops,
                        "executed_evidence_hops": [min(hop, (count + 1) // 2) for hop in hops],
                        "planned_conditions": len(values) * len(hops) * ARMS_PER_HOP})
    manifest = {"version": VERSION, "generator": "scripts/prepare_focused_e2.py",
                "generator_sha256": sha(Path(__file__).read_bytes()),
                "source": {"path": str(source), "sha256": sha(raw), "rows": len(lines)},
                "selection": "First 16 source rows, then stable partition by evidence document count; no outcome-based selection.",
                "selected_question_ids_in_source_order": [row["id"] for _, row in selected],
                "original_hop_caps": list(ORIGINAL_HOPS), "evidence_documents_per_hop": 2,
                "arms_per_hop": ARMS_PER_HOP, "planned_conditions": sum(row["planned_conditions"] for row in outputs),
                "omitted_repeated_schedule_conditions": 16 * len(ORIGINAL_HOPS) * ARMS_PER_HOP - sum(row["planned_conditions"] for row in outputs),
                "outputs": outputs,
                "limits": ["Group-specific curves have small n=3 or4; they are mechanics diagnostics, not precise population estimates.",
                           "Pooled nominal-hop results have changing structural composition. Never draw a pooled hop-effect curve without that qualification.",
                           "At7 documents the nominal5-hop cap executes4 evidence hops; empty remaining hops are skipped by the frozen harness.",
                           "Original JSON records are preserved verbatim, including question and scoring annotations; only grouping changes."]}
    manifest_path = output_dir / "focused_e2_groups.manifest.json"
    write_verified(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/synthetic_multihop.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    result = prepare(args.source, args.output_dir)
    print(json.dumps({"planned_conditions": result["planned_conditions"], "groups": len(result["outputs"]),
                      "omitted_repeated_schedule_conditions": result["omitted_repeated_schedule_conditions"]}))


if __name__ == "__main__":
    main()
