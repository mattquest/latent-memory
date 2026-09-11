#!/usr/bin/env python3
"""Archive validation, development, and excluded test receipts apart from metrics.

Standard library only: never imports a model, experiment harness, or dataset.
Exact source bytes are wrapped in deterministic gzip, including already-gzipped
audit ledgers (whose artifact therefore ends in .gz.gz). Original failures and
partial JSONL tails remain untouched. Source identities are copied from receipts;
current code hashes are never substituted for missing historical identities.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re


DIRECTORIES = {
    "native-validation": ("numerical_original_failed_threshold",
        "Original absolute-error criterion; failure retained, not counted as a passing validation."),
    "native-validation-relative": ("numerical_revised_relative_threshold",
        "Revised development criterion after inspecting scale and the unshifted control; small mechanics check only."),
    "official-preflight": ("official_checkpoint_smoke_check", "Two toy inference checks; not benchmark accuracy."),
    "pilot-private": ("private_evidence_development_pilot", "Private-fact development conditions; excluded from final test metrics."),
    "development": ("operator_stopped_earlier_development",
        "Earlier development run intentionally stopped, as documented in runs/SESSION.md; original guard status is authoritative."),
    "development-v2": ("revised_development_matrix", "Revised development conditions before the main test; excluded from final metrics."),
    "judge-pilot": ("same_model_rubric_judge_pilot", "Development-only local same-model judging; not official BEAM scores."),
}
OPTIONAL_DIRECTORIES = {
    "adaptive-preflight": ("adaptive_real_weight_mechanics_preflight",
        "Fixed synthetic mechanics trace and toy reranker check; raw logits are diagnostic, not benchmark results."),
    "adaptive-dev": ("adaptive_retrieval_development_run",
        "Adaptive development subset before the final test; all results excluded from final metrics."),
    "context-pressure-dev": ("context_pressure_development_run",
        "Matched-trace retention development check; excluded from final pressure metrics."),
}
OPTIONAL_FILES = {
    "backend-report-notes.md": ("historical_methods_and_numerical_notes",
        "Historical methods notes may predate later adaptive/reranker additions; not final results."),
    "bm25_hashseed_audit.json": ("hashseed_reproducibility_audit", "Diagnostic rankings and floating-point variation; not scored model inference."),
    "bm25_hashseed_ledgers.json.gz": ("hashseed_reproducibility_raw_ledger", "Exact original compressed ledger is preserved inside an outer gzip."),
    "adaptive-data-validation.json": ("data_reproduction_audit", "Input reproduction/provenance check; not model performance."),
    "overnight/amendment.json": ("operator_budget_amendment_receipt",
        "Operator-approved scheduling/budget amendment; not a model result and not a development condition."),
}
ALLOWED_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ".gz"}
MAX_FILE_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
IDENTITY_FIELDS = {"identity", "backend", "model", "model_revision", "protocol_version",
                   "backend_identity", "reranker_identity", "judge_version", "source_code_sha256",
                   "source_sha256", "output_sha256"}
STATUS_FIELDS = {"status", "reason", "returncode", "stop_reason", "planned_runs", "planned",
                 "completed_unique_runs", "completed_runs", "n_successful_runs", "n_failed_runs",
                 "successful_unique_runs", "n_valid", "n_invalid"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def gzip_bytes(raw):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as archive:
        archive.write(raw)
    return buffer.getvalue()


def log_objects(raw):
    """Read leading or line-delimited JSON objects amid ordinary log text."""
    text, position, decoder = raw.decode(errors="replace"), 0, json.JSONDecoder()
    while (position := text.find("{", position)) >= 0:
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            position += 1
            continue
        position = end
        if isinstance(value, dict):
            yield value


def extract_identity(value):
    result = {key: value[key] for key in sorted(value)
              if key in IDENTITY_FIELDS or key.endswith("_sha256")}
    machine = value.get("machine")
    if isinstance(machine, dict):
        result["machine_recorded_source"] = {key: machine[key] for key in
            ("git_head", "git_dirty", "code_sha256") if key in machine}
    return result


def describe_group(name, role, note, files):
    statuses, identities = [], []
    authoritative_status = None
    status_source = None
    for relative, raw in files:
        values = []
        if relative.suffix == ".json":
            try:
                value = json.loads(raw)
                values = [value] if isinstance(value, dict) else []
            except (ValueError, UnicodeDecodeError):
                pass  # Preserve malformed receipts unchanged, without inventing metadata.
        elif relative.suffix == ".log":
            values = list(log_objects(raw))
        for value in values:
            identity = extract_identity(value)
            if identity:
                identities.append({"receipt": str(relative), "reported_identity": identity})
            if relative.name in {"guard-status.json", "summary.json", "matrix-status.json", "preflight.json"}:
                observed = {key: value[key] for key in sorted(value) if key in STATUS_FIELDS}
                if "jobs" in value:
                    observed["jobs"] = value["jobs"]
                statuses.append({"receipt": str(relative), "reported_status": observed})
                if relative.name == "guard-status.json" and relative.parent == Path(name):
                    authoritative_status = value.get("status", "not_recorded")
                    status_source = str(relative)
                elif relative.name == "preflight.json" and relative.parent == Path(name) and authoritative_status is None:
                    authoritative_status = value.get("status", "not_recorded")
                    status_source = str(relative)
                elif relative.name == "summary.json" and relative.parent == Path(name) and authoritative_status is None:
                    authoritative_status = value.get("status") or value.get("stop_reason")
                    status_source = str(relative) if authoritative_status else None
        if relative.name == "backend-report-notes.md":
            declared = dict(re.findall(r"^- `([^`]+)`: `([0-9a-f]{64})`", raw.decode(errors="replace"), re.MULTILINE))
            if declared:
                identities.append({"receipt": str(relative), "reported_identity": {"note_declared_source_hashes": declared}})
    if authoritative_status is None and role == "unused_partial_test_run_after_budget_amendment":
        authoritative_status = "incomplete_operator_identified"
        status_source = "Explicit interrupted-test directory classification; no root status receipt present"
    return {"source": name, "role": role, "note": note,
            "observed_status": authoritative_status or ("document_only" if len(files) == 1 else "not_recorded"),
            "status_source": status_source,
            "eligible_for_main_metrics": False, "status_receipts": statuses, "source_identity_receipts": identities,
            "identity_caveat": "Only identities present in historical receipts are reported; a command path is not an executed-source hash.",
            "artifacts": []}


def archive_history(runs_dir, output_dir, *, extend=False):
    runs_dir, output_dir = Path(runs_dir).resolve(), Path(output_dir).resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(runs_dir)
    if output_dir == runs_dir or output_dir.is_relative_to(runs_dir):
        raise ValueError("Validation archive must be outside its source runs directory")
    selections, missing, excluded = [], [], []
    directories = {**DIRECTORIES, **OPTIONAL_DIRECTORIES}
    for path in sorted(runs_dir.glob("original-interrupted-multihop*")):
        directories[path.name] = ("unused_partial_test_run_after_budget_amendment",
            "Original test condition interrupted after the operator budget amendment. It is not development/tuning data and is excluded from final metrics.")
    for name, (role, note) in directories.items():
        directory = runs_dir / name
        if not directory.exists():
            if name in DIRECTORIES:
                missing.append(name)
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Expected a real validation directory: {name}")
        paths = []
        for path in sorted(directory.rglob("*")):
            allowed = path.suffix in ALLOWED_SUFFIXES or (
                name == "adaptive-preflight" and path.name == "logits.npz")
            if not allowed:
                if path.is_file():
                    excluded.append(str(path.relative_to(runs_dir)))
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing a symlink in validation receipts: {path}")
            if path.is_file():
                paths.append(path)
        selections.append((name, role, note, paths))
    optional = dict(OPTIONAL_FILES)
    for path in sorted(runs_dir.glob("reranker_prompt_audit*.json")):
        optional[path.name] = ("reranker_tokenizer_prompt_audit",
            "CPU prompt equivalence only, bound to the helper hash recorded in this receipt; no model-quality claim.")
    for path in sorted(runs_dir.glob("adaptive-initial-retrieval-audit*.json")):
        optional[path.name] = ("pre_inference_retrieval_annotation_diagnostic",
            "Pre-inference supporting-ID coverage only; not measured model accuracy or proof retained text contains the answer.")
    for name, (role, note) in sorted(optional.items()):
        path = runs_dir / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Expected a real validation receipt: {name}")
            selections.append((name, role, note, [path]))
    artifacts, groups, payloads, total = [], [], {}, 0
    for name, role, note, paths in selections:
        files = []
        for path in paths:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
                raise ValueError("Validation archive size bound exceeded; model/data payloads are not allowed")
            raw = path.read_bytes()
            if len(raw) != size:
                raise ValueError(f"Validation receipt changed while being archived: {path}")
            total += size
            files.append((path.relative_to(runs_dir), raw))
        group = describe_group(name, role, note, files)
        for relative, raw in files:
            target = "artifacts/" + str(relative) + ".gz"
            compressed = gzip_bytes(raw)
            payloads[target] = compressed
            record = {"source": str(relative), "artifact": target,
                      "source_size_bytes": len(raw), "source_sha256": sha(raw),
                      "artifact_size_bytes": len(compressed), "artifact_sha256": sha(compressed),
                      "encoding": "gzip_of_exact_source_bytes", "role": role,
                      "observed_group_status": group["observed_status"], "eligible_for_main_metrics": False}
            artifacts.append(record)
            group["artifacts"].append(target)
        groups.append(group)
    if not artifacts:
        raise ValueError("No validation-history receipts found")
    manifest = {"schema_version": "validation-history-v1", "archive_scope": "validation_development_and_excluded_test_history",
                "eligible_for_main_metrics": False, "archive_status": "complete_for_discovered_sources",
                "missing_expected_sources": missing, "excluded_nonreceipt_files": excluded,
                "source_bytes": total, "artifact_count": len(artifacts), "groups": groups, "artifacts": artifacts,
                "archiver_source_sha256": sha(Path(__file__).read_bytes()),
                "limitations": ["Archive completeness does not mean every validation passed; original guard statuses are retained.",
                    "Failed/stopped runs, prior protocols, and budget-excluded partial test runs are never pooled into main benchmark metrics.",
                    "Historical source hashes may be absent; current code identity is not substituted.",
                    "Raw receipts are exact snapshots, including malformed or partial trailing records.",
                    "No model weights, full input corpora, private laptop files, or unrelated run directories are selected."]}
    payloads["manifest.json"] = json_bytes(manifest)
    lines = ["# Validation and excluded run history", "",
             "These receipts are excluded from the main benchmark metrics. Archive completeness does not imply all checks passed.", "",
             "Interrupted original test conditions retain their test-run role; they are not relabeled as development data.", "",
             "| Source | Role | Recorded status | Files |", "| --- | --- | --- | --- |"]
    lines += [f"| {group['source']} | {group['role']} | {group['observed_status']} | {len(group['artifacts'])} |" for group in groups]
    lines += ["", "See [manifest.json](manifest.json) for source identities, original statuses, and both source/artifact SHA-256 hashes.", "",
              "Each artifact is gzip of the exact original file bytes. Decompress once to recover that source file; an original .gz ledger therefore has an outer .gz.gz wrapper.", "",
              "The original numerical run failed its absolute-error threshold. The relative-threshold run is a later development validation, not a prespecified benchmark criterion.", ""]
    if missing:
        lines += ["Expected sources absent from this snapshot: " + ", ".join(missing) + ".", ""]
    payloads["README.md"] = "\n".join(lines).encode()
    # Extension adds receipts without revising or discarding already archived bytes.
    if extend and (output_dir / "manifest.json").exists():
        previous = json.loads((output_dir / "manifest.json").read_bytes())
        if previous.get("schema_version") != manifest["schema_version"]:
            raise ValueError("Cannot extend a different validation archive schema")
        current = {record["artifact"]: record for record in artifacts}
        for record in previous["artifacts"]:
            match = current.get(record["artifact"])
            if match is None or any(record[key] != match[key] for key in ("source_sha256", "artifact_sha256")):
                raise ValueError("Archive extension cannot change or remove an existing receipt")
    # Validate all collisions before writing anything; reruns are idempotent.
    for name, raw in payloads.items():
        target = output_dir / name
        if target.exists() and (not target.is_file() or target.read_bytes() != raw):
            if extend and name in {"manifest.json", "README.md"} and target.is_file():
                continue
            raise ValueError(f"Existing archive differs; choose a fresh output directory: {target}")
    for name, raw in payloads.items():
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or (extend and name in {"manifest.json", "README.md"}):
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(target)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="For example reports/2026-09-11/validation-history")
    parser.add_argument("--extend", action="store_true",
                        help="Add newly discovered receipts while refusing changes/removal of archived source bytes")
    args = parser.parse_args()
    result = archive_history(args.runs_dir, args.output_dir, extend=args.extend)
    print(json.dumps({"output_dir": str(args.output_dir), "artifacts": result["artifact_count"],
                      "source_bytes": result["source_bytes"], "missing_sources": result["missing_expected_sources"]}))


if __name__ == "__main__":
    main()
