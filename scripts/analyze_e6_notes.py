#!/usr/bin/env python3
"""Describe saved E6 note failures without loading models or changing prompts.

Exact record-ID/value presence is a lexical diagnostic, not semantic evaluation
or an intervention proving what caused the final answer. Gold values are read
only for this posthoc analysis of completed, immutable result files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def exact_token_present(text, token):
    if token is None:
        return None
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(token)) + r"(?![A-Za-z0-9_])",
                          text, re.IGNORECASE))


def note_receipt(row, example, relay_cap, answer_cap, source_line, line_hash):
    if row["question"] != example["question"] or row["answers"] != example["answers"]:
        raise ValueError("Saved condition does not match the exact dataset question/answers")
    targets = set(re.findall(r"\bK\d{6}\b", example["question"]))
    if len(targets) != 1:
        raise ValueError("E6 analysis expects exactly one synthetic record ID per question")
    target = next(iter(targets))
    metadata = example["metadata"]
    current, old = metadata.get("current_answer"), metadata.get("old_answer")
    conflicts = metadata.get("conflicting_values", [])
    notes = []
    for index, event in enumerate(row.get("trace", [])):
        if "relay_text" not in event:
            continue
        text, tokens = event["relay_text"], event["decoded_tokens"]
        if not isinstance(tokens, int) or not 0 <= tokens <= relay_cap:
            raise ValueError("Saved note token count lies outside the configured output cap")
        try:
            parsed = json.loads(text)
            valid_facts = isinstance(parsed, dict) and isinstance(parsed.get("facts"), list)
        except ValueError:
            valid_facts = False
        notes.append({"source_trace_index": index, "hop": event.get("hop"),
                      "evidence_ids": event.get("evidence_ids", []), "text": text,
                      "text_sha256": sha(text.encode()), "decoded_tokens": tokens,
                      "reached_configured_token_cap": tokens == relay_cap,
                      "target_id_present": exact_token_present(text, target),
                      "current_answer_present": exact_token_present(text, current),
                      "old_answer_present": exact_token_present(text, old),
                      "conflicting_values_present": [exact_token_present(text, value) for value in conflicts],
                      "valid_json_facts_object": valid_facts})
    if not notes:
        raise ValueError("Completed text condition contains no saved evidence note")
    evidence = {item["id"]: item for item in example["evidence"]}
    if not row["evidence_ids"] or any(key not in evidence for key in row["evidence_ids"]):
        raise ValueError("Saved selected evidence IDs differ from the exact input dataset")
    final = notes[-1]
    return {"result_id": row["id"], "source_result_line": source_line,
            "source_result_line_sha256_without_newline": line_hash,
            "example_id": example["id"], "category": example["category"],
            "update_policy": row["case"]["update_policy"], "case": row["case"],
            "question": example["question"], "answers": example["answers"], "target_record_id": target,
            "current_answer": current, "old_answer": old, "conflicting_values": conflicts,
            "selected_evidence_ids": row["evidence_ids"],
            "all_selected_evidence_contains_target_id": all(exact_token_present(evidence[key]["text"], target)
                                                              for key in row["evidence_ids"]),
            "final_note": final, "notes": notes,
            "any_note_contains_target_id": any(note["target_id_present"] for note in notes),
            "prediction": row["prediction"], "final_unknown": row["prediction"].strip().casefold() == "unknown",
            "recorded_exact_match": row["scores"]["exact_match"],
            "final_answer_tokens": len(row["output_token_ids"]), "final_answer_token_cap": answer_cap,
            "final_answer_generation_stop": row.get("generation_stop"), "relay_token_cap": relay_cap,
            "relay_stop_reason_recorded": False}


def summarize(receipts):
    groups = defaultdict(list)
    for receipt in receipts:
        groups[(receipt["category"], receipt["update_policy"])].append(receipt)
        groups[(receipt["category"], "both_policies")].append(receipt)
    result = []
    for (category, policy), rows in sorted(groups.items()):
        final_notes = [row["final_note"] for row in rows]
        lengths = [note["decoded_tokens"] for note in final_notes]
        applicable = [note for note in final_notes if note["current_answer_present"] is not None]
        conflict_notes = [note for note in final_notes if note["conflicting_values_present"]]
        result.append({"category": category, "update_policy": policy, "conditions": len(rows),
            "distinct_examples": len({row["example_id"] for row in rows}),
            "final_target_id_retained": sum(note["target_id_present"] for note in final_notes),
            "final_target_id_omitted": sum(not note["target_id_present"] for note in final_notes),
            "target_id_retained_in_any_note": sum(row["any_note_contains_target_id"] for row in rows),
            "current_answer_applicable": len(applicable),
            "final_current_answer_retained": sum(note["current_answer_present"] for note in applicable),
            "final_old_answer_retained": sum(note["old_answer_present"] is True for note in final_notes),
            "conflict_value_pairs_applicable": len(conflict_notes),
            "final_both_conflict_values_retained": sum(all(note["conflicting_values_present"]) for note in conflict_notes),
            "unknown": sum(row["final_unknown"] for row in rows),
            "correct": sum(row["recorded_exact_match"] for row in rows),
            "final_note_tokens_min": min(lengths), "final_note_tokens_median": statistics.median(lengths),
            "final_note_tokens_max": max(lengths),
            "final_note_at_cap": sum(note["reached_configured_token_cap"] for note in final_notes),
            "any_note_at_cap": sum(any(note["reached_configured_token_cap"] for note in row["notes"]) for row in rows),
            "final_valid_json_facts_objects": sum(note["valid_json_facts_object"] for note in final_notes),
            "final_answer_stop_reasons": dict(Counter(row["final_answer_generation_stop"] for row in rows)),
            "result_ids": sorted(row["result_id"] for row in rows)})
    return result


def analyze(run_dir, dataset=None):
    run_dir = Path(run_dir)
    manifest_raw, summary_raw, result_raw = ((run_dir / name).read_bytes()
                                            for name in ("manifest.json", "summary.json", "results.jsonl"))
    manifest, summary = json.loads(manifest_raw), json.loads(summary_raw)
    identity = manifest["identity"]
    config = identity["config"]
    dataset_path = Path(dataset or config["dataset"])
    dataset_raw = dataset_path.read_bytes()
    if sha(dataset_raw) != identity["dataset_sha256"]:
        raise ValueError("Dataset SHA-256 differs from the executed run manifest")
    examples = [json.loads(line) for line in dataset_raw.splitlines() if line.strip()]
    by_id = {example["id"]: example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError("Dataset contains duplicate example IDs")
    latest = {}
    for index, line in enumerate(result_raw.splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            latest[row["id"]] = (row, index, sha(line))
    if (summary.get("stop_reason") != "complete" or summary.get("n_failed_runs", 0) or
        len(latest) != summary["planned_runs"] or any(row["status"] != "ok" for row, _, _ in latest.values())):
        raise ValueError("E6 note analysis requires a complete successful per-job run")
    receipts, arm_counts = [], defaultdict(list)
    for row, index, line_hash in latest.values():
        case = row["case"]
        if case["experiment"] != "E6" or case.get("repeat", 0) != 0:
            continue
        if row["example_id"] not in by_id:
            raise ValueError("Saved result example is absent from the exact dataset")
        arm_counts[(case["arm"], case["precision"], case["update_policy"], row["category"])].append(row)
        if case["arm"] == "text":
            receipts.append(note_receipt(row, by_id[row["example_id"]], config["max_relay_tokens"],
                                         config["max_answer_tokens"], index, line_hash))
    if not receipts:
        raise ValueError("No E6 text conditions found")
    receipts.sort(key=lambda row: (row["example_id"], row["update_policy"], row["result_id"]))
    analysis = {"analysis_version": "e6-saved-note-diagnostic-v1", "scope": "posthoc_descriptive_not_causal",
        "source": {"run_directory": str(run_dir), "dataset": str(dataset_path),
                   "manifest_sha256": sha(manifest_raw), "summary_sha256": sha(summary_raw),
                   "results_sha256": sha(result_raw), "dataset_sha256": sha(dataset_raw),
                   "executed_identity": identity, "analysis_source_sha256": sha(Path(__file__).read_bytes())},
        "successful_source_conditions": len(latest), "text_conditions": len(receipts),
        "distinct_text_examples": len({row["example_id"] for row in receipts}),
        "relay_token_cap": config["max_relay_tokens"], "strata": summarize(receipts),
        "all_arm_accuracy_context": [{"arm": key[0], "precision": key[1], "update_policy": key[2],
            "category": key[3], "conditions": len(rows), "correct": sum(row["scores"]["exact_match"] for row in rows)}
            for key, rows in sorted(arm_counts.items())],
        "limitations": ["Record IDs and values are exact case-insensitive lexical matches with alphanumeric boundaries; presence does not establish semantic association.",
            "The final note is the last generated relay_text event, skipping later empty-hop trace records.",
            "Relay stop reasons were not saved. A note shorter than its configured cap did not reach that cap; no more specific stop reason is inferred.",
            "All/current conditions for the same question are repeated policy conditions, not independent examples.",
            "Omitted IDs and UNKNOWN outputs co-occur; this saved-output analysis does not prove that omission alone caused failure.",
            "No notes were repaired, no prompts were tuned, and no model calls were made in this analysis.",
            "These results describe this particular evidence-note baseline; they do not establish a fundamental latent-memory advantage for updates."]}
    return analysis, receipts


def write_report(analysis, receipts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipts).encode()
    analysis["example_receipts"] = {"path": "e6-note-examples.jsonl", "sha256": sha(receipt_raw), "conditions": len(receipts)}
    (output_dir / "e6-note-examples.jsonl").write_bytes(receipt_raw)
    (output_dir / "e6-note-analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    lines = ["# Saved E6 note diagnostic", "", "Posthoc and descriptive; no causal intervention or generation change.", "",
             f"The completed source contains {analysis['successful_source_conditions']} conditions; this note analysis covers {analysis['text_conditions']} text conditions on {analysis['distinct_text_examples']} questions.", "",
             "| Stratum | Policy | Conditions | ID retained / omitted | Current number retained | UNKNOWN | Correct | Final note tokens min–max | Notes at cap |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in analysis["strata"]:
        current = f"{row['final_current_answer_retained']}/{row['current_answer_applicable']}" if row["current_answer_applicable"] else "Not applicable"
        lines.append(f"| {row['category']} | {row['update_policy']} | {row['conditions']} | "
            f"{row['final_target_id_retained']} / {row['final_target_id_omitted']} | {current} | {row['unknown']} | "
            f"{row['correct']:g} | {row['final_note_tokens_min']}–{row['final_note_tokens_max']} | {row['final_note_at_cap']} |")
    lines += ["", f"Configured relay cap: {analysis['relay_token_cap']} tokens. Conflict-value retention and per-arm accuracy context are in the JSON.", "",
              "[Analysis and source hashes](e6-note-analysis.json) · [Every text condition and note](e6-note-examples.jsonl)", "",
              *["- " + note for note in analysis["limitations"]], ""]
    (output_dir / "e6-note-analysis.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/overnight/updates"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.run_dir.resolve()):
        parser.error("Write posthoc analysis outside the immutable source run")
    result, receipts = analyze(args.run_dir, args.dataset)
    write_report(result, receipts, args.output_dir)
    print(json.dumps({"text_conditions": len(receipts), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
