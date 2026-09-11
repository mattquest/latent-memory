"""Release-integrity fixtures, not model experiment measurements."""
from dataclasses import asdict
import gzip
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("pressure_report", Path(__file__).parents[1] / "scripts/build_pressure_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = root / "run"
    run.mkdir()
    source = root / "source.py"
    source.write_text("# immutable executable fixture\n")
    question = {"id": "q1", "question": "Which answer?", "answers": ["answer"], "supporting_context_ids": ["d1"]}
    question_raw = (json.dumps(question) + "\n").encode()
    corpus_raw = b'{"id":"d1","title":"Title","text":"answer"}\n'
    config = asdict(report.PressureConfig("data/questions.jsonl", "data/corpus.jsonl", "original-run", "unused", expected_questions=1))
    for key in ("output_dir", "resume", "max_runtime_seconds"):
        config.pop(key)
    source_rows = []
    for arm, rounds in (("basic_rag", 1), ("iterative_text", 5)):
        source_rows.append({"id": "source-" + arm, "example_id": "q1", "status": "ok", "case": {"arm": arm, "max_rounds": rounds},
                            "question": question["question"], "answers": question["answers"], "prediction": "answer", "raw_output": "answer",
                            "scores": {"exact_match": 1., "f1": 1.}, "cold_end_to_end_ms": 30., "evidence_ids": ["d1"],
                            "trace": [{"event": "retrieval", "new_document_ids": ["d1"]}],
                            "counts": {"retrieved_evidence_tokens": 20}, "component_ms": {"controller_ms": 3., "retrieval_ms": 2.}})
    source_raw = ("\n".join(json.dumps(row) for row in source_rows) + "\n").encode()
    manifest = {"identity": {"protocol": "matched-trace-pressure-v1", "config": config,
                            "dataset_sha256": report.sha(question_raw), "corpus_sha256": report.sha(corpus_raw),
                            "source_results_sha256": report.sha(source_raw), "source_sha256": {"source.py": report.sha(source.read_bytes())}},
                "archives": []}
    for name, raw in (("questions.jsonl.gz", question_raw), ("corpus.jsonl.gz", corpus_raw), ("source-results.jsonl.gz", source_raw)):
        packed = gzip.compress(raw, mtime=0)
        (run / name).write_bytes(packed)
        manifest["archives"].append({"artifact": name, "artifact_sha256": report.sha(packed), "source_sha256": report.sha(raw)})
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "summary.json").write_text('{"status":"complete","stop_reason":"complete"}\n')
    rows = []
    for arm in report.ARMS:
        rows.append({"id": report.expected_id("matched-trace-pressure-v1", "q1", arm, config), "status": "ok", "example_id": "q1", "arm": arm,
                     "question": question["question"], "answers": question["answers"], "prediction": "answer", "raw_output": "answer",
                     "scores": {"exact_match": 1., "f1": 1.}, "replay_cold_ms": 15., "component_ms": {"final_generation_ms": 15.},
                     "source_controller_ms": 3., "source_retrieval_ms": 2., "replay_plus_source_controller_and_retrieval_ms": 20.,
                     "retained_document_ids": ["d1"], "retained_evidence_tokens": 20, "cumulative_source_tokens": 20,
                     "retained_evidence_token_ids_sha256": "matching-evidence-digest", "trace_overflow": False,
                     "retained_original_support_id_coverage": 1., "source_result_id": "source-iterative_text",
                     "source_trace_sha256": report.sha(json.dumps(source_rows[1]["trace"], sort_keys=True).encode()),
                     "model_calls": [{"evidence_input_tokens": 20, "total_context_reservation_tokens": 128}]})
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return root, run, rows


def write_rows(run, rows):
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_complete_export_preserves_exact_inputs_results_and_source(tmp_path):
    root, run, rows = fixture(tmp_path)
    assert report.inspect_run(run, root)["status"] == "complete"
    output = root / "release"
    summary = report.build(run, output, root)
    assert summary["report_status"] == "complete"
    receipts = json.loads((output / "artifact-manifest.json").read_text())["artifacts"]
    inputs = [row for row in receipts if "original_basename" in row]
    assert {row["original_basename"] for row in inputs} == {"questions.jsonl", "corpus.jsonl"}
    for row in inputs:
        assert report.sha(gzip.decompress((output / row["artifact"]).read_bytes())) == row["source_sha256"]
    raw_receipt = next(row for row in receipts if row["artifact"].endswith("run/results.jsonl.gz"))
    assert gzip.decompress((output / raw_receipt["artifact"]).read_bytes()) == (run / "results.jsonl").read_bytes()
    assert any(row["artifact"].endswith("source/source.py") for row in receipts)


def test_stale_score_and_wrong_question_fail_even_with_correct_case_id(tmp_path):
    root, run, rows = fixture(tmp_path)
    rows[0]["scores"]["exact_match"] = 0
    write_rows(run, rows)
    inspected = report.inspect_run(run, root)
    assert inspected["status"] != "complete"
    assert any(row["kind"] == "scores_differ_from_pinned_answers" for row in inspected["issues"])
    rows[0]["scores"]["exact_match"] = 1
    rows[0]["question"] = "Another question?"
    write_rows(run, rows)
    assert any(row["kind"] == "question_or_answers_differ_from_pinned_dataset" for row in report.inspect_run(run, root)["issues"])


def test_wrong_identity_trace_or_budget_prevents_complete_release(tmp_path):
    root, run, rows = fixture(tmp_path)
    rows[0]["id"] = "wrong-id"
    rows[1]["source_trace_sha256"] = "wrong-trace"
    rows[2]["model_calls"][0]["evidence_input_tokens"] = 2000
    write_rows(run, rows)
    inspected = report.inspect_run(run, root)
    kinds = {row["kind"] for row in inspected["issues"]}
    assert {"result_identity_mismatch", "source_controller_trace_mismatch", "evidence_or_context_budget_violation"} <= kinds
    assert inspected["status"] != "complete"


def test_changed_archive_source_and_torn_ledger_are_visible(tmp_path):
    root, run, rows = fixture(tmp_path)
    (root / "source.py").write_text("# changed implementation\n")
    (run / "questions.jsonl.gz").write_bytes(b"corrupted")
    with (run / "results.jsonl").open("ab") as handle:
        handle.write(b'{"id":')
    inspected = report.inspect_run(run, root)
    assert inspected["status"] != "complete"
    kinds = {row["kind"] for row in inspected["issues"]}
    assert {"source_verification_failed", "archive_verification_failed", "partial_trailing_record"} <= kinds
