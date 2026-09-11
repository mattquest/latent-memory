"""Artifact-integrity tests; fixtures are not model evaluation results."""
import importlib.util
import json
import gzip
from pathlib import Path

spec = importlib.util.spec_from_file_location("adaptive_report", Path(__file__).parents[1] / "scripts/build_adaptive_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    questions = root / "questions.jsonl"
    corpus = root / "corpus.jsonl"
    source = root / "source.py"
    questions.write_text('{"id":"q1","question":"Where?","answers":["FOUND"]}\n')
    corpus.write_text('{"id":"d1","title":"T","text":"Text"}\n')
    source.write_text("# fixture\n")
    run = root / "run"
    run.mkdir()
    identity = {"protocol": "fixture-v1", "config": {"dataset": "questions.jsonl", "corpus": "corpus.jsonl",
                "arms": ["basic_rag"], "rounds": [5], "limit": 1},
                "questions_sha256": report.sha(questions.read_bytes()), "corpus_sha256": report.sha(corpus.read_bytes()),
                "source_sha256": {"source.py": report.sha(source.read_bytes())}}
    case = {"arm": "basic_rag", "max_rounds": 1}
    identifier = report.sha(json.dumps(["fixture-v1", "q1", case], sort_keys=True).encode())[:24]
    (run / "manifest.json").write_text(json.dumps({"identity": identity}))
    (run / "summary.json").write_text('{"stop_reason":"complete"}')
    record = {"id": identifier, "status": "ok", "example_id": "q1", "case": case,
              "question": "Where?", "answers": ["FOUND"], "prediction": "FOUND",
              "scores": {"exact_match": 1, "f1": 1}, "cold_end_to_end_ms": 10,
              "component_ms": {}, "counts": {"retrieved_documents": 1, "host_output_tokens": 1,
                                               "internal_decoded_tokens": 0}, "executed_retrieval_rounds": 1,
              "invalid_actions": 0, "generation_stop": "eos", "support_annotation_coverage": None,
              "stop_reason": "single_round"}
    (run / "results.jsonl").write_text(json.dumps(record) + "\n")
    data = root / "data"
    data.mkdir()
    (data / "adaptive_musique.manifest.json").write_text(json.dumps({"outputs": [
        {"path": "questions.jsonl", "sha256": report.sha(questions.read_bytes())}]}))
    return root, run, questions


def test_completion_verifies_expected_ids_not_only_counts(tmp_path):
    root, run, _ = fixture(tmp_path)
    assert report.inspect_run(run, root)["status"] == "complete"
    output = tmp_path / "report"
    summary = report.build(run, output, root, no_plots=True)
    assert summary["report_status"] == "complete"
    assert summary["metrics"][0]["n"] == 1
    assert gzip.decompress((output / "artifacts/run/results.jsonl.gz").read_bytes()) == (run / "results.jsonl").read_bytes()
    assert "CC BY 4.0" in (output / "DATA_LICENSE.md").read_text()
    record = json.loads((run / "results.jsonl").read_text())
    record["scores"]["exact_match"] = 0
    (run / "results.jsonl").write_text(json.dumps(record) + "\n")
    inspection = report.inspect_run(run, root)
    assert any(issue["kind"] == "result_score_mismatch" for issue in inspection["issues"])
    assert inspection["status"] == "INCOMPLETE_OR_UNVERIFIED"
    record["scores"]["exact_match"] = 1
    record["id"] = "wrong"
    (run / "results.jsonl").write_text(json.dumps(record) + "\n")
    inspection = report.inspect_run(run, root)
    assert inspection["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert inspection["missing_unique_runs"] == 1
    assert any(issue["kind"] == "unexpected_result_ids" for issue in inspection["issues"])


def test_changed_inputs_and_torn_ledgers_never_become_complete(tmp_path):
    root, run, questions = fixture(tmp_path)
    questions.write_text('{"id":"changed"}\n')
    inspection = report.inspect_run(run, root)
    assert inspection["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert any(issue["kind"] == "input_verification_failed" for issue in inspection["issues"])
    path = run / "results.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"id":')
    raw = path.read_bytes()
    assert report.snapshot(path)[2][-1]["kind"] == "partial_trailing_record"
    assert path.read_bytes() == raw
    with path.open("ab") as handle:
        handle.write(b'\n{"status":"ok"}\n[]\n')
    inspection = report.inspect_run(run, root)
    assert inspection["status"] == "INCOMPLETE_OR_UNVERIFIED"
    assert any(issue["kind"] == "missing_record_id" for issue in inspection["issues"])
    assert len(inspection["records"]) == 1
