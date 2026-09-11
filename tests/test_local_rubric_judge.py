import json

import pytest

from eval.local_rubric_judge import judge_input, judge_results, parse_judgment


def test_judge_parsing_rejects_invalid_denominators():
    assert parse_judgment('{"criterion_met":[true,false]}', 2)["criterion_fraction"] == .5
    assert parse_judgment('```json\n{"criterion_met":[true]}\n```', 1)["all_criteria_met"]
    for text in ('{"criterion_met":[1]}', '{"criterion_met":[true,false]}',
                 '{"criterion_met":["true"]}', 'not JSON'):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_judgment(text, 1)


def test_judge_input_excludes_condition_and_timings():
    example = {"id": "x", "question": "Q", "answers": ["A"],
               "metadata": {"rubric": ["says A"], "case": "latent", "latency": 123}}
    payload = json.loads(judge_input(example, "A"))
    assert set(payload) == {"question", "reference_answers", "criteria", "candidate_answer"}
    assert "latent" not in json.dumps(payload)


class ProtocolOnlyJudge:
    """Fixed JSON for resume bookkeeping tests, never model-quality scoring."""
    calls = 0

    def metadata(self):
        return {"backend": "unit-test-json-protocol"}

    def chat_prompt(self, user, system):
        return list(user.encode())

    def synchronize(self):
        pass

    def decode(self, prefix, prompt, max_tokens):
        self.calls += 1
        return list(b'{"criterion_met":[true],"reason":"unit test"}')

    def decode_tokens(self, tokens):
        return bytes(tokens).decode()


def write_judge_inputs(tmp_path):
    examples = [{"id": name, "dataset": "beam", "question": "Question " + name,
                 "answers": ["A"], "evidence": [{"id": name, "text": "A"}],
                 "metadata": {"requires_rubric_judge": True, "rubric": ["says A"]}}
                for name in ("one", "two")]
    candidates = [{"status": "ok", "id": name + "-result", "example_id": name,
                   "prediction": "A", "case": {"arm": "latent", "repeat": 0}}
                  for name in ("one", "two")]
    dataset, results = tmp_path / "dataset.jsonl", tmp_path / "results.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in examples))
    results.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    return dataset, results, candidates


def test_identical_judge_resume_does_not_repeat_or_inflate_denominators(tmp_path):
    dataset, results, _ = write_judge_inputs(tmp_path)
    backend = ProtocolOnlyJudge()
    directory = tmp_path / "judge"
    first = judge_results(backend, str(dataset), [str(results)], str(directory))
    second = judge_results(backend, str(dataset), [str(results)], str(directory))
    assert first["n_valid"] == second["n_valid"] == backend.calls == 2
    identity = json.loads((directory / "manifest.json").read_text())["identity"]
    assert identity["candidate_count"] == 2
    assert len(identity["candidate_set_sha256"]) == 64


def test_changed_candidate_rejects_resume_before_adding_a_second_grade(tmp_path):
    dataset, results, candidates = write_judge_inputs(tmp_path)
    backend = ProtocolOnlyJudge()
    directory = tmp_path / "judge"
    judge_results(backend, str(dataset), [str(results)], str(directory))
    previous = (directory / "judgments.jsonl").read_bytes()
    candidates[0]["prediction"] = "Changed answer"
    results.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    with pytest.raises(ValueError, match="resume identity mismatch"):
        judge_results(backend, str(dataset), [str(results)], str(directory))
    assert (directory / "judgments.jsonl").read_bytes() == previous
    assert backend.calls == 2


def test_selected_limit_binds_the_judge_candidate_set(tmp_path):
    dataset, results, _ = write_judge_inputs(tmp_path)
    backend = ProtocolOnlyJudge()
    directory = tmp_path / "judge"
    first = judge_results(backend, str(dataset), [str(results)], str(directory), limit=1)
    assert first["n_valid"] == 1
    with pytest.raises(ValueError, match="resume identity mismatch"):
        judge_results(backend, str(dataset), [str(results)], str(directory), limit=2)
    assert backend.calls == 1


def test_conflicting_answers_in_input_files_cannot_count_as_two_examples(tmp_path):
    dataset, results, candidates = write_judge_inputs(tmp_path)
    other = tmp_path / "other-results.jsonl"
    candidates[0]["prediction"] = "Changed answer"
    other.write_text(json.dumps(candidates[0]) + "\n")
    backend = ProtocolOnlyJudge()
    with pytest.raises(ValueError, match="Conflicting candidates"):
        judge_results(backend, str(dataset), [str(results), str(other)], str(tmp_path / "judge"))
    assert backend.calls == 0
