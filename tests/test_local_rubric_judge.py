import json

import pytest

from eval.local_rubric_judge import judge_input, parse_judgment


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
