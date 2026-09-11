"""CPU checks for numerical decisions; these do not exercise real weights."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("adaptive_preflight", Path(__file__).parents[1] / "scripts/preflight_adaptive.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def test_predeclared_logit_tolerances_and_argmax_are_separate():
    reference = np.array([1., 2., 10., -3.])
    close = preflight.compare_logits(reference, reference + .01)
    assert close["scale_aware_close"] and close["argmax_equal"]
    changed = preflight.compare_logits(reference, reference + 1)
    assert changed["argmax_equal"] and not changed["scale_aware_close"]
    near_tie = preflight.compare_logits([1, 2, 2.001], [1, 2.001, 2])
    assert near_tie["scale_aware_close"] and not near_tie["argmax_equal"]
    assert near_tie["reference_top_two_margin"] == pytest.approx(.001)
    tied = preflight.compare_logits([5, 5, 0], [5, 4.99, 0])
    assert tied["argmax_equal"] and tied["reference_argmax"] == 0
    assert [value["token_id"] for value in tied["reference_top10"][:2]] == [0, 1]
    with pytest.raises(ValueError, match="Nonfinite"):
        preflight.compare_logits([1, 2], [float("nan"), 2])


def test_cap_error_requires_rejection():
    assert not preflight.expect_cap_error(lambda: None)["rejected"]
    def reject():
        raise ValueError("8193 exceeds 8192")
    result = preflight.expect_cap_error(reject)
    assert result["rejected"] and "8192" in result["message"]
