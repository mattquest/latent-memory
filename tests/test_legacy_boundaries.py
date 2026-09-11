import pytest

from agents.latent_agents import run_latent_pipeline
from agents.text_agents import run_text_pipeline


class RealBackendLabel:
    name = "real-model"


@pytest.mark.parametrize("pipeline", [run_latent_pipeline, run_text_pipeline])
def test_legacy_scaffolds_cannot_silently_score_real_models(pipeline):
    with pytest.raises(NotImplementedError, match="mock-only"):
        pipeline(RealBackendLabel(), "question", lambda *a, **kw: [])
