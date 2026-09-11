"""Agents: planner / reader / verifier / synthesizer.

Two variants share one interface:
  - Text agents: each hop decodes a short message; the next agent
    re-prefills it. This is the BASELINE (control arm) and the
    fallback product. Build this first.
  - Latent agents: hops pass KVBlock relays, no decoding until the
    synthesizer. This is the experiment.

Roles are LoRA adapters on one backbone in production; here they are
a label threaded through the Backend.
"""

from .interface import Agent, Relay, TextMessage
from .text_agents import (
    TextPlanner,
    TextReader,
    TextVerifier,
    TextSynthesizer,
    run_text_pipeline,
)
from .latent_agents import (
    LatentPlanner,
    LatentReader,
    LatentVerifier,
    LatentSynthesizer,
    run_latent_pipeline,
)

__all__ = [
    "Agent",
    "Relay",
    "TextMessage",
    "TextPlanner",
    "TextReader",
    "TextVerifier",
    "TextSynthesizer",
    "run_text_pipeline",
    "LatentPlanner",
    "LatentReader",
    "LatentVerifier",
    "LatentSynthesizer",
    "run_latent_pipeline",
]
