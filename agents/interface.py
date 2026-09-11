"""Agent interface: what flows between hops.

TextMessage: decoded JSON-ish dict. The text pipeline passes these.
Relay: a KVBlock plus the latent query vector (planner output) and
    hop metadata. The latent pipeline passes these.

Both carry a `trace` dict: which chunks were attended, attention
mass per block, verifier verdicts. The `why` tool decodes the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from relay.block import KVBlock


@dataclass
class TextMessage:
    role: str            # 'planner' | 'reader' | 'verifier' | 'synthesizer'
    kind: str            # 'query' | 'evidence' | 'verdict' | 'answer' | 'refine'
    text: str
    chunk_ids: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relay:
    """Latent handoff between agents. Passing it costs nothing in-process."""

    blocks: KVBlock               # concatenated relayed blocks
    latent_query: np.ndarray | None = None  # planner -> retriever projection
    source_roles: list[str] = field(default_factory=list)
    hop: int = 0
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return self.blocks.seq_len


class Agent:
    """Base class. Subclasses implement one hop."""

    role: str = "base"

    def __init__(self, backend, adapter: str | None = None):
        self.backend = backend
        self.adapter = adapter or self.role
