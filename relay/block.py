"""KVBlock: the serializable unit of latent state.

A KVBlock holds the key and value tensors for one chunk of tokens,
for every transformer layer, as produced by a prefill with independent
attention (chunks do not attend to each other at remember time).

Layout (per layer l):
    K[l]: (n_heads, seq_len, head_dim)
    V[l]: (n_heads, seq_len, head_dim)

The block also carries:
    - chunk_id: content hash of the source text (idempotency)
    - positions: the local position ids 0..L-1 used at prefill time.
      These are re-indexed on concatenation (see concat.py).
    - dtype / quant: how the tensors are stored ('bf16', 'int8', 'int4')
    - scales: per-tensor (or per-channel) quantization scales when
      quant != 'bf16'

On-disk format: one safetensors file per chunk (see store/kv_store.py).
In RAM: a KVBlock holds live arrays; passing a relay between agents
in-process costs nothing (the single-box advantage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np

QuantMode = Literal["bf16", "int8", "int4"]


@dataclass
class KVBlock:
    """Per-chunk KV cache with position metadata."""

    chunk_id: str
    # keys[layer] -> (n_heads, seq_len, head_dim)
    keys: List[np.ndarray]
    # values[layer] -> (n_heads, seq_len, head_dim)
    values: List[np.ndarray]
    # local position ids, 0..L-1 at prefill; re-indexed on concat
    positions: np.ndarray
    quant: QuantMode = "bf16"
    # quantization scales, one per layer per tensor ('k'/'v'), when quant != 'bf16'
    scales: Dict[str, np.ndarray] = field(default_factory=dict)
    # source metadata (filled by store layer)
    source: str = ""
    token_count: int = 0

    def __post_init__(self) -> None:
        assert len(self.keys) == len(self.values), "K/V layer count mismatch"
        n = self.keys[0].shape[1]
        assert self.positions.shape == (n,), (
            f"positions shape {self.positions.shape} != seq_len {n}"
        )
        for k, v in zip(self.keys, self.values):
            assert k.shape == v.shape, "K/V shape mismatch within layer"
            assert k.shape[1] == n, "layer seq_len mismatch"
        if self.token_count == 0:
            self.token_count = n

    @property
    def n_layers(self) -> int:
        return len(self.keys)

    @property
    def seq_len(self) -> int:
        return self.keys[0].shape[1]

    @property
    def nbytes(self) -> int:
        """Approximate live memory footprint in bytes."""
        total = sum(k.nbytes + v.nbytes for k, v in zip(self.keys, self.values))
        total += self.positions.nbytes
        total += sum(s.nbytes for s in self.scales.values())
        return total

    def layer(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.keys[i], self.values[i]
