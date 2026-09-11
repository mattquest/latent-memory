"""NumPy mock block concatenation with position metadata re-indexing.

This does not rotate already-RoPE-encoded keys and is not TurboRAG.
Real Qwen3 relay concatenation is implemented by MLXBackend.concat().

Each chunk is prefilled with independent attention and local position
ids 0..L-1. When a receiver consumes several relays, the blocks are
concatenated along the sequence axis and the position ids are
re-indexed into one consecutive range:

    block0: 0..L0-1, block1: L0..L0+L1-1, ...

This is what makes precomputed blocks reusable across queries: the
stored layout is position-independent, and the re-indexing happens
at concat time, cheaply, in the receiver.

Quantization note: legacy quant='bf16' denotes unquantized NumPy arrays,
which in the mock are actually float32. Inputs in int8 are
dequantized first (per-tensor scales don't survive concatenation).
The receiver re-quantizes the result according to its policy.

Causal masking note: the receiver's attention mask must treat the
concatenated blocks as prefix (fully visible) with the receiver's own
new tokens attending causally over them. The engine layer builds that
mask; this module only handles the tensors and position ids.
"""

from __future__ import annotations

import numpy as np

from .block import KVBlock
from .quant import dequantize_int8


def reindex_positions(blocks: list[KVBlock]) -> list[np.ndarray]:
    """Consecutive position ids for each block in concat order."""
    out = []
    offset = 0
    for b in blocks:
        out.append(np.arange(offset, offset + b.seq_len, dtype=np.int64))
        offset += b.seq_len
    return out


def concat_blocks(blocks: list[KVBlock]) -> KVBlock:
    """Concatenate blocks along seq_len; re-index positions to bf16."""
    if not blocks:
        raise ValueError("concat_blocks needs at least one block")

    work = [dequantize_int8(b) if b.quant != "bf16" else b for b in blocks]

    n_layers = work[0].n_layers
    for b in work:
        assert b.n_layers == n_layers, "layer count mismatch in concat"
        assert b.keys[0].shape[0] == work[0].keys[0].shape[0], "head count mismatch"
        assert b.keys[0].shape[2] == work[0].keys[0].shape[2], "head dim mismatch"

    keys = [np.concatenate([b.keys[l] for b in work], axis=1) for l in range(n_layers)]
    values = [np.concatenate([b.values[l] for b in work], axis=1) for l in range(n_layers)]
    positions = np.concatenate(reindex_positions(work)).astype(np.int64)

    return KVBlock(
        chunk_id="+".join(b.chunk_id for b in work),
        keys=keys,
        values=values,
        positions=positions,
        quant="bf16",
        scales={},
        source="concat",
        token_count=sum(b.token_count for b in work),
    )
