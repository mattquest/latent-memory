"""Bridge-token selection for selective recompute (CacheBlend/CacheClip-style).

Pure concatenation loses cross-chunk attention: tokens in block B
never attended to tokens in block A at prefill time. The fix is to
recompute a small set of "bridge" tokens at block boundaries with
full attention over the concatenated prefix. In practice 15-20% of
tokens recomputed recovers most of the quality (CacheBlend, EuroSys
2025); the exact ratio is experiment E4.

This module selects *which* token positions to recompute. The engine
performs the actual recompute (it owns the model weights).
"""

from __future__ import annotations

import numpy as np


def select_bridge_tokens(
    block_lens: list[int],
    ratio: float = 0.15,
) -> list[int]:
    """Pick token indices to recompute, concentrated at block boundaries.

    Args:
        block_lens: token counts per block in concat order.
        ratio: fraction of total tokens to mark for recompute.

    Returns:
        Sorted list of absolute token indices (in the concatenated
        sequence) to recompute with full attention.

    Strategy: take a window around each block boundary (the last
    `w` tokens of the left block and first `w` of the right block),
    where `w` is sized so the total is ~ratio of all tokens. Boundary
    tokens carry the most cross-chunk signal; interior tokens are
    well-approximated by their independent prefill.
    """
    total = sum(block_lens)
    if total == 0 or ratio <= 0:
        return []
    if ratio >= 1.0:
        return list(range(total))

    n_boundaries = max(len(block_lens) - 1, 1)
    # tokens per side of each boundary
    w = max(1, int(total * ratio / (2 * n_boundaries)))

    chosen: set[int] = set()
    offset = 0
    for i, ln in enumerate(block_lens):
        if i > 0:
            # left side: last w tokens of previous block
            for t in range(max(offset - w, offset - ln_prev), offset):
                chosen.add(t)
            # right side: first w tokens of this block
            for t in range(offset, min(offset + w, offset + ln)):
                chosen.add(t)
        ln_prev = ln
        offset += ln

    return sorted(chosen)


def bridge_coverage(block_lens: list[int], ratio: float) -> dict:
    """Diagnostic: how many tokens per block are selected."""
    sel = set(select_bridge_tokens(block_lens, ratio))
    out = {}
    offset = 0
    for i, ln in enumerate(block_lens):
        n = sum(1 for t in range(offset, offset + ln) if t in sel)
        out[f"block_{i}"] = {"len": ln, "recomputed": n}
        offset += ln
    return out
