"""NumPy mock KV quantization (real native MLX precision lives in its backend).

The historical quant='bf16' label means unquantized here, usually float32.
The smoke test's 4x reduction is relative to float32, not actual bf16.

Default is int8 with per-tensor scales (QKVShare-style). bf16 is kept
for the final synthesizer hop when accuracy demands it. int4 is a stub:
the format is reserved, the implementation lands in Phase 2.

Scale convention: x_int8 = round(x_fp / scale), x_fp ~= x_int8 * scale,
with scale = max(abs(x)) / 127 computed per (layer, k/v) tensor.
"""

from __future__ import annotations

import numpy as np

from .block import KVBlock


def _quantize_tensor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.max(np.abs(x)).astype(np.float32)
    if scale == 0:
        scale = np.float32(1.0)
    scale = scale / 127.0
    q = np.round(x / scale).astype(np.int8)
    # saturate (round can produce 128 for the max value)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, np.asarray(scale, dtype=np.float32)


def _dequantize_tensor(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * float(scale)


def quantize_int8(block: KVBlock) -> KVBlock:
    """Return a copy of block with K/V stored as int8 + scales."""
    if block.quant == "int8":
        return block
    if block.quant != "bf16":
        raise ValueError(f"cannot quantize from {block.quant}; dequantize first")
    qk, qv, scales = [], [], {}
    for i, (k, v) in enumerate(zip(block.keys, block.values)):
        qki, ski = _quantize_tensor(k.astype(np.float32))
        qvi, svi = _quantize_tensor(v.astype(np.float32))
        qk.append(qki)
        qv.append(qvi)
        scales[f"k{i}"] = ski
        scales[f"v{i}"] = svi
    return KVBlock(
        chunk_id=block.chunk_id,
        keys=qk,
        values=qv,
        positions=block.positions.copy(),
        quant="int8",
        scales=scales,
        source=block.source,
        token_count=block.token_count,
    )


def dequantize_int8(block: KVBlock) -> KVBlock:
    """Return a copy of block with K/V restored to float32."""
    if block.quant == "bf16":
        return block
    if block.quant != "int8":
        raise ValueError(f"cannot dequantize from {block.quant}")
    k, v = [], []
    for i in range(block.n_layers):
        k.append(_dequantize_tensor(block.keys[i], block.scales[f"k{i}"]))
        v.append(_dequantize_tensor(block.values[i], block.scales[f"v{i}"]))
    return KVBlock(
        chunk_id=block.chunk_id,
        keys=k,
        values=v,
        positions=block.positions.copy(),
        quant="bf16",
        scales={},
        source=block.source,
        token_count=block.token_count,
    )


def quantize_error(block: KVBlock) -> float:
    """Max relative error introduced by an int8 round-trip (diagnostic)."""
    rt = dequantize_int8(quantize_int8(block))
    worst = 0.0
    for i in range(block.n_layers):
        k0, _ = block.layer(i)
        k1, _ = rt.layer(i)
        denom = np.max(np.abs(k0))
        if denom > 0:
            worst = max(worst, float(np.max(np.abs(k0 - k1)) / denom))
    return worst
