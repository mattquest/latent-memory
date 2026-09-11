"""Latent relay: KV-cache block format, quantization, and relay mechanics.

A relay is the unit of latent communication between agents. It holds
per-layer key/value tensors for a chunk of tokens, plus the metadata
needed to concatenate blocks from different chunks and re-index them
into a single consecutive position range (TurboRAG-style).

The array backend is abstracted: this package works with NumPy arrays
for local testing. The MLX backend on the M5 Max swaps in MLX arrays
without changing this interface (see engine/backends.py).
"""

from .block import KVBlock
from .quant import quantize_int8, dequantize_int8
from .concat import concat_blocks, reindex_positions
from .bridge import select_bridge_tokens

__all__ = [
    "KVBlock",
    "quantize_int8",
    "dequantize_int8",
    "concat_blocks",
    "reindex_positions",
    "select_bridge_tokens",
]
