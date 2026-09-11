"""MLXBackend: production runtime for the M5 Max (STUB).

Target: Qwen3-8B via mlx-lm, with hot-swappable LoRA role adapters
(planner / reader / verifier / synthesizer). The base weights never
move; a role switch is an adapter switch.

What needs implementing here (morning work):
  1. Load Qwen3-8B with mlx-lm; keep one model resident.
  2. LoRA adapter switching per role (mlx-lm supports this; verify
     the switch cost is negligible vs a prefill).
  3. prefill(): run with independent attention. mlx-lm's generate
     pipeline assumes causal chat; you need direct access to the
     model's forward to capture per-layer K/V. See mlx-lm's
     `model()` call pattern.
  4. decode(): custom attention mask over concatenated relay blocks
     (blocks fully visible as prefix; new tokens causal) plus
     position-id handling for the re-indexed range. This is the
     single hardest piece of engineering in the project.
  5. KV extraction: pull per-layer K/V as MLX arrays, wrap in the
     relay.KVBlock interface (this package's block.py is
     NumPy-typed; add an `array_module` seam or convert at the
     boundary -- conversion costs a copy, measure it).

Do NOT try to run this file on Linux; it imports mlx unconditionally
when instantiated. The import is deferred so the package imports
cleanly everywhere.
"""

from __future__ import annotations


class MLXBackend:  # not a Backend subclass yet -- wire up on the Mac
    """Stub. Raises on any use until implemented."""

    name = "mlx-qwen3-8b"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MLXBackend is a stub for the M5 Max. "
            "Implement per the checklist in backends_mlx.py, then "
            "subclass engine.backends.Backend."
        )
