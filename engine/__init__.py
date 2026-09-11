"""Engine: model backend abstraction.

All agent code talks to a Backend, never to a framework directly.
Two implementations exist:

- NumpyMockBackend (this machine): a tiny random-weight transformer
  implemented in NumPy. It is NOT a language model -- it produces
  garbage text -- but its KV-cache mechanics (prefill, concat,
  causal decode over a prefix) are structurally faithful. It exists
  so the relay machinery, the audit harness, and the agent loop can
  be tested here without a GPU or MLX.

- MLXBackend (M5 Max, stub): loads Qwen3-8B via mlx-lm, hot-swaps
  LoRA role adapters, and performs prefill / relayed decode with
  custom attention masks and position-id handling. This is the
  production path; see backends_mlx.py.

The Backend contract:
    prefill(tokens, adapter) -> KVBlock
        Run the model over `tokens` with independent attention
        (no cross-chunk context) and return the KV block.
    decode(prefix, prompt_tokens, adapter, max_tokens) -> str
        Concatenate prefix blocks, re-index positions, recompute
        bridge tokens, then decode `max_tokens` causally.
    hidden_state(tokens, adapter) -> np.ndarray
        Final-layer hidden state of the last token (for the
        planner -> retriever projection head, Coconut-style).
"""

from .backends import Backend, NumpyMockBackend

__all__ = ["Backend", "NumpyMockBackend"]
