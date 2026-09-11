"""Engine: model backend abstraction.

All agent code talks to a Backend, never to a framework directly.
Two implementations exist:

- NumpyMockBackend (this machine): a tiny random-weight transformer
  implemented in NumPy. It is NOT a language model -- it produces
  garbage text -- but its KV-cache mechanics (prefill, concat,
  causal decode over a prefix) are structurally faithful. It exists
  so the relay machinery, the audit harness, and the agent loop can
  be tested here without a GPU or MLX.

- MLXBackend (M5 Max): loads local Qwen3-8B BF16 via mlx-lm. It uses
  native MLXKVBlock snapshots, RoPE-correct concatenation, causal
  prefix continuation, real bridge recomputation, and packed KV storage.
  There are no trained role adapters. Use eval.real_experiments rather
  than the historical mock-agent pipeline.

The Backend contract:
    prefill(tokens, adapter) -> KVBlock
        Run the model over `tokens` with independent attention
        (no cross-chunk context) and return the KV block.
    decode(prefix, prompt_tokens, adapter, max_tokens) -> list[int]
        Decode causally over an already assembled prefix. The native
        backend's concat() owns RoPE reindexing and bridge recomputation.
    hidden_state(tokens, adapter) -> np.ndarray
        Final-layer hidden state of the last token (for the
        planner -> retriever projection head, Coconut-style).
"""

from .backends import Backend, NumpyMockBackend

__all__ = ["Backend", "NumpyMockBackend"]
