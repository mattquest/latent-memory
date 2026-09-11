"""Backend implementations.

NumpyMockBackend: a miniature transformer (2 layers, 4 heads) with
random weights in NumPy. Structurally faithful KV mechanics, zero
linguistic ability. Used for:
  - unit-testing relay concat / re-index / bridge logic end-to-end
  - validating the E3 mismatched-cache audit harness mechanics
  - smoke-testing the agent loop and MCP server plumbing

It must NEVER be used to draw conclusions about model quality --
only about machinery correctness.

MLXBackend: stub for the M5 Max. Fill in with mlx-lm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from relay.block import KVBlock


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------

class Backend(ABC):
    """What the agents need from any model runtime."""

    name: str = "base"

    @abstractmethod
    def prefill(self, tokens: list[int], adapter: str = "base") -> KVBlock:
        """Prefill tokens with independent attention; return the KV block."""

    @abstractmethod
    def decode(
        self,
        prefix: KVBlock | None,
        prompt_tokens: list[int],
        adapter: str = "base",
        max_tokens: int = 64,
    ) -> list[int]:
        """Decode max_tokens attending over prefix + prompt (causal)."""

    @abstractmethod
    def hidden_state(self, tokens: list[int], adapter: str = "base") -> np.ndarray:
        """Final-layer hidden state of the last token."""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    # -- tokenizer (byte-fallback, good enough for plumbing) -----------------
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_tokens(self, tokens: list[int]) -> str:
        return bytes(t & 0xFF for t in tokens).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# NumPy mock: tiny random transformer, faithful KV mechanics
# ---------------------------------------------------------------------------

@dataclass
class MockConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    vocab_size: int = 256  # byte-level
    seed: int = 0


class NumpyMockBackend(Backend):
    """Random-weight transformer in NumPy. Mechanics only, no language."""

    name = "numpy-mock"

    def __init__(self, config: MockConfig | None = None):
        self.cfg = config or MockConfig()
        rng = np.random.default_rng(self.cfg.seed)
        d = self.cfg.d_model
        self.wte = rng.standard_normal((self.cfg.vocab_size, d)).astype(np.float32) * 0.1
        self.layers = []
        for _ in range(self.cfg.n_layers):
            self.layers.append({
                "wq": rng.standard_normal((d, d)).astype(np.float32) * 0.1,
                "wk": rng.standard_normal((d, d)).astype(np.float32) * 0.1,
                "wv": rng.standard_normal((d, d)).astype(np.float32) * 0.1,
                "wo": rng.standard_normal((d, d)).astype(np.float32) * 0.1,
            })
        self.lm_head = rng.standard_normal((d, self.cfg.vocab_size)).astype(np.float32) * 0.1
        # adapter name -> no-op (mock has no adapters; role is a label)
        self.adapter = "base"

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    def _forward(self, x: np.ndarray, cache_k=None, cache_v=None, start_pos: int = 0):
        """x: (T, d). Returns (logits, new_k_list, new_v_list)."""
        T = x.shape[0]
        d = self.cfg.d_model
        H = self.cfg.n_heads
        hd = d // H
        new_ks, new_vs = [], []
        h = x
        for li, w in enumerate(self.layers):
            q = (h @ w["wq"]).reshape(T, H, hd)
            k = (h @ w["wk"]).reshape(T, H, hd)
            v = (h @ w["wv"]).reshape(T, H, hd)
            if cache_k is not None:
                k_full = np.concatenate([cache_k[li], k], axis=0)
                v_full = np.concatenate([cache_v[li], v], axis=0)
            else:
                k_full, v_full = k, v
            # causal attention over the full (cached + new) sequence
            scores = np.einsum("thd,shd->hts", q, k_full) / np.sqrt(hd)
            pos_q = start_pos + np.arange(T)
            pos_k = np.arange(start_pos + T)
            causal = pos_k[None, :] <= pos_q[:, None]
            scores = np.where(causal[None, :, :], scores, -1e9)
            # subtract max for stability
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            o = np.einsum("hts,shd->thd", attn, v_full).reshape(T, d)
            h = h + o @ w["wo"]  # residual, no norm (mock)
            new_ks.append(k_full)
            new_vs.append(v_full)
        logits = h @ self.lm_head
        return logits, new_ks, new_vs

    def prefill(self, tokens: list[int], adapter: str = "base") -> KVBlock:
        x = self.wte[np.asarray(tokens) % self.cfg.vocab_size]
        _, ks, vs = self._forward(x)
        # store as (n_heads, seq_len, head_dim)
        keys = [np.transpose(k, (1, 0, 2)).copy() for k in ks]
        values = [np.transpose(v, (1, 0, 2)).copy() for v in vs]
        import hashlib
        cid = hashlib.sha256(bytes(tokens)).hexdigest()[:16]
        return KVBlock(
            chunk_id=cid,
            keys=keys,
            values=values,
            positions=np.arange(len(tokens)),
            token_count=len(tokens),
        )

    def decode(self, prefix, prompt_tokens, adapter="base", max_tokens=64) -> list[int]:
        # Build cache from prefix (transpose back to (T, H, hd))
        if prefix is not None:
            ck = [np.transpose(k, (1, 0, 2)).copy() for k in prefix.keys]
            cv = [np.transpose(v, (1, 0, 2)).copy() for v in prefix.values]
            start = prefix.seq_len
        else:
            ck, cv, start = None, None, 0
        out: list[int] = []
        # prefill the prompt through the cache
        if prompt_tokens:
            x = self.wte[np.asarray(prompt_tokens) % self.cfg.vocab_size]
            logits, ck, cv = self._forward(x, ck, cv, start_pos=start)
            start = start + len(prompt_tokens)
        else:
            # need at least one forward to get logits; use last cache state
            raise ValueError("decode needs prompt_tokens in the mock backend")
        for _ in range(max_tokens):
            nxt = int(np.argmax(logits[-1]))
            out.append(nxt)
            x = self.wte[[nxt]]
            logits, ck, cv = self._forward(x, ck, cv, start_pos=start)
            start += 1
        return out

    def hidden_state(self, tokens: list[int], adapter: str = "base") -> np.ndarray:
        x = self.wte[np.asarray(tokens) % self.cfg.vocab_size]
        h = x
        d = self.cfg.d_model
        H = self.cfg.n_heads
        hd = d // H
        for w in self.layers:
            T = h.shape[0]
            q = (h @ w["wq"]).reshape(T, H, hd)
            k = (h @ w["wk"]).reshape(T, H, hd)
            v = (h @ w["wv"]).reshape(T, H, hd)
            scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
            causal = np.arange(T)[None, :] <= np.arange(T)[:, None]
            scores = np.where(causal[None, :, :], scores, -1e9)
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            o = np.einsum("hts,shd->thd", attn, v).reshape(T, d)
            h = h + o @ w["wo"]
        return h[-1].copy()
