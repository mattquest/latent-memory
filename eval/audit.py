"""E3 -- mismatched-cache audit.

The evidence standard: a latent relay transmits example-specific
information ONLY if the receiver's behavior depends on the sender's
private content. The audit:

  1. Build a small set of (context, question) pairs where the answer
     requires the context (the receiver must need private content).
  2. For each pair, run the latent pipeline three ways:
       a. CORRECT cache: the relay built from this example's chunks.
       b. MISMATCHED cache: a relay built from a DIFFERENT example's
          chunks (same shapes, wrong content).
       c. ZEROED cache: same shapes, all zeros.
       d. (optional) MOMENT-MATCHED RANDOM: random tensors with the
          same per-layer mean/variance as the correct cache.
  3. Measure: does the synthesizer's output change across (a)-(d)?

Pass criterion: accuracy/behavior with (a) >> (b), (c), (d).
If the receiver behaves the same with mismatched caches, the relay
is NOT transmitting information -- the pipeline is succeeding (or
failing) for other reasons, and no latent claim can be made.

With the NumpyMockBackend there is no real language, so the audit
measures a mechanical proxy: the L2 distance between the
synthesizer's pre-decode hidden state under each cache condition.
A correct relay must move the hidden state; mismatched/zeroed must
move it differently (or not at all). On the MLX backend, replace
the proxy with actual answer accuracy on a small QA set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from relay import KVBlock, concat_blocks
from engine.backends import Backend


@dataclass
class AuditResult:
    n_examples: int
    # mean L2 distance of final hidden state vs the correct-cache run
    drift_mismatched: float
    drift_zeroed: float
    drift_random: float
    # did the correct cache move the state at all vs no-prefix baseline?
    signal_strength: float
    passed: bool
    detail: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"E3 audit: {status} over {self.n_examples} examples\n"
            f"  signal (correct cache vs none): {self.signal_strength:.4f}\n"
            f"  drift mismatched: {self.drift_mismatched:.4f}\n"
            f"  drift zeroed:     {self.drift_zeroed:.4f}\n"
            f"  drift random:     {self.drift_random:.4f}\n"
            "Pass = signal > 0 and mismatched/zeroed/random all drift "
            "significantly from correct."
        )


def _zeroed(block: KVBlock) -> KVBlock:
    return KVBlock(
        chunk_id=block.chunk_id + ":zeroed",
        keys=[np.zeros_like(k) for k in block.keys],
        values=[np.zeros_like(v) for v in block.values],
        positions=block.positions.copy(),
        token_count=block.token_count,
    )


def _moment_matched_random(block: KVBlock, rng: np.random.Generator) -> KVBlock:
    keys, values = [], []
    for k, v in zip(block.keys, block.values):
        kk = rng.standard_normal(k.shape).astype(np.float32)
        kk = (kk - kk.mean()) / (kk.std() + 1e-6) * k.std() + k.mean()
        vv = rng.standard_normal(v.shape).astype(np.float32)
        vv = (vv - vv.mean()) / (vv.std() + 1e-6) * v.std() + v.mean()
        keys.append(kk.astype(k.dtype))
        values.append(vv.astype(v.dtype))
    return KVBlock(
        chunk_id=block.chunk_id + ":random",
        keys=keys, values=values,
        positions=block.positions.copy(), token_count=block.token_count,
    )


def _final_hidden(backend: Backend, prefix: KVBlock | None,
                  prompt: str, adapter: str = "synthesizer") -> np.ndarray:
    """Hidden state the synthesizer would decode from (mechanical proxy)."""
    toks = backend.encode(prompt)
    if prefix is None:
        return backend.hidden_state(toks, adapter)
    # simulate: prefill prompt tokens attending over the prefix by
    # concatenating prefix K/V with a fresh prefill of the prompt.
    # The mock backend's decode() threads the cache through, so we
    # read the hidden state after one cached forward instead.
    return _cached_hidden(backend, prefix, toks, adapter)


def _cached_hidden(backend: Backend, prefix: KVBlock, toks: list[int],
                   adapter: str) -> np.ndarray:
    # Mock-backend path: reuse its internals via a 1-token decode and
    # read the last hidden state. For other backends, override this.
    if backend.name == "numpy-mock":
        ck = [np.transpose(k, (1, 0, 2)).copy() for k in prefix.keys]
        cv = [np.transpose(v, (1, 0, 2)).copy() for v in prefix.values]
        x = backend.wte[np.asarray(toks) % backend.vocab_size]
        # replicate _forward to grab the last hidden state
        h = x
        d, H = backend.cfg.d_model, backend.cfg.n_heads
        hd = d // H
        T = h.shape[0]
        start = prefix.seq_len
        for li, w in enumerate(backend.layers):
            q = (h @ w["wq"]).reshape(T, H, hd)
            k = (h @ w["wk"]).reshape(T, H, hd)
            v = (h @ w["wv"]).reshape(T, H, hd)
            k_full = np.concatenate([ck[li], k], axis=0)
            v_full = np.concatenate([cv[li], v], axis=0)
            scores = np.einsum("thd,shd->hts", q, k_full) / np.sqrt(hd)
            pos_q = start + np.arange(T)
            pos_k = np.arange(start + T)
            causal = pos_k[None, :] <= pos_q[:, None]
            scores = np.where(causal[None, :, :], scores, -1e9)
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            o = np.einsum("hts,shd->thd", attn, v_full).reshape(T, d)
            h = h + o @ w["wo"]
        return h[-1].copy()
    raise NotImplementedError(f"audit hidden-state probe for {backend.name}")


def mismatched_cache_audit(
    backend: Backend,
    examples: list[tuple[str, str]],
    seed: int = 0,
) -> AuditResult:
    """Run E3. examples: list of (context_text, question_text)."""
    rng = np.random.default_rng(seed)
    detail = []
    drifts_mm, drifts_z, drifts_r, signals = [], [], [], []

    # prefill one block per example context
    blocks = [backend.prefill(backend.encode(ctx), "reader") for ctx, _ in examples]

    for i, ((ctx, q), blk) in enumerate(zip(examples, blocks)):
        h_none = _final_hidden(backend, None, q)
        h_ok = _final_hidden(backend, blk, q)
        # mismatched: another example's block (wrap around)
        h_mm = _final_hidden(backend, blocks[(i + 1) % len(blocks)], q)
        h_z = _final_hidden(backend, _zeroed(blk), q)
        h_r = _final_hidden(backend, _moment_matched_random(blk, rng), q)

        signal = float(np.linalg.norm(h_ok - h_none))
        d_mm = float(np.linalg.norm(h_mm - h_ok))
        d_z = float(np.linalg.norm(h_z - h_ok))
        d_r = float(np.linalg.norm(h_r - h_ok))
        signals.append(signal)
        drifts_mm.append(d_mm)
        drifts_z.append(d_z)
        drifts_r.append(d_r)
        detail.append({"example": i, "signal": signal, "drift_mm": d_mm,
                       "drift_z": d_z, "drift_r": d_r})

    sig = float(np.mean(signals))
    dmm, dz, dr = (float(np.mean(drifts_mm)), float(np.mean(drifts_z)),
                   float(np.mean(drifts_r)))
    # pass: the correct cache moves the state, and every corrupted
    # cache moves it somewhere else (all drifts >> 0 relative to signal)
    passed = sig > 1e-6 and min(dmm, dz, dr) > 0.5 * sig
    return AuditResult(
        n_examples=len(examples),
        drift_mismatched=dmm, drift_zeroed=dz, drift_random=dr,
        signal_strength=sig, passed=passed, detail=detail,
    )
