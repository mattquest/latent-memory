#!/usr/bin/env python3
"""Smoke test: validates the relay machinery end-to-end on any machine.

Runs without a GPU, without MLX, without model weights:
  1. relay core (quant / concat / reindex / bridge)
  2. E3 mismatched-cache audit (mechanical proxy)
  3. text + latent pipeline hop mechanics

Usage: python scripts/smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from relay import KVBlock, quantize_int8, concat_blocks, select_bridge_tokens
from relay.quant import quantize_error
from engine.backends import NumpyMockBackend, MockConfig
from eval.audit import mismatched_cache_audit
from agents import run_text_pipeline, run_latent_pipeline


def fake_block(cid, L, n_layers=2, n_heads=4, hd=16):
    ks = [np.random.randn(n_heads, L, hd).astype(np.float32) for _ in range(n_layers)]
    vs = [np.random.randn(n_heads, L, hd).astype(np.float32) for _ in range(n_layers)]
    return KVBlock(cid, ks, vs, np.arange(L), token_count=L)


def main() -> int:
    # 1. relay core
    b = fake_block("a", 64)
    q = quantize_int8(b)
    assert q.nbytes < b.nbytes * 0.3, "int8 should be ~4x smaller"
    assert quantize_error(b) < 0.05, "quant error too large"
    c = concat_blocks([q, fake_block("b", 32)])
    assert c.seq_len == 96 and (c.positions == np.arange(96)).all()
    sel = select_bridge_tokens([64, 32], ratio=0.15)
    assert sel and all(abs(t - 64) <= 10 for t in sel)
    print("[ok] relay core: quant 4x, concat reindexed, bridge at boundaries")

    # 2. E3 audit
    backend = NumpyMockBackend(MockConfig(seed=7))
    examples = [
        ("The Eiffel Tower is in Paris.", "Where is the Eiffel Tower?"),
        ("KV caches store keys and values.", "What do KV caches store?"),
        ("TurboRAG re-indexes positions.", "What does TurboRAG re-index?"),
    ]
    res = mismatched_cache_audit(backend, examples)
    assert res.passed, "E3 audit should pass on the mock backend"
    print(f"[ok] E3 audit: PASS (signal={res.signal_strength:.3f})")

    # 3. pipelines
    chunks = ["chunk one text.", "chunk two text.", "chunk three text."]
    blocks = [backend.prefill(backend.encode(t), "reader") for t in chunks]
    tr = run_text_pipeline(backend, "q?", lambda q, k=8: chunks, max_hops=2)
    lr = run_latent_pipeline(backend, "q?", lambda qv, k=8: blocks, max_hops=2)
    assert tr.hops >= 1 and lr.hops >= 1
    print(f"[ok] pipelines: text {tr.hops} hops, latent {lr.hops} hops, "
          f"{lr.trace[-1].get('relay_tokens')} relay tokens, zero decoded mid-pipe")

    print("\nAll smoke tests pass. Machinery is sound; plug in MLXBackend next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
