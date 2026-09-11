# Latent Retrieval Memory

A local, latent-space multi-agent memory server. Text goes in and out over MCP;
inside, agents communicate by relaying transformer KV caches instead of decoded text.

**Status (as of Sep 11, 2026):** relay machinery + audit harness + agent pipelines are
built and tested on a Linux VM with a mock backend. The MLX backend for the M5 Max
is stubbed with a precise implementation checklist. Nothing here has touched real
model weights yet.

## The idea in 30 seconds

A normal multi-agent retrieval pipeline decodes text at every hop: reader reads
chunks → writes a summary → verifier re-reads the summary → writes a verdict →
synthesizer re-reads everything → writes the answer. Every hop pays a full
decode + re-prefill.

This pipeline skips the decode. Each reader prefills its chunks once (at ingest
time, even) and hands the **KV cache** to the next agent as a Python object.
The verifier and synthesizer attend over relayed KV blocks directly. Only the
synthesizer decodes, once, at the end.

Why it might win: latent hops are ~100–200ms (one prefill over precomputed blocks),
so the pipeline can afford many more retrieve → verify → re-query iterations per
question than a text pipeline. If iteration budget converts to accuracy on a
long-horizon memory benchmark, that's the publishable result.

Why it might not: a causal audit (Cheng et al., Aug 2026) found KV relay only
transmits information when the receiver needs the sender's *private* content —
which retrieval satisfies — but predicts **parity, not wins, at equal hop counts**.
The win has to come from running *more* hops. Experiment E2 is the thesis;
if the latent accuracy curve goes flat past 5 hops, kill the latent path and
ship the text baseline (it's already built).

## Repository layout

```
latent-memory/
  relay/        KVBlock format, int8 quantization, block concat with
                position re-indexing (TurboRAG-style), bridge-token
                selection (CacheBlend-style). TESTED.
  engine/       Backend abstraction. NumpyMockBackend (random-weight
                transformer, faithful KV mechanics, zero language)
                works everywhere. backends_mlx.py is the M5 Max
                stub with an implementation checklist.
  agents/       Planner / reader / verifier / synthesizer in two
                variants sharing one interface:
                  text_agents.py    BASELINE control arm. Built first,
                                    works now, ships if latent fails.
                  latent_agents.py  Experiment arm. Relay mechanics
                                    wired; verifier head + planner
                                    projection are stubs.
  store/        SQLite metadata (chunks, versions, trace log) +
                KVBlockStore (one safetensors file per chunk, LRU).
  eval/         audit.py = E3 mismatched-cache audit. IMPLEMENTED
                and passing. runners.py = E1/E2/E4/E5/E6 specified,
                awaiting the MLX backend.
  server/       MCP server skeleton (remember / retrieve / why).
                Wire up with the MCP SDK on the Mac.
  train/        (empty) LoRA recipes: role adapters, compressor,
                planner→retriever projection head.
  scripts/      smoke.py -- validates everything on any machine.
```

## What works right now

Run this anywhere (no GPU, no model weights):

```bash
pip install -r requirements.txt
python scripts/smoke.py
```

It validates:
1. **Relay core** — int8 quantization at 4x compression with <0.4% max relative
   error; block concatenation with consecutive position re-indexing; bridge
   tokens clustering at block boundaries.
2. **E3 audit** — the mismatched-cache evidence standard, passing on the mock
   backend (mechanical proxy: hidden-state drift, not answer accuracy).
3. **Both pipelines** — text and latent hop mechanics end-to-end, with the
   latent path moving 49 relay tokens and decoding zero mid-pipe.

## Morning checklist (in order)

**1. MLX backend** — `engine/backends_mlx.py` has the checklist. This is the
hard engineering: one resident Qwen3-8B via mlx-lm, LoRA adapter hot-swap per
role, and the custom attention mask + position-id handling for relayed decode.
Nothing else unblocks without this.

**2. Swap the E3 proxy for accuracy** — `eval/audit.py::_final_hidden` uses a
hidden-state drift proxy on the mock. On MLX, replace it with actual answer
accuracy over a small QA set where the answer requires the context. The
pass/fail logic stays the same.

**3. Train the verifier head** — `agents/latent_agents.py::LatentVerifier` has
a hop-count heuristic. Replace with the learned head over attention-mass
features before trusting E1/E2.

**4. Run E1, then E2** — E1 should show parity (audit prediction). E2 is the
kill criterion: latent accuracy must keep climbing past 5 hops where text
gets too slow. Flat curve = kill the latent path, ship `text_agents.py`.

**5. MCP server** — `server/server.py` has the wiring checklist. `remember`
does chunk → embed (Qwen3-Embedding-0.6B) → contradiction check → prefill →
int8 → store. `retrieve` runs the latent pipeline. `why` decodes the trace.

**6. Fill the doc gaps** — the design doc's §§3.1, 3.3, 7 are empty: backbone
specifics, the 128GB memory-budget math for 1M–10M tokens at int8, and risks.

## Design decisions worth knowing

- **Concat is always bf16.** int8 blocks are dequantized before concatenation
  (per-tensor scales don't survive it); the receiver re-quantizes per policy.
- **Positions are re-indexed at concat, not at store.** Disk layout is
  position-independent (0..L-1); the consecutive range is built cheaply
  in the receiver. This is what makes blocks reusable across queries.
- **Bridge recompute defaults to 15%.** Tunable via `bridge_ratio`; E4 finds
  the knee. 0% means pure concat — if that matches 100%, simplify the design.
- **The text pipeline is not a prototype.** It's the control arm for every
  claim and the fallback product. Treat it with the same care.
- **Security model = the text boundary.** Latent state never leaves the
  process; the MCP tools are text-only. Per-tenant cache salting (PROMPTPEEK
  defense) applies if this ever goes multi-tenant.

## References

- Cheng et al., "When Does Latent Communication Pay?" arXiv 2608.04893 — the audit, the evidence standard.
- Zou et al., LatentMAS (2025) — the relay pattern that passed the audit.
- TurboRAG (EMNLP 2025), CacheBlend (EuroSys 2025) — position re-indexing, selective recompute.
- BEAM (1M/10M-token memory benchmark) — the scoreboard.
- Full design doc: "Latent Retrieval Memory" (Google Doc, Sep 10, 2026).
