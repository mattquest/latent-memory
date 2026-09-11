# Experiment protocol, September 11, 2026

Recorded before scoring the experimental arms. The guiding source is
[the supplied design document](https://docs.google.com/document/d/1_Ug_QCTOmf-D1QDMl8qOVvevATzU7QZtuQLed260hJ4/edit),
snapshotted in `guiding-plan.txt`. Changes to this protocol must be reported.

Use the official Qwen3-8B BF16 model, revision
`b968826d9c46dd6066d109eabc6255188de91218`, with one resident model shared
across conditions. No trained role adapters, verifier, embedding projection,
or compressor exist. Those mechanisms must not be simulated and reported as
trained systems. Disable thinking and use deterministic greedy decoding for
controlled comparisons; this is a reproducibility choice rather than the
model card's recommended non-thinking sampling configuration.

Run an answer-scored E3 private-fact cache audit first, with correct, mismatched,
zero, moment-matched random, and no-context conditions. Test real RoPE
repositioning, causal prefix continuation, and 100% recomputation against
ordinary joint prefill before interpreting a quality difference.

E1 and E2 compare a decoded text relay, a KV relay, and direct text context
where useful. Give conditions the same question, evidence acquisition order,
budgets, and output caps. Record the precise prompt and all intermediate costs.
Use hop budgets 1, 3, 5, 10, and 20 where the context cap permits. Separate
fixed-order and oracle-evidence diagnostics from actual retrieval over a
full corpus. Increasing available evidence is not evidence of learned search.
Do not infer the project's thesis or kill it from an oracle schedule, an
exhausted evidence set, a tiny sample, or an untrained controller.

E4 varies actual recomputation over boundary-adjacent positions at ratios
0, 0.1, 0.2, and 1. This is a boundary heuristic, not a reproduction of
CacheBlend's token selection. E5 compares actual BF16, int8, and packed int4
cache storage, recording scale overhead and dequantized working memory.
E6 separates chronological updates, active-version filtering, and unresolved
conflicts. Gold annotations may be used for scoring or explicitly labeled
oracle controls only, never hidden inside retrieval or ordinary prompts.

Use separate development and test synthetic seeds. Public QA development
subsets use deterministic hash-ranked sampling and retain dataset provenance.
BEAM inputs must be retrieved from full downloaded conversation histories,
without putting probe answers or rubrics into the memory corpus. Its rubric
scores require a judge; exact match against its free-form reference is only a
diagnostic and must never be labeled the official BEAM score.

Save per-example JSONL with answers, conditions, tokens returned to the host,
internal decoded tokens, prefill/recompute counts, synchronized wall time,
ingestion cost, and failures. Aggregate exact match, token F1, uncertainty,
p50 and p95. Preserve rejected or failed pilot runs and identify excluded
records. Model loading and cache ingestion are separate from warm query
latency. A sample too small for a 2-point noninferiority conclusion stays
inconclusive; a point estimate alone is not a parity test.

Laptop limits: one guarded model process, 32 GiB MLX allocation limit,
32 GiB process-tree RSS limit, at least 12 GiB available system RAM and
40 GiB free disk, an 8,192-token context cap, and a six-hour wall-time cap per
overnight batch. Stop on battery power or severe reported CPU thermal
throttling. Keep idle sleep inhibited only while the guarded child runs.
No administrator commands, system memory tuning, external model APIs,
personal-data ingestion, or paid compute are needed.

## Development decisions before the main run

The initial eight-example private-fact development pilot passed all correct
cache and precision cases. The initial multi-document development run showed
truncated Markdown evidence notes and poor ranking from unweighted word
overlap. That run was stopped and retained in `runs/development`. A single
revision replaced the notes with compact JSON, increased their ceiling from
96 to 192 tokens, and used query-only BM25 for both arms. BM25 breaks ties in
source order; annotation-like document IDs never determine ranking. It keeps
zero-score documents available for larger evidence budgets. The revised
development run is `runs/development-v2`; no synthetic test-seed outcomes
were consulted to make those changes.

The final generation protocol is `fixed-evidence-qwen3-v3-bm25-json`.
E1/E2/E6 include both BF16 and int8 latent arms so channel and precision can
be examined separately. E3 is restricted to the fixed-length, single-document
private-fact generator with zero recomputation; all 128 test documents are
35 Qwen tokens, all answers are unique, and donor pairing has no fixed points.
Synthetic numeric answers use strict six-digit scoring, not prose punctuation
normalization. The actual tensor RoPE check uses a normalized error threshold
because double BF16 rounding is scale-dependent; the initial absolute-error
threshold failed and its log is retained.

The bounded main matrix in `configs/overnight.json` specifies 128 private-fact
examples, 128 update/conflict examples, 64 synthetic multihop examples, 128
questions each for public equal-hop QA comparisons, 64 each for public
recompute/precision sweeps, and 32 each for public hop sweeps. BEAM equal-hop
and update-category runs use all 60 probes from three complete histories;
the expensive hop sweep uses 30 probes, one deterministic hash-ranked probe
per category and conversation. The size choices were made before scoring the
main runs, using development throughput to fit the overnight window.

BEAM documents have no automatic supersession metadata. Its E6 run therefore
tests answers in the update/conflict categories, without duplicating an
identical current-version filter. Only synthetic E6 tests explicit metadata
filtering. BEAM grading uses the released rubrics with a separately timed,
condition-blind local Qwen3 judge; report it as an uncalibrated same-model
evaluation and retain every criterion decision. Never call its scores official
BEAM leaderboard results.
