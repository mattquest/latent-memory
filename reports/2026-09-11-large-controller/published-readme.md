# Latent Retrieval Memory

A research harness for testing transformer key/value (KV) cache relay in local
retrieval, with real model inference and measured text/cache baselines.

**Status, September 11, 2026:** real Qwen3-8B, Qwen3-14B and Qwen3.5-122B-A10B
experiments run on Apple Silicon. The repository contains a native MLX backend, controlled comparisons,
dataset preparation, resource guards, and an auditable reporting pipeline.
The original role adapters, trained verifier, zero-text latent planner, compressor,
and production MCP service remain unfinished. A separate adaptive experiment
uses short decoded search actions and compares seven retrieval/cache methods.

## Large-model controller screen: September 11 follow-up

**A stronger model shows useful short-controller behavior, but the full retrieval
system still needs work.** Qwen3.5-122B-A10B recovered one clearly grounded
four-hop answer that its basic-RAG baseline missed. It also improved two answers
to coarse years, while producing an unsupported country/year answer elsewhere.
The unchanged exact-match scorer misses the four-hop recovery because the model
says **“Third”** and the supplied alias is **“third-largest.”** We preserve the
scores and report the semantic evidence separately.

### What ran

All **48 conditions** completed: the same twelve development questions × two
models × basic RAG or iterative text retrieval. This reruns Qwen3-14B BF16 as a
control against the pinned community **Qwen3.5-122B-A10B 5-bit** checkpoint.
The latter has 122B total parameters and 10B active per token; its weights occupy
79.03 GiB on disk. It ran successfully on the 128 GiB M5 Max without changing
system memory limits. The 96-question test split remains unused.
[Official model description](https://huggingface.co/Qwen/Qwen3.5-122B-A10B),
[pinned conversion](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-5bit/tree/958b33bf6418f8c462a5cbec59366ef763cc3069).

Both models use native chat templates with thinking disabled. Controllers receive
32 output tokens and the same seeded sampler (temperature 0.7, top-p 0.8,
top-k 20, presence penalty 1.5 on generated tokens only). A separate greedy
48-token final call receives evidence and answer instructions. Controller outputs
are not substituted for that final answer. Every model call starts with a fresh
native cache; the 122B model's recurrent/attention cache is not passed through the
older KV-relay implementation.

Basic RAG retrieves six BM25 paragraphs once. Iterative retrieval starts with the
same six and can add three per search, up to five rounds and eighteen documents.
Both search the same **21,100-paragraph corpus**. Limits are 300 tokens per
paragraph, 6,144 evidence tokens, and an 8,192-token complete prompt/output
reservation. All questions, including ambiguous and poorly annotated ones, remain
in the denominator. This tests discovery and combination of scattered evidence;
the final relevant passages need not exceed context capacity.

### Measured local results

| Model | Retrieval | Exact match / 12 | Token F1 | Cold median / p95 (s) | New annotated supports | Full annotated support sets / 12 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 14B BF16 | Basic RAG | 0 | 0.0% | 0.85 / 0.97 | 0 | 1 |
| 14B BF16 | Iterative text | 0 | 0.0% | 4.90 / 7.91 | 6 | 1 |
| 122B, 5-bit | Basic RAG | 1 | 12.5% | 1.68 / 5.65 | 0 | 1 |
| 122B, 5-bit | Iterative text | 1 | 19.0% | 8.04 / 13.36 | 8 | 3 |

“Cold” means fresh query/call caches with the model loaded once per job, not a
new model process per question. Query timing includes searches, tokenization,
controller calls, final synthesis and explicit cache cleanup. Load time and the
shared offline BM25 index are separate. The first 122B basic query took 9.66 s;
first-call compilation and weight paging are included and affect its small-sample
p95. These twelve-query percentiles are descriptive, not production latency SLAs.

| Iterative model | Mean retrieval rounds | Repeated-query stops / 12 | Mean input tokens / query | Mean generated tokens / query |
| --- | ---: | ---: | ---: | ---: |
| 14B BF16 | 2.17 | 9 | 4,056 | 28.4 |
| 122B, 5-bit | 2.33 | 6 | 4,427 | 25.6 |

Neither model hit a generation cap, emitted an invalid action, or leaked a search
command into its final answer. The larger model repeats less and finds more
support IDs, but eight additional IDs are not eight correct answers. Different
tokenizers make equal token counts an imperfect measure of cross-model work.
The two benchmark jobs took **4 minutes 17 seconds** including guarded startup;
model download/hash verification and synthetic checks are separate. Peak MLX
allocation was **28.04 GiB for 14B** and **78.97 GiB for 122B**. All guards finished
cleanly and both inference processes exited.

### What actually improved, and what still failed

- **Clear four-hop recovery:** the 122B controller follows the song to Papa Roach,
  finds the band's California location, resolves Veoh to San Diego, then obtains
  San Diego's urban-area rank. All four relevant paragraphs reach the final call
  untruncated. Basic RAG says UNKNOWN; iterative retrieval says “Third.” This is
  a grounded semantic recovery, although unchanged exact match and F1 both score
  it zero against “third-largest.”
- **Better use of initial evidence:** 122B answers 1848 in both arms where 14B
  abstains. The needed Jefferson Davis/war-year facts occur in non-annotated
  initial passages, so support-ID coverage misleadingly reports zero.
- **Partial answers:** iterative 122B gives 2012 for the Lexus sales-date question
  and 1642 for the New Zealand arrival question. These omit the reference's
  month or full date. They are not silently counted as exact successes.
- **Distractor use remains:** 122B copies Medical College of Georgia without
  completing the European migration chain. Its additional 1968 answer comes
  from an Austria distractor, not the target New Zealand abolition relation.
- **Dataset and delivery failures matter:** “Archostemata” is a grounded alternate
  taxonomic rank, and 701+ describes another university where the named professor
  worked. The supplied aliases target different answers. A spouse support
  paragraph lists co-stars without establishing marriage; the Luther→Wittenberg
  sentence is cut off by the 300-token limit. These are distinct from a model
  ignoring a fully delivered answer.

The [all-question evidence audit](reports/2026-09-11-large-controller/qualitative-audit.md)
checks every delivered fragment behind these statements. Its exact witnesses,
query sequences, per-condition verdicts and hashes are available in
[JSON](reports/2026-09-11-large-controller/qualitative-audit.json). This is a posthoc
qualitative audit, not a blinded regrading or a replacement aggregate accuracy
metric. A punctuation-only diagnostic changed none of the 48 exact-match scores.

### Infrastructure-adjusted latency: conditional provider scenarios

Faster inference hardware changes absolute latency, while sequential model calls
still incur repeated first-token delays. We replay the **recorded workload**, not
new model inference, against sourced provider/hardware rates. Coverage is all
**96 conditions and 254 logical model calls from the two controller screens**:
the new 48 conditions above and the prior 48 short/thinking conditions below.
They reuse twelve questions. The original KV-relay/cache experiments are excluded
because an ordinary hosted text API does not implement their cache operations;
posthoc final-answer diagnostics are also excluded.

The table reports **mean seconds per question**, including measured local work
outside model-call windows. Each projected call costs `TTFT + max(G − 1, 0) / TPS`,
where G counts its actual generated reasoning, action and final tokens. The first
token is already in TTFT. These are conditional estimates, not measured hosted
results, hardware-normalized ground truth or confidence intervals.

| Screen / model / method | Measured laptop mean (s) | Alibaba rate scenario (s) | Lambda 4× B200 / SGLang scenario (s) |
| --- | ---: | ---: | ---: |
| Native 14B / basic RAG | 0.85 | 0.40 | — |
| Native 14B / iterative | 5.17 | 1.68 | — |
| Native 122B / basic RAG | 2.37 | 0.99 | 1.18 |
| Native 122B / iterative | 8.11 | 3.46 | 4.17 |
| Earlier 8B / short iterative | 2.49 | 2.72 | — |
| Earlier 8B / thinking iterative | 24.42 | 17.14 | — |
| Earlier 14B / short iterative | 5.19 | 1.77 | — |
| Earlier 14B / thinking iterative | 43.32 | 11.79 | — |

The cached observations pair reported throughput and latency: 43 tok/s with
0.73 s for 8B, 57 with 0.38 s for 14B, and 99 with 0.96 s for 122B. Cached
latency is **assumed to be TTFT** for these scenarios; the page wording also
calls it round-trip time. The 8B performance series lacks a provider label, so
its Alibaba attribution is inferred from the current sole endpoint and is not
established historically. The first two are cached one-week P50-series
averages; the **122B table was crawled last month**, so it is a historical rate
scenario. Exact performance-window dates are unavailable. Lambda's B200 figures
are separate mean TTFT 1.156 s and TPOT 13 ms (76.9 output tok/s per request),
measured with 8,192-token inputs, 1,024-token outputs and concurrency 32. That
workload differs from our short calls. Its benchmark date and exact checkpoint
precision are unspecified.
[8B source](https://openrouter.ai/qwen/qwen3-8b/pricing),
[14B source](https://openrouter.ai/qwen/qwen3-14b-04-28/uptime),
[122B source](https://openrouter.ai/qwen/qwen3.5-122b-a10b/benchmarks),
[Lambda benchmark](https://lambda.ai/inference-models/qwen/qwen3.5-122b-a10b).

The [full adjustment report](reports/2026-09-11-large-controller/provider-latency/provider-latency.md)
retains **all fourteen calibration profiles**, every per-condition calculation,
and question-paired comparisons. Across the four cached OpenRouter 122B profiles,
the decode-rate scenario means are **0.83–1.54 s for basic RAG** and
**2.96–5.74 s for iterative retrieval**. These describe different provider
scenarios, not an uncertainty interval. All six Lambda hardware/engine pairs are
included; applying those loaded-benchmark rates to the saved short calls gives
iterative scenario means of 3.92–26.21 s. Sufficient memory alone does not imply low queueing or
fast first-token service.

OpenRouter's published throughput definitions disagree about whether TTFT is
already included. The appendix therefore also computes the distinct
`max(TTFT, G / TPS)` inclusive-rate proxy per call. For the Alibaba 122B profile,
that gives 0.98 s basic and 3.23 s iterative, versus 0.99 s and 3.46 s above.
It is not a lower bound. No extra prefill charge is added to TTFT, and no
prompt-length scaling law is inferred. EOS termination, forced thinking-phase
closure and work inside the local call wrappers are not separately modeled;
exact old phase control would require a custom server.
[Provider metric definition](https://openrouter.ai/docs/guides/community/for-providers),
[latency guide](https://openrouter.ai/docs/guides/best-practices/latency-and-performance).

These scenarios hold local predictions and trajectories fixed as workload
assumptions. They do not transfer accuracy to a provider with different weights,
precision or decoding. No hosted inference or dollar-cost saving was measured.
The practical signal is that the capable short controller avoids hundreds of
reasoning tokens; it still needs about 3.33 sequential calls per iterative
question, versus one for basic RAG. Improved accuracy must justify that remaining
latency and token cost.

### Interpretation and reproducibility

This supports continuing with a capable, short-action controller: the larger
model completes a useful successive-discovery chain without extended reasoning.
It does **not** establish a reliable accuracy gain over basic RAG across a clean
held-out set, or any latent-memory latency/cost advantage. Better final-answer
specificity, evidence-complete questions, and a frozen semantic scoring policy are
needed before another broader comparison. No additional model run is queued.

The model change also changes architecture, training, tokenizer and precision;
these results cannot isolate parameter count. The current prompts and sampler
differ from the earlier controller screen below, so the two experiments are not
pooled. No trained compressor, latent planner, independent KV relay, corpus-wide
cache ingestion or production amortization was tested here.

Generation used commit `9d52b58` with every executed source and both checkpoint
receipts pinned. The exporter initially rejected MLX-LM 0.31.3's exact RoPE key
rename (`rope_type` to `type`); its narrow validation repair accepts that rename
and still requires the full configuration to match. This changed reporting only;
no condition was rerun. The failed export receipt is preserved. Code changes are merged in [PR #5](https://github.com/mattquest/latent-memory/pull/5).
The final code passes all 446 CPU tests;
the strict exporter verifies all 48 new conditions without loading model weights.

See the [frozen protocol](docs/large-controller-protocol.md),
[execution matrix](configs/large-controller-screen.json), and
[complete report package](reports/2026-09-11-large-controller/README.md).
The package preserves every prompt, sampled/output token ledger, retrieval trace,
source snapshot, normalized dataset, model receipt and resource guard. Verify its
published bytes without loading any model:

```bash
python reports/2026-09-11-large-controller/verify_publication.py
```

Add `--check-current-readme` to also require today's root README to match the
archived publication snapshot. Earlier reports retain their original bytes and
historical conclusions; the preceding controller report's verifier binds its
README at commit `5406c4e` and should be run from that revision.

## Earlier limited controller screen: September 11 follow-up

**Decision: defer the large matrix.** A slightly larger model and longer
controller reasoning found more annotated supporting documents, but did not
produce a useful accuracy/latency tradeoff on this screen. A separate final-prompt
diagnostic exposed instruction interference and improved 8B by one answer;
all four configurations still ended at only 1/12 on that diagnostic. These are
development findings, not evidence against every possible latent-memory design.

The relevant hypothesis is **cheaper successive discovery and combination of
scattered evidence in a large store**. The final supporting documents can fit in
context while discovering them still requires costly searches. This follow-up
therefore screens controller and answer-stage viability before another cache-relay
comparison. It does not require oversized final evidence as a precondition.

### What ran and what was held back

Completed **48 end-to-end iterative-text conditions**, followed by **96 short
final-answer diagnostic generations** on saved evidence. The twelve new MuSiQue
questions contain four each with two, three and four structural hops. Selection
used a fixed structural hash order, excluded every preserved prior question ID
and normalized question match, and did not use observed accuracy. No question was
dropped after inspection. All four configurations answered the same twelve
questions against the same **21,100-paragraph, approximately 2.52-million-token**
corpus. The prepared 96-question test set remains unused.

The checkpoints are official BF16 Qwen3-8B and Qwen3-14B at pinned revisions.
Short controllers allow 32 action tokens. Thinking controllers allow up to 256
reasoning tokens followed by a separate 32-token action. Both use the same seeded
sampler: temperature 0.6, top-p 0.95, top-k 20. Final answers are separate greedy,
nonthinking calls capped at 48 tokens. This is a fresh sampled-controller
comparison; the original study used greedy controllers.

Every job starts with six BM25 paragraphs and can add three per search, with
five retrieval rounds, eighteen documents, 300 tokens per document, 6,144 evidence
tokens and an 8,192-token full input/output reservation. Model loading and the
shared index are offline setup; reported cold query latency includes all online
search, tokenization, controller and final-generation work. The full model/cache
matrix, basic-RAG baseline and held-out test were deliberately deferred.

### Original end-to-end results

| Configuration | Correct / 12 | Token F1 | Cold median / p95 (s) | New annotated supports after initial retrieval, total | Questions with every annotated support |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B, short | 0 / 12 | 0.0% | 2.08 / 4.92 | 5 | 1 / 12 |
| 8B, 256-token reasoning | 0 / 12 | 0.0% | 21.25 / 41.72 | 7 | 3 / 12 |
| 14B, short | 1 / 12 | 9.4% | 4.96 / 7.64 | 6 | 1 / 12 |
| 14B, 256-token reasoning | 1 / 12 | 8.3% | 38.43 / 77.48 | 11 | 4 / 12 |

On matched questions, thinking/short median latency ratios were **9.45× for 8B**
and **8.93× for 14B**, without an exact-match gain. These are medians of paired
ratios, not ratios of the two group medians. The larger short model took 2.20×
the paired latency of the smaller short model for one additional correct answer.
Twelve development questions and one correct answer cannot establish superiority.
No cache-relay accuracy, latency or cost improvement is claimed from this screen.

Mean final annotation coverage rose from 50.0% to 55.6% with 8B thinking and from
52.8% to 66.7% with 14B thinking. The common initial coverage was 36.8%.
The report retains all questions after early stops when plotting support gains;
it does not turn the surviving trajectories into a selected success denominator.

| Configuration | Mean reasoning / action / final tokens per question | Mean retrieval rounds | Repeated-query stops | Invalid-action stops |
| --- | ---: | ---: | ---: | ---: |
| 8B, short | 0 / 29.3 / 1.2 | 1.83 | 10 / 12 | 0 / 12 |
| 8B, reasoning | 594.7 / 38.2 / 2.0 | 2.33 | 7 / 12 | 2 / 12 |
| 14B, short | 0 / 28.5 / 4.2 | 2.25 | 9 / 12 | 0 / 12 |
| 14B, reasoning | 575.8 / 26.2 / 2.1 | 2.25 | 6 / 12 | 4 / 12 |

The reasoning cap was reached in **27/28 8B decisions and 26/27 14B decisions**.
Thus this tests aggressively bounded thinking, not either checkpoint's fully
completed reasoning. No trajectory reached a fifth retrieval round; none hit the
final-answer token cap. Forced phase-closure and terminal-EOS sampling are counted
separately in the raw records. Token work and local runtime are measured cost
proxies; this study does not estimate API dollar savings or energy consumption.

### What the traces reveal

The strongest useful-discovery example is the Papa Roach/Veoh/San Diego chain.
8B thinking finds the fourth annotated support and states the correct urban-area
rank in its controller answer. The protocol then discards that transient payload;
a separate final call returns another search command instead of the rank.
On a Vatican City chain, thinking finds two additional relevant supports and
produces the year, while the final call returns UNKNOWN; the year alone is not
the full date required by the benchmark. These observations separate retrieval
progress from final-answer accuracy. They do not prove that preserving the
controller state would yield a reliable or cheaper system.

Annotation coverage also overstates evidence sufficiency in some cases. For
example, the source's spouse-support paragraph lists Sam Elliott and Katharine
Ross as co-stars without stating that they are married. Other selected chains
omit residence-at-death or citizenship relationships from their labeled supports.
These gaps exist in the original benchmark source and are not normalization
losses. The questions remain in every denominator. Exact question/document IDs,
short excerpts, ambiguities and delivered-text checks are in the
[qualitative audit](docs/controller-scaling-qualitative-audit.md).

The original finals contain five SEARCH commands across the 48 conditions.
These remain incorrect responses; incidental F1 overlap with an answer is not
successful answering. One such overlap explains why the 14B-short F1 slightly
exceeds its exact-match rate. A trailing-space newline variant of the ANSWER
command also failed the pilot parser. A narrow post-pilot parser repair is
recorded separately; the original results are neither rescored nor replaced.

### Final-answer prompt diagnostic

To test instruction interference without rerunning retrieval, every saved
condition received two final calls: the exact original prompt, then the same
prompt with only the system's search-command sentence removed. Document token
order, truncation, question, final suffix, checkpoint and greedy 48-token budget
were unchanged. All **48 original output token sequences and raw responses
reproduced exactly** before accepting their paired ablations. Controller thoughts,
answer payloads and benchmark answers were not added to either prompt.

| Saved evidence from | Original correct / 12 | Final-only system correct / 12 | Original / final-only F1 | Final-stage median, original / final-only (s) |
| --- | ---: | ---: | ---: | ---: |
| 8B, short | 0 | 1 | 0.0% / 12.5% | 0.487 / 0.537 |
| 8B, reasoning | 0 | 1 | 0.0% / 8.3% | 0.596 / 0.645 |
| 14B, short | 1 | 1 | 9.4% / 8.3% | 1.275 / 1.258 |
| 14B, reasoning | 1 | 1 | 8.3% / 8.3% | 1.239 / 1.192 |

The ablation eliminates all five stray SEARCH outputs. All four configurations
then answer the same one question correctly: the Mexican-American War date.
It does not recover the San Diego rank lost at final synthesis. This identifies
some prompt interference without explaining the whole accuracy failure. The F1
decrease for 14B short removes incidental overlap from an invalid search command.

This diagnostic is posthoc and uses already-inspected development evidence.
Its timings measure only final synthesis, in a fixed original-then-ablation order;
they are not revised end-to-end latency measurements. There are twelve questions,
not 48 independent questions. It does not justify promoting the best-looking
condition to a claimed held-out result.

### Conclusion, execution and reproducibility

The screen was useful for locating bottlenecks. It does **not** yet justify a
large cache-method matrix. A further small development pass should first make
final synthesis reliable, test a less truncated controller, and verify that its
questions have text sufficient to support the required relations. A later fair
comparison must include standard RAG, stronger agentic text retrieval and ordinary
causal prefix reuse, charge every model call, and keep failed questions visible.
No additional experiment is queued by this report.

All annotated support sets fit within 794 wrapped tokens on this development
set. That is compatible with testing costly discovery over a larger corpus;
it does not establish benefit from exceeding context or from compressing it.
This public paragraph corpus, small sample, weak grounding annotations, short
trajectories and untrained controller leave the original large-store hypothesis
unresolved. No trained latent query head, compressor or ingestion-precomputed
whole-corpus KV store was tested.

The 48 guarded pipeline conditions took **15 minutes 22 seconds** including job
startup. The 96 final-only generations took **1 minute 56 seconds** including
verification/startup. Model inference was serial on the local Apple Silicon
laptop; sampled process RSS stayed below 15.65 GiB for 8B and 27.91 GiB for 14B,
within the declared 32/40 GiB ceilings. The guards retained RAM, disk, power and
runtime checks. Separate setup included a verified 29.55 GB 14B download taking
about five minutes and synthetic preflights for both models. A diagnostic launch
with a wrong model label failed before loading a model; its receipt is preserved,
and it contributed no generations. No paid API inference was used.

The full suite passes **284 tests** after the post-pilot parser and report-rendering
repairs. The original 48 generation conditions used commit
`d1d4ffa08c65d97b2ba4587c0683566a3840e712`; the final-prompt diagnostic used the
unchanged generation implementation plus its separately pinned diagnostic source.
The reporting-only rendering amendment and subsequent parser repair are archived
with before/after hashes. Future runs of repaired code are not exact replays of
the frozen pilot.

See the [protocol](docs/controller-scaling-protocol.md),
[data selection and hashes](docs/controller-scaling-data.md),
[diagnostic protocol](docs/controller-final-diagnostic.md), and
[complete report package](reports/2026-09-11-controller-scaling/README.md).
The package includes the [all-question table](reports/2026-09-11-controller-scaling/per-question.md),
[paired metrics](reports/2026-09-11-controller-scaling/pairs.csv),
[raw-artifact manifest](reports/2026-09-11-controller-scaling/artifact-manifest.json),
and archived generation sources, data, guards, model receipts and original
outputs. The earlier 5,076-condition study is retained below as a separate result.

## Findings from the September 11 experiment

Completed **5,076 generation conditions plus 288 local rubric judgments**.
The result is an implemented and audited cache-relay prototype, with these findings:

- **Cache information transfer works:** correct caches answer 128/128 private
  fact questions; wrong, zeroed, random, and absent caches each answer 0/128.
- **Ordinary causal prefix reuse saves time:** on 96 adaptive questions it
  preserves every search action and answer, with a median paired latency ratio
  of 0.884 versus full-text replay, about 12% less time.
- **The independent-document relay has no demonstrated accuracy/latency
  advantage here.** On global adaptive retrieval it answers 2/96 correctly,
  versus 3/96 for iterative text and ordinary prefix reuse, and is slower.
  All methods have low accuracy; sparse scores cannot establish equivalence.
- **Storage quantization works, with limits:** int8 and int4 payloads are
  53.125% and 28.125% of BF16. Int8 matches BF16 outputs on the 64 public
  precision examples; int4 loses one correct MuSiQue answer. Attention still
  dequantizes to BF16, and the number of context positions is unchanged.
- **The original long-horizon/context-capacity thesis remains untested.**
  The controller stalls before sustained search. Only 9/96 replay traces
  exceed the smaller evidence budget, and every complete annotated support
  set still fits. These runs cannot establish benefit when necessary evidence
  exceeds context, or rule out that proposed benefit.

The private-fact audit establishes that the implementation can transfer useful
information through a cache. A successful cache audit alone does not establish
an advantage over text. The direct-context baseline measures overhead when the selected evidence fits
in one prompt. It cannot decide the proposed advantage from cheaper adaptive
search over a much larger memory. The adaptive and constrained-context
supplements address parts of that question, with limits stated below.

## What ran

The initial protocol is `fixed-evidence-qwen3-v3-bm25-json`. The original
[`configs/overnight.json`](configs/overnight.json) is preserved; the completed
initial jobs are reproduced by
[`configs/initial-diagnostics.json`](configs/initial-diagnostics.json), and the
focused follow-up by [`configs/focused-ablation.json`](configs/focused-ablation.json). Each run preserves selected evidence, outputs, token IDs, costs, and
conditions, plus a manifest pinning the model, source, and data identities. It uses the official BF16 `Qwen/Qwen3-8B` checkpoint at
`b968826d9c46dd6066d109eabc6255188de91218`, deterministic greedy decoding with
thinking disabled, one resident model, and no trained role adapters.

| Arm | Evidence processing | Intermediate decoded text |
| --- | --- | --- |
| Direct text | Jointly prefill the selected text, then answer | None |
| Text notes | Read each evidence batch and carry a compact JSON fact summary forward | Up to 192 tokens per nonempty batch |
| Latent BF16 | Append independently precomputed document KV, recompute boundary spans, and append a retention instruction | None |
| Latent int8 | Same relay with cache quantization after every nonempty batch | None |

All arms receive the same selected documents at a given evidence budget. The
default batch contains two documents; BEAM and update probes use one. Query-only
BM25 ranks all supplied distractor paragraphs for HotpotQA and MuSiQue, and
synthetic linked facts. BEAM uses precomputed question-only BM25 retrieval over
each complete public conversation history, with 256-token chunks, 32-token
overlap, and at most two chunks from one message in the top 20. Synthetic update
probes preserve source chronology. Gold answers and supporting annotations are
reserved for evaluation; no main job uses an oracle evidence schedule.

The hop sweep increases the evidence budget through 1, 3, 5, 10, and 20 batches.
It does not learn new queries or perform adaptive retrieval. Empty batches are
skipped and actual executed counts are saved. Public QA caps each paragraph at
300 tokens and selected evidence at 6,144 tokens. A flat curve after exhausting
the supplied documents cannot decide the original adaptive-search thesis.

The original plan declared 8,120 conditions across 11 jobs. An explicit budget
amendment retained the first five jobs (3,360 conditions), canceled the original
4,760-condition remainder, and replaced the redundant sweeps with focused
ablations plus the adaptive and context-pressure tests below. Runtime estimates
based on measured note decoding put the original remainder at another 6–9 hours.
The original plan, amendment, and any interrupted next-job records are retained.
No questions were selected by whether a model answered them correctly.

Three synthetic sets contain 128 held-out questions each. The focused linked-fact
sweep uses the same first 16 test questions, grouped only by document count, and
runs each distinct evidence schedule once. Saturated nominal hop settings are
not counted as additional model evaluations.
HotpotQA and MuSiQue each have 128 deterministically selected public development
questions. BEAM has 60 probes across three complete public histories containing
985,192, 937,158, and 826,779 Qwen tokens. The size of the indexed history must
not be confused with the selected evidence presented to a model.

| Dataset (128 questions) | Arm | Exact match | Token F1 | Warm p50 / p95 (s) | Mean host / internal tokens |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | Direct text | 39.06% (50/128) | 47.9% | 0.47 / 0.61 | 3.38 / 0 |
| HotpotQA | Text notes | 37.50% (48/128) | 47.1% | 6.16 / 11.58 | 2.95 / 169.72 |
| HotpotQA | Latent BF16 | 31.25% (40/128) | 39.2% | 1.27 / 1.43 | 2.91 / 0 |
| HotpotQA | Latent int8 | 31.25% (40/128) | 39.9% | 1.28 / 1.47 | 2.95 / 0 |
| MuSiQue | Direct text | 9.38% (12/128) | 11.9% | 0.38 / 0.55 | 1.73 / 0 |
| MuSiQue | Text notes | 14.06% (18/128) | 18.5% | 5.95 / 12.05 | 2.15 / 167.77 |
| MuSiQue | Latent BF16 | 6.25% (8/128) | 10.5% | 1.22 / 1.38 | 2.10 / 0 |
| MuSiQue | Latent int8 | 6.25% (8/128) | 10.1% | 1.25 / 1.40 | 2.05 / 0 |

These E1 conditions deliver six ranked paragraphs in three batches. They test
relay quality and note-decoding overhead. They are not a saturated-context or
adaptive-retrieval experiment. Notes improve MuSiQue accuracy on this sample,
while their total warm latency, dominated by note generation, is about six
seconds at the median.
The independently precomputed relay loses accuracy against notes on both
public QA sets. It does not establish a Pareto improvement.

The BF16 relay minus direct-text EM difference is −7.8 percentage points on
HotpotQA (paired empirical 95% interval −15.6 to −0.8) and −3.1 points on MuSiQue
(−7.0 to approximately 0). Against notes, the differences are −6.3 points on
HotpotQA (−14.1 to +1.6) and −7.8 on MuSiQue (−13.3 to −2.3). These exploratory
intervals describe these fixed subsets and are not adjusted for multiple
comparisons. Equal BF16/int8 correctness on these examples is not proof of
population equivalence.

The same ordered evidence reached direct text and BF16 relay for every paired
public E1 question. The initial selection contained all annotated supporting
documents for 92/128 HotpotQA questions and 34/128 MuSiQue questions. Even in
that MuSiQue stratum, direct text answered 8/34 exactly and notes 9/34. Document
coverage alone does not guarantee that answer-bearing text survived truncation
or that the model can combine it. Detailed support/truncation/abstention strata
are included as descriptive diagnostics, not causal explanations.

The focused E2 sweep completes **212 conditions on 16 questions**, grouped
by available document count. The table shows each group at its largest
distinct evidence budget; the full curves retain every executed schedule.
Groups are tiny (three or four questions), so one answer changes a group
score by 25–33 points. They do not support a population accuracy ranking.

| Documents / questions | Nominal / executed batches | Correct: direct / notes / BF16 / int8 | Warm p50 seconds: direct / notes / BF16 / int8 |
| --- | --- | --- | --- |
| 6 / 4 | 3 / 3 | 4/4 / 1/4 / 0/4 / 0/4 | 0.37 / 11.54 / 1.04 / 1.07 |
| 7 / 3 | 5 / 4 | 2/3 / 2/3 / 0/3 / 1/3 | 0.38 / 17.08 / 1.49 / 1.52 |
| 10 / 3 | 5 / 5 | 2/3 / 1/3 / 2/3 / 1/3 | 0.44 / 18.96 / 2.15 / 2.13 |
| 20 / 3 | 10 / 10 | 0/3 / 2/3 / 0/3 / 0/3 | 0.57 / 41.23 / 7.17 / 7.16 |
| 40 / 3 | 20 / 20 | 0/3 / 1/3 / 0/3 / 0/3 | 0.83 / 87.34 / 28.63 / 28.63 |

On the 40-document group, BF16 relay has correct counts **0, 0, 1, 0, 0**
at 1, 3, 5, 10, and 20 batches; notes have **0, 0, 1, 1, 1**, and direct
text **0, 1, 0, 0, 0**. There is no sustained independent-relay improvement.
The 20-batch note arm averages 2,305 internal decoded tokens; direct text
and the two relays decode none internally. Mean host outputs at that budget
are 6.00 / 2.67 / 4.33 / 1.00 tokens in the same arm order as the table.

Relay time is not constant per batch in this implementation. At every
concatenation, its boundary heuristic revisits the flattened accumulated
block boundaries. The 20-batch BF16/int8 cases each replace 5,456 token
positions cumulatively, while the evidence itself totals 1,606 tokens.
Repeated sparse recomputation and growing context contribute to the cost.
This does not validate the guiding estimate of 100–200 ms per latent hop,
and no optimized production latency claim follows from it.

The documents still fit a direct prompt. These fixed-schedule curves test
progressive evidence delivery and retention; they are not useful adaptive
search trajectories and cannot apply the original long-horizon kill criterion.
Some note conditions solve questions direct text misses, but those small
differences also involve a different intermediate representation.
[All focused tables and curves](reports/2026-09-11/focused/tables.md)
show separate cohorts and exact denominators.

## Adaptive search over a global corpus

The global test completed **672/672 conditions** without errors. All arms
struggle: their exact-match scores range from 1/96 to 3/96. This is a
retrieval/reasoning floor, not an accuracy-saturated benchmark. Small score
differences cannot establish a reliable ranking or a usable system.

| Arm (96 questions each) | Exact match | F1 | Cold p50 / p95 (s) | Mean host / internal tokens |
| --- | --- | --- | --- | --- |
| Basic BM25 RAG | 1.04% (1/96) | 2.37% | 0.39 / 1.36 | 4.7 / 0.0 |
| Query expansion | 1.04% (1/96) | 2.60% | 1.38 / 2.26 | 4.5 / 24.2 |
| BM25 + reranker | 3.12% (3/96) | 4.52% | 0.75 / 1.52 | 5.2 / 0.0 |
| Iterative full text | 3.12% (3/96) | 4.88% | 2.56 / 4.14 | 5.4 / 33.1 |
| Ordinary incremental BF16 KV | 3.12% (3/96) | 4.88% | 2.18 / 3.55 | 5.4 / 33.1 |
| Independent BF16 relay | 2.08% (2/96) | 3.78% | 2.95 / 4.90 | 1.5 / 33.1 |
| Independent int8 relay | 2.08% (2/96) | 3.78% | 2.99 / 5.03 | 1.5 / 32.6 |

Among the **95 basic-RAG failures**, reranking, iterative text, ordinary
incremental KV, and each independent relay recover two questions. The
independent relays also lose the sole question basic RAG answered correctly;
reranking and ordinary incremental/full-text iteration retain it. Expansion
recovers none. These conditional counts accompany the full-test results.

Ordinary incremental KV has the same EM/F1 as iterative text, with a median
per-question latency ratio of **0.884**. The paired mean latency saving is
0.367 seconds (empirical 95% interval 0.297–0.448). This is a useful cache-reuse
result. It does not require independent document relays. Against iterative
text, BF16 independent relay has a median paired latency ratio of **1.180**
and an EM difference of −1.04 points (95% interval −4.17 to +2.08). Against
ordinary incremental KV, its ratio is **1.299**. Reranking achieves 3/96 with
a much lower median cost than any iterative arm; it is still far below
acceptable absolute accuracy.

Ordinary incremental KV and full-text iteration produce identical queries,
actions, retrieved-document trajectories, and final output tokens on **96/96**
questions. This strengthens the interpretation of their timing difference as
prefix-reuse savings in this implementation. The independent relays can change
later search decisions, so their full adaptive runs do not hold all evidence
constant.

The [descriptive round breakdown](reports/2026-09-11/prefix-reuse-by-rounds.json)
is consistent with more reuse at longer trajectories: median incremental/text
latency ratios are 0.984 after one round (n=20), 0.867 after two (n=63),
0.811 after three (n=12), and 0.624 after four (n=1). These are different
controller-selected question groups, not a randomized scaling experiment. The controller usually stops early: full text and ordinary incremental KV
hit the repeated-query fallback on 90/96 questions, invalid-action fallback
on 4/96, and ANSWER on 2/96. Their mean is 1.94 executed retrieval rounds;
no arm reaches five. Both independent relays repeat queries on 88/96. Thus
this run does not test sustained useful 10–20-round adaptive search.

All-support document exposure rises from 11/96 for basic RAG to 17/96 for
text/incremental iteration, but only 3/17 of those fully covered questions
are correct. Each arm scores 0/32 on four-hop questions. Reranking's three
correct answers are all two-hop; text/incremental solve two two-hop and one
three-hop question. Detailed [independent trace analysis](reports/2026-09-11/adaptive-analysis.md)
preserves coverage, truncation, stop counts, and exact output comparisons.

![Adaptive accuracy and cold latency](reports/2026-09-11/adaptive/figures/accuracy_vs_cold_latency.png)

[Full adaptive tables](reports/2026-09-11/adaptive/tables.md) and
[paired comparisons](reports/2026-09-11/adaptive/pairs.csv)
retain the denominators and exploratory uncertainty.

The separate corpus contains 21,100 distinct title/text pairs and 2,518,129 Qwen
tokens, pooled from all supplied paragraphs in the official MuSiQue answerable
development split. Gold-answer fields, question decompositions, and support labels
are excluded from the index and prompts. Six development and 96 held-out local
test questions are disjoint from the original 128 by ID and exact question
text. Test selection uses seed/hash within 2-, 3-, and 4-hop groups (32 each),
without selecting questions by any model's answer. Articles and component facts
can still overlap. This is a global supplied-paragraph corpus, not all Wikipedia.

The seven arms are basic BM25 RAG, query expansion with reciprocal-rank fusion,
BM25 plus Qwen3-Reranker-0.6B, iterative full text, ordinary incremental BF16 KV,
and independent-document BF16/int8 relay. Basic RAG and iterative arms begin
with the same six paragraphs; expansion and reranking can change the six.
Iterative arms add up to three new paragraphs per round, for five rounds and
18 paragraphs at most. Every iterative arm uses the same brief SEARCH/ANSWER
controller and final-answer prompt. This controller decodes search text; it
does not implement the proposed trained latent query projection.

Primary adaptive latency charges all per-question work, including retrieval,
query generation, cache construction, recomputation, and final decoding. Model
loading and the shared index are separate setup. Caches start cold for every
question and arm. This does not measure a persistent whole-corpus KV store
precomputed at ingestion. The initial warm diagnostics are separate tests;
the adaptive construction field combines required assembly/recomputation
with preparation, so subtracting it would not produce a valid warm-query
measurement. Results report the full 96 questions, recovery among basic
RAG failures, and regressions on questions basic RAG got right. The failure-only
slice is a conditional diagnostic, not the headline population.

On average, full-text iteration spends 2.023 seconds in controller inference
(including its repeated evidence prefills) and 0.659 in final generation.
Ordinary incremental KV spends 0.753 seconds constructing/appending its
cache, 1.307 in controller inference, and 0.254 in final generation. BF16
independent relay spends 1.614 seconds on cache construction/assembly and
1.299 in controller inference. These measured components explain the local
costs; they do not predict the speed or accuracy of an unbuilt latent planner.

## Constrained working context

All **384/384 replay conditions** completed. Only **9/96** source traces
actually exceed 1,536 evidence tokens. Every method scores **0/9 EM and F1**
in that subset. Retention removes no annotated supporting-document IDs on
any question, and retained text produces exactly the source iterative-text
output on 96/96. This cohort therefore does not demonstrate an accuracy
penalty from discarding retrieved material.

| Arm | Full-test exact match (n=96) | F1 | Replay p50 / p95 (s) | Overflow p50 / p95 (s), n=9 |
| --- | --- | --- | --- | --- |
| Retained raw text | 3.12% (3/96) | 4.88% | 0.53 / 1.61 | 0.67 / 0.79 |
| Retained BF16 relay | 1.04% (1/96) | 3.16% | 1.41 / 1.82 | 1.67 / 1.86 |
| Retained int8 relay | 2.08% (2/96) | 3.68% | 1.44 / 1.85 | 1.76 / 1.93 |
| Rolling text summary | 4.17% (4/96) | 5.78% | 0.81 / 13.99 | 11.10 / 20.79 |

Mean host/internal token counts are 5.36/0 for text,
1.46/0 for BF16 relay, 1.46/0 for int8, and
5.66/80.58 for the rolling summary. These internal counts exclude
the original source controller, whose decoded work is recorded separately.
BF16/int8 relay have median paired replay-time ratios of 2.698/2.742 versus
retained text. Their paired EM differences are −2.08 points (95% interval
−5.21 to 0) and −1.04 (−3.13 to 0). Summary adds one answer (+1.04 points;
interval 0 to +3.13), with a 2.851-second mean additional cost and a long
latency tail. These sparse, exploratory comparisons do not establish
population superiority; the full [independent audit](reports/2026-09-11/pressure-final-analysis.md)
includes casewise changes and timing definitions.

The rolling summary runs on 39 questions, including 30 whose cumulative
trace fits within 1,536 tokens, because it reserves a 384-token memory slot.
Its one additional correct answer occurs outside the overflow subset. This
is not evidence of a capacity advantage. Raw-text and KV retention have
identical retained token IDs on all 96 questions.

[Full pressure tables](reports/2026-09-11/context-pressure/tables.md) report
original references and separate source costs. Original controller/search
costs average 2.02/0.01 seconds across all questions and 3.11/0.01 in the
overflow subset; adding these to replay cost is accounting, not a measured
end-to-end pressured search run.

The constrained-context supplement replays all iterative-text retrieval traces
with a 1,536-token evidence budget. It compares relevance-retained raw text,
BF16/int8 independent-document caches over the same retained fragments, and a
rolling text summary plus recent evidence. The 384-token summary allowance,
including its wrappers, counts against the evidence budget. Gold supports do
not guide retention. All model calls record evidence, fixed-prompt, and output
reservations separately.

This replay measures retention and interference on matched search traces.
Its queries were chosen by the larger-context controller, so replay latency
is not end-to-end adaptive-search latency under pressure. All questions and
the subset that actually exceed the evidence budget are reported separately.
A separate tokenizer audit finds that all 96 questions' annotated support
fragments fit below the limit: the largest complete set is 989 tokens (median
420), or 965 tokens after the per-document cap. Supporting annotations do not
guarantee sufficient reasoning, but this cohort does not force necessary
evidence to overflow. Neither corpus size nor excess cumulative retrieval
proves that necessary information exceeds context. No trained compressor or larger effective latent context is tested.

The pressure replay builds KV caches cold per question and includes that
construction in its replay timing. It does not measure a persistent cache
store prepared during ingestion.

## Cache dependence, recomputation, and precision

The E3 test uses 128 unseen synthetic private-fact questions with unique
six-digit answers. Every evidence document is exactly 35 Qwen tokens. Donor
pairing has no fixed points, and the receiver prompt contains no answer.
Correct, wrong, zeroed, per-layer/per-KV-head moment-matched random, and absent
caches hold the answer procedure fixed. Correct caches answer **128/128**; wrong,
zeroed, random, and absent caches each answer **0/128**. The correct-cache warm
median is 0.271 seconds (p95 0.276), excluding preparation. No bridge recomputation can restore
the corrupted evidence in this test. Correct-cache answers return six tokens
on average and decode no internal text. The random-cache control hits the
16-token output cap in 124/128 cases; its zero accuracy is retained with that
behavior, rather than treated as a normal abstention.

E4 recomputes contiguous spans around document boundaries at requested ratios
0%, 10%, 20%, and 100%. This boundary heuristic is not CacheBlend's learned or
deviation-based selection procedure. The ratio counts replaced token positions, not a fraction of total FLOPs. At
100%, it performs ordinary joint prefill. Independently computed deep-layer KV does not become joint-context KV
merely by correcting rotary positions.

E5 measures BF16, affine int8, and packed int4 cache storage, including scale
and bias metadata. Quantized caches are dequantized to BF16 before attention;
the test does not demonstrate quantized attention acceleration. E4/E5 assemble the selected evidence once; E1/E2 append retention instructions
and quantize between batches where specified. Their results are separate
conditions, even when a precision or ratio label is the same.

All 128 private lookups remain correct under BF16, int8, and int4. Their small
answers and fixed cache geometry make this a sensitivity check; it does not
establish unchanged reasoning quality on the public tasks.

| Cache storage | Bytes per token | Fraction of BF16 | Arithmetic payload at 1M tokens |
| --- | ---: | ---: | ---: |
| BF16 | 147,456 | 100% | 147.456 GB |
| Packed affine int8 | 78,336 | 53.125% | 78.336 GB |
| Packed affine int4 | 41,472 | 28.125% | 41.472 GB |

The measured payload includes quantization scales and biases. It excludes
weights, allocator/metadata overhead, and dequantized working copies. The
million-token figures are arithmetic extrapolations, not a constructed cache
store. The cache still has one position per input token; smaller byte storage
does not give the model a larger attention context.

The focused public ablations use 32 questions per dataset, with matched
selected evidence within each comparison. E4 gives:

| Dataset | Recomputation ratio | Correct / 32 | Token F1 | Warm p50 / p95 (s) | Mean host / internal tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | 0% | 6 | 27.51% | 0.113 / 0.570 | 3.94 / 0 |
| HotpotQA | 10% | 9 | 33.24% | 0.441 / 0.578 | 2.09 / 0 |
| HotpotQA | 20% | 8 | 30.12% | 0.503 / 0.614 | 2.00 / 0 |
| HotpotQA | 100% | 14 | 53.94% | 0.480 / 0.613 | 2.88 / 0 |
| MuSiQue | 0% | 3 | 13.79% | 0.109 / 0.950 | 4.84 / 0 |
| MuSiQue | 10% | 4 | 14.06% | 0.427 / 0.535 | 1.56 / 0 |
| MuSiQue | 20% | 4 | 14.06% | 0.480 / 0.598 | 1.47 / 0 |
| MuSiQue | 100% | 6 | 18.75% | 0.443 / 0.557 | 1.50 / 0 |

Full joint recomputation has the highest observed accuracy in both subsets.
The 20% heuristic does not show a quality/cost sweet spot: a single full forward
is also faster at the median than the separate partial forwards in these runs.
These small cohorts and output-length differences do not establish that more
recomputation is generally cheaper. There is no demonstrated 20% optimum.

E5 fixes recomputation at 20% and varies stored precision:

| Dataset | Cache precision | Correct / 32 | Token F1 | Warm p50 / p95 (s) | Mean host / internal tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | bf16 | 8 | 30.12% | 0.509 / 0.618 | 2.00 / 0 |
| HotpotQA | int8 | 8 | 30.12% | 0.514 / 0.629 | 2.00 / 0 |
| HotpotQA | int4 | 8 | 28.33% | 0.514 / 0.630 | 1.94 / 0 |
| MuSiQue | bf16 | 4 | 14.06% | 0.480 / 0.595 | 1.47 / 0 |
| MuSiQue | int8 | 4 | 14.06% | 0.486 / 0.607 | 1.47 / 0 |
| MuSiQue | int4 | 3 | 10.94% | 0.487 / 0.603 | 1.47 / 0 |

Int8 produces exactly the BF16 output tokens on all 64 public questions. Int4
changes one output in each dataset: a partially matching HotpotQA answer becomes
UNKNOWN, and one correct MuSiQue answer becomes incorrect. The small timing
differences do not show a quantization speedup. The E4 20% and E5 BF16 conditions
repeat the same 64 predictions; they are executed conditions, not independent
accuracy evidence. [Focused analysis](reports/2026-09-11/focused-final-analysis.md)
contains exact counts, output changes, support/truncation checks, and the
[full curves and tables](reports/2026-09-11/focused/tables.md).

## Updates, conflicts, and BEAM

Synthetic E6 compares keeping every version with filtering records already
marked superseded. Explicit updates and unresolved conflicts are separate
categories. Filtering uses supplied version metadata and does not test a
learned contradiction detector. BEAM has no such metadata; its E6 conditions
are category-specific answering tests only. Its 48 E6 outputs (12 questions ×
four arms) exactly repeat the corresponding E1 predictions, so they are a
category slice, not independent evidence of update handling.

BEAM answers receive a separate, condition-blind local Qwen3 rubric evaluation.
The judge sees the official question, reference answer, rubric, and candidate,
without the arm name or timing. Each criterion decision and malformed output
is retained. These are exploratory scores from an uncalibrated same-model
judge on three conversation clusters, not official BEAM leaderboard scores.
BEAM reference-text exact match/F1 is retained only as a lexical diagnostic.

On the synthetic update/conflict set, direct text and both KV variants answer
**128/128** correctly under both policies. Notes answer **32/128**, entirely from
the 32 conflict questions. Filtering superseded records reduces the BF16 relay
median from 0.571 to 0.393 seconds, with accuracy unchanged. Direct text remains
about 0.293 seconds and already achieves the same perfect score. These three
arms return five tokens on average without internal decoding. Notes return
1.25 tokens and decode 45 internal tokens under all-versions versus 33 under
current-only.

The note failure is specific and inspectable: in all 192 update conditions
(96 questions × two policies), the final note preserves the current six-digit
number but omits the target record identifier, and the final answer is UNKNOWN.
Every update note is valid JSON and 16 tokens, below the 192-token cap. The 64 conflict
conditions preserve the identifier and both values and answer CONFLICT correctly.
This co-occurrence does not prove that the missing ID alone caused the failure.
It exposes a weakness in this particular summary representation, not a
fundamental inability of text to preserve updates or a unique latent advantage.
[Full note analysis](reports/2026-09-11/e6-note-analysis.md) includes every
condition and exact source hashes.

| Arm (60 probes) | Mean fraction of criteria passed | All criteria passed | Warm p50 / p95 (s) | Mean host / internal tokens | Answers reaching 128-token cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct text | 34.86% | 19/60 | 3.14 / 4.74 | 84.67 / 0.00 | 23/60 |
| Text notes | 32.08% | 17/60 | 9.77 / 18.49 | 74.40 / 192.03 | 16/60 |
| Latent BF16 | 35.28% | 18/60 | 4.22 / 5.27 | 91.48 / 0.00 | 27/60 |
| Latent int8 | 34.17% | 17/60 | 4.19 / 5.29 | 90.78 / 0.00 | 25/60 |

The mean criterion fraction weights each candidate equally after scoring its
1–10 criteria; it is not exact-match accuracy. The relay's small descriptive
criterion-score difference from direct text is accompanied by a lower count
of fully passing answers and higher warm latency. Three shared conversation
histories, an uncalibrated same-model judge, and frequent capped answers do
not support a superiority claim. The exported raw judge summary preserves its
naive per-example intervals for audit, but those intervals ignore conversation
clustering and are not used for population claims here.

In the 12-probe E6 category slice, direct text, notes, and BF16 fully pass 3/12;
int8 passes 2/12. The slice repeats E1 outputs and is not another independent
update-handling experiment. Judge inference is separate from the generation
latencies above. All 288 candidate judgments are valid structured outputs;
all judges stopped at EOS. Grading used 105,463 input and 14,007 output tokens,
with 524.99 seconds of measured inference, excluded from answer latency.
Syntactic validity does not establish grading correctness. An independent
review preserves a plausible false-negative grading example without changing
any scores. [BEAM audit](reports/2026-09-11/beam-final-analysis.md) provides
category and conversation breakdowns, caps, separate grading costs, and the
unadjudicated example.

## Costs and reliability

Warm latency includes synchronized cache assembly, position correction,
recomputation, intermediate decoding where applicable, and final decoding.
It excludes model loading, offline retrieval, fixed schedule selection, shared
final-question tokenization, and document-cache preparation.
The recorded preparation cost covers the union of documents used by all cases
for an example; it is not a separately measured cold latency for each arm.
Judge inference is a separate cost. Tables and artifacts report median/p95 latency and generated-token counts;
different output lengths can change timing. These warm fixed-evidence times
are not directly comparable with adaptive cold per-query times or the
pressure replay accounting sum.

The five guarded stages took **2.55 hours** in total, including startup and
shutdown. Downloads, data preparation, development checks, and reporting are
additional. These measurements used an Apple M5 Max with 128 GB unified memory,
macOS 26.4.1, Python 3.12.7, MLX 0.32.2, and MLX-LM 0.31.3.

| Guarded stage | Completed conditions | Wall time (min) | Peak sampled RSS (GiB) |
| --- | ---: | ---: | ---: |
| Initial diagnostics | 3,360 | 84.3 | 15.58 |
| Adaptive retrieval | 672 | 22.4 | 16.88 |
| Context-pressure replay | 384 | 11.3 | 15.60 |
| Focused ablations | 660 | 25.9 | 15.57 |
| Separate rubric judging | 288 | 8.8 | 15.58 |

All amended generation cohorts completed, with **5,076 conditions and zero
recorded generation errors**, plus **288 valid local rubric judgments and zero
invalid judgments**. The original guard was deliberately interrupted at the
budget amendment after five completed jobs; the newly started sixth job
recorded zero completed conditions. It is preserved as excluded partial work.
Peak recorded MLX allocation was **18.38 GiB**, a different measure from sampled
RSS. Across these guards, the lowest sampled available RAM was
**60.73 GiB**; every sample reported AC power.
The [completion and resource ledger](reports/2026-09-11/resource-and-completion.json)
contains each stage's limits, observed resources, and source receipt hashes.

The final code passed **220 tests** in 2.09 seconds. Independent audits check
condition grids, matched evidence, token budgets, source identities, and packed
artifact bytes. Empirical paired intervals in the tables are exploratory;
small counts, shared questions, and multiple comparisons limit interpretation.
BEAM's three conversation clusters are too few for a credible population
interval here, so its headline table reports descriptive scores only.

On the original 22-token real-weight mechanics check, 100% bridge recomputation
matched joint-prefill keys and next-token logits exactly. Causal append selected
the same next token, with maximum logit difference 0.15625. The checked donor
layer-zero keys stayed unchanged. Repositioned layer-zero keys had normalized
RMSE 0.000279, versus 0.080455 without repositioning. The initial absolute-error
criterion failed at 0.125 (target key magnitude 209); after inspecting scale,
rounding, and the unshifted control, the development criterion was revised.
Both receipts are retained. This small revised development check is not broad
proof of equivalent deep-layer cache states or generated answers.

A separate predeclared three-round synthetic preflight compared full-text and
ordinary incremental KV at 201, 258, and 313 retained tokens. Next-token logits
were exactly equal at all three points; every checked source K/V array stayed
unchanged. Independent-document BF16/int8 relay passed geometry and immutability
checks without being required to match joint-attention logits. The reranker
assigned 0.99998 to a relevant toy passage and 0.0000064 to an unrelated one.
The 8,192-token context guards rejected oversized calls. These are small
mechanics checks, separate from benchmark accuracy.

Both new development passes completed: 42 adaptive conditions and 24 pressure
conditions. Raw trace/token/score/cost reconciliation found no discrepancies,
and no prompts changed before the 96-question test. All six adaptive development
questions were missed by every arm; this tiny result was retained, not used to
select easier questions or tune away failures.

A development-only revision replaced truncated Markdown notes with compact
JSON and replaced unweighted overlap ranking with BM25 before the main run.
The stopped development run, corrected pilot, and numerical checks are kept
separate from the main results. The generation prompts were frozen for each declared final run.

The initial public cohorts retain two HotpotQA questions, two MuSiQue questions,
and two BEAM probes used in runtime pilots. The judge pilot reused the same two
BEAM probes, and the public QA overlaps also occur in the focused subsets.
The revised public pilots and initial main jobs have matching executable
source hashes. Excluding the two pilot questions per QA dataset in a descriptive
E1 check leaves 126 questions and the same accuracy ordering; the headline
cohorts remain unchanged. Synthetic development/test and adaptive test96 versus
original MuSiQue128 and adaptive dev6 are disjoint by ID and exact question.
The [overlap audit](reports/2026-09-11/development-overlap-audit.md) preserves
all joins, exact exposed questions, source chronology, and sensitivity results.
These checks cannot assess model pretraining contamination.

## Reproduce and inspect

Install into a repository-local Python 3.12 environment on Apple Silicon. The
weight downloads total about 17.6 GB. `requirements-lock.txt` matches all
71 package versions in the captured experiment environment. The report inputs restore exact normalized
bytes with both compressed and uncompressed SHA-256 verification; conflicting
existing inputs are rejected.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/download_reranker.py
.venv/bin/python scripts/restore_report_data.py --report-dir reports/2026-09-11/initial
.venv/bin/python scripts/restore_report_data.py --report-dir reports/2026-09-11/focused
.venv/bin/python scripts/restore_report_data.py --report-dir reports/2026-09-11/adaptive
```

Run jobs serially in new output directories. These commands reproduce the
retained conditions, not the canceled original remainder. Example outer limits
are safety ceilings; measured runtime depends on hardware and output lengths.

```sh
.venv/bin/python scripts/run_guarded.py --output runs/reproduce-initial --max-seconds 10800 -- \
  .venv/bin/python scripts/run_matrix.py --matrix configs/initial-diagnostics.json \
  --output-dir runs/reproduce-initial --max-seconds 10700
.venv/bin/python scripts/run_guarded.py --output runs/reproduce-focused --max-seconds 7200 -- \
  .venv/bin/python scripts/run_matrix.py --matrix configs/focused-ablation.json \
  --output-dir runs/reproduce-focused --max-seconds 7100
.venv/bin/python scripts/run_guarded.py --output runs/reproduce-adaptive --max-seconds 5400 -- \
  .venv/bin/python scripts/run_adaptive.py --model models/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --reranker models/Qwen3-Reranker-0.6B --dataset data/adaptive_musique_test.jsonl \
  --output-dir runs/reproduce-adaptive --limit 96 --rounds 5 --max-runtime-seconds 5300
.venv/bin/python scripts/run_guarded.py --output runs/reproduce-pressure --max-seconds 7200 -- \
  .venv/bin/python scripts/run_context_pressure.py --source-dir runs/reproduce-adaptive \
  --output-dir runs/reproduce-pressure --max-runtime-seconds 7100
.venv/bin/python scripts/run_guarded.py --output runs/reproduce-judge --max-seconds 7200 -- \
  .venv/bin/python scripts/judge_beam.py --model models/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 --dataset data/beam_1m_retrieved.jsonl \
  --results runs/reproduce-initial/beam_equal_updates/results.jsonl \
  --output-dir runs/reproduce-judge --max-tokens 192 --max-runtime-seconds 7100
```

Current code includes a deterministic BM25 accumulation fix and a retrieval
timing sidecar added after the first five jobs. Historical executable bytes
are archived with their generation hashes. The adaptive input's 102 initial
top-20 rankings were unchanged by the deterministic fix. A rerun on current
code is a new measurement; timing and numerical output can vary.

The [artifact index](reports/2026-09-11/README.md) links raw predictions, exact
inputs, source snapshots, metrics, paired comparisons, plots, local judge
outputs, resource logs, and [excluded validation history](reports/2026-09-11/validation-history/README.md).
[Data preparation and attribution](docs/data-and-protocol.md) explain how to
rebuild from the original public sources. The [adaptive protocol](docs/adaptive-protocol.md)
and [budget amendment](docs/budget-amendment.md) define the additional tests.
Generate tables and figures from saved outputs without rerunning inference:

```sh
.venv/bin/python scripts/build_report.py --run-root runs/reproduce-initial \
  --matrix configs/initial-diagnostics.json --judge-dir runs/reproduce-judge \
  --output-dir reports/reproduced-initial --require-complete
.venv/bin/python scripts/analyze_failure_modes.py --run-dir runs/reproduce-initial \
  --output-dir reports/reproduced-initial --require-complete
.venv/bin/python scripts/build_report.py --run-root runs/reproduce-focused \
  --matrix configs/focused-ablation.json --judge-dir runs/no-focused-judge \
  --output-dir reports/reproduced-focused --require-complete
.venv/bin/python scripts/analyze_failure_modes.py --run-dir runs/reproduce-focused \
  --output-dir reports/reproduced-focused --require-complete
.venv/bin/python scripts/build_adaptive_report.py --run-dir runs/reproduce-adaptive \
  --output-dir reports/reproduced-adaptive --require-complete
.venv/bin/python scripts/build_pressure_report.py --run-dir runs/reproduce-pressure \
  --output-dir reports/reproduced-pressure --require-complete
```

The fixed-evidence exporter's completeness flag verifies generation; its
separate `judge_status` must also be complete before using BEAM grades.
Exporters reject destinations that overlap protected source directories.

The runs use a repository-local environment and one guarded model process.
The guard polls process RSS, available RAM, free disk, power, and reported
thermal status every 5–10 seconds in these runs. It terminates a run on
RSS above 32 GiB, available RAM below 12 GiB, free disk below 40 GiB, battery
power, severe reported CPU throttling, or its wall-time limit. These are sampled
stopping thresholds, not guarantees against brief between-sample excursions. It prevents concurrent guarded runs in this workspace
and releases its sleep inhibition when the child process ends. No paid model
API, personal laptop documents, administrator commands, or system memory
tuning were used.

## Repository map

- `engine/backends_mlx.py`: native Qwen3 cache storage, RoPE correction, append,
  boundary recomputation, perturbations, and packed precision variants.
- `eval/real_experiments.py`: real answer-scored E1–E6 harness.
- `eval/local_rubric_judge.py`: separate local BEAM rubric evaluation.
- `eval/datasets.py`, `eval/retrieval.py`: normalized data and query-only BM25.
- `scripts/`: bounded download/preparation, guarded runs, matrix, and reports.
- `relay/`, `engine/backends.py`, `agents/`: legacy NumPy mock mechanics and
  pipeline scaffolds; real-model use of the mock agent paths fails explicitly.
- `store/`: metadata and legacy block-store components, not a demonstrated
  million-token native MLX cache service.
- `server/`, `train/`: unfinished service and training work.

## Sources and next experiments

The [original guiding document](https://docs.google.com/document/d/1_Ug_QCTOmf-D1QDMl8qOVvevATzU7QZtuQLed260hJ4/edit)
is preserved in [`docs/guiding-plan.txt`](docs/guiding-plan.txt).
[`docs/data-and-protocol.md`](docs/data-and-protocol.md) records source revisions,
licenses, selection rules, and the scope of the cited research. This run is an
independent implementation and does not reproduce the full LatentMAS,
TurboRAG, CacheBlend, or causal-audit training/evaluation setups.

The implemented independent-document relay has not demonstrated the proposed
accuracy/latency advantage. Cache transfer works, and ordinary prefix reuse
saves time, but the current untrained controller rarely performs sustained
search. The fixed-evidence comparisons and the nine-trace pressure subset
cannot settle the original long-horizon hypothesis. A product-readiness or
research-breakthrough claim would be premature.

The next useful experiment should first improve the retrieval/controller on
a separate development set: use a stronger first-stage retriever, diagnose
query repetition, and verify that added rounds expose new supporting facts
and increase answer accuracy. Keep a strong text baseline and ordinary prefix
reuse. Then freeze a fresh test set and compare accuracy at equal total
latency and equal retrieved evidence, with genuinely useful longer trajectories.

A separate capacity claim requires tasks whose relevant evidence or retained
state actually stresses the chosen working budget—such as distributed
aggregation or temporal reconciliation—plus a representation that compresses
attention positions. Add distractors and corpus size as controlled variables,
and measure retained support as well as final accuracy. Changing KV storage
precision alone cannot establish that capability. Any new controller,
compressor, or learned latent query head needs a fresh causal cache audit and
held-out evaluation before crediting latent communication for improvements.
