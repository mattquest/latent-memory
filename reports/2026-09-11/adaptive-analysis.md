# Independent final adaptive analysis — 2026-09-11

The completed test provides no positive evidence for an accuracy or cold-latency advantage from this independent-document relay. Ordinary incremental prefix reuse gives a measurable latency saving with identical behavior. All methods have very low accuracy, so small differences in correct counts do not establish reliable accuracy superiority. This evaluates the implemented untrained controller and cache mechanism, not the original trained latent-memory architecture.

## Verified scope

All 672 unique planned conditions completed: 96 questions × 7 arms, with 32 questions each at 2/3/4 structural hops. The guard returned success. Independent recomputation from pinned answers matches saved EM/F1; every cold component sum reconciles, initial evidence matches across Basic and all iterative arms, and input/source/case verification has no issues. The guarded run took about 22.4 minutes. No prompt, selection, or generation changes followed test outcomes.

## Full test

| Arm | Correct /96 | EM (nominal Wilson 95%) | F1 | Cold p50 / p95 (s) | All support IDs exposed /96 |
| --- | --- | --- | --- | --- | --- |
| Basic BM25 | 1 | 1.04% (0.18–5.67) | 2.37% | 0.394 / 1.355 | 11 |
| Query expansion | 1 | 1.04% (0.18–5.67) | 2.60% | 1.378 / 2.259 | 9 |
| BM25 + Qwen reranker | 3 | 3.12% (1.07–8.79) | 4.52% | 0.749 / 1.521 | 13 |
| Iterative full text | 3 | 3.12% (1.07–8.79) | 4.88% | 2.555 / 4.142 | 17 |
| Exact incremental KV | 3 | 3.12% (1.07–8.79) | 4.88% | 2.179 / 3.550 | 17 |
| Independent relay BF16 | 2 | 2.08% (0.57–7.28) | 3.78% | 2.952 / 4.899 | 13 |
| Independent relay int8 | 2 | 2.08% (0.57–7.28) | 3.78% | 2.987 / 5.032 | 13 |

Cold means an evidence cache built on demand for each query; it includes document tokenization/cache construction, retrieval, all SEARCH/ANSWER decoding, reranker scoring, and final generation. Resident model loading and the shared offline BM25 index are excluded. Old fixed-evidence warm timings are not comparable to this total online measurement. Support exposure means annotated document-ID coverage, not proof that truncation or a model preserved each necessary fact.

## Recovery and regressions

Basic BM25 answered 1 question and failed 95. Query expansion recovered 0/95 and lost 0/1 successes. Reranking, iterative full text, and exact incremental KV each recovered 2/95 and lost 0/1; reranking and the iterative methods recovered different questions. Each independent relay recovered 2/95 but lost the sole basic success (1/1), leaving a net gain of only1 answer. Recovery alone would hide that regression. These are posthoc paired counts with no population recovery interval.

## Cache comparisons

Exact incremental KV and full text have identical actual search-query sequences, document arrival order, action tokens, evidence-token digests, and final output tokens on **96/96 questions**. EM/F1 are identical. The median per-question incremental/text latency ratio is 0.884: an 11.6% saving. The mean paired difference is −367 ms, with a nominal paired-bootstrap 95% interval of [−445, −297] ms. This supports ordinary prefix reuse for this workload.

Independent BF16/int8 relay is slower than exact incremental KV: median paired latency ratios 1.299/1.319; mean differences +717/+743 ms, with 95% intervals[+496,+958]/[+518,+990]ms. Relative to full text, ratios are 1.180/1.198 and mean differences +351/+376 ms, with intervals[+107,+607]/[+125,+642]ms. For each relay versus either text or incremental, 1 question is relay-only correct and 2 are comparator-only correct. The paired EM change is −1.04 percentage points (95%[−4.17,+2.08]); F1 changes −1.10 points (95%[−4.58,+2.57]). The accuracy intervals include zero.

Reranking has 3 correct answers at 0.749 s cold p50, compared with 3 at 2.179 s for incremental KV. The two methods solve different questions (2 unique successes each, 1 shared), so this is a count tie with uncertain accuracy differences, not identical capability. The independent relays have lower observed counts and roughly 3.7–3.8× reranker latency by median paired ratio.

## Controller and context limits

Full text and incremental KV stop on repeated queries in 90/96 cases, invalid actions in 4/96, and ANSWER in 2/96. Each independent relay stops on repeated queries in 88/96, invalid actions in 3/96, and ANSWER in 5/96. Mean executed rounds are 1.94 for text/incremental and 1.82/1.84 for relays; no arm reaches 5 rounds. All iterative arms decode about 33 internal action tokens per question. They are not zero-text latent planners. Repeated-query fallback is an observed controller limitation, not a reason to tune on the final test.

All-support ID exposure remains low (11/96 Basic; 17/96 text/incremental), and only 3/17 text/incremental questions with all support IDs exposed are correct. Both retrieval coverage and answering behavior remain weak; missing-ID coverage and token truncation prevent a clean attribution to reasoning alone. All methods score 0/32 on 4-hop questions.

The largest complete model input/output reservation is 2,836 tokens, with at most 2,550 delivered evidence tokens and 15 documents; there are no context-budget skips. Only 9/96 iterative-text traces exceed the later 1,536-token replay evidence budget. The pressure replay therefore has a small overflow subset and uses queries originally chosen with a larger context.

An independent capacity audit uses all 96 questions and no model outcomes: **complete, untruncated annotated supporting fragments require at most 989 tokens (median 420); applying the 300-token/document cap gives at most 965**. Every annotated support set fits below 1,536. These counts include exact title/text/document wrappers, and exclude system/question/output tokens. The replay can test relevance retention among distractors; it cannot establish necessary-information overflow or a compression advantage. Plain/int8 KV storage does not reduce attention positions.

## Artifacts and uncertainty

[adaptive-analysis.json](adaptive-analysis.json) contains independently recomputed metrics, paired2×2 counts, recovery/regression IDs, query-stop distributions, all 96 trajectory comparisons, full scope checks, and the embedded per-question capacity audit. [adaptive-support-size-audit.json](adaptive-support-size-audit.json) separately preserves every support-document count and exact corpus/question/tokenizer/format-source hashes. Both are analysis artifacts; no benchmark was rerun.

Intervals here use nominal Wilson bounds and an independent 10,000-draw question bootstrap (seed 20260915). They can differ slightly from the runner’s 2,000-draw intervals; constant paired differences have no estimated interval. The 96 balanced questions come from public answerable-development data, selected without outcomes, but can share component facts/articles. The 21,100-paragraph BM25 corpus is the supplied development corpus, not full Wikipedia. These are descriptive uncertainties on this fixed small subset, not general-population or published MuSiQue performance claims.
