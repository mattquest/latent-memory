# Independent final BEAM audit

All 288 candidates and 288 judgments are unique, complete, correctly matched, and syntactically valid. No invalid attempts, retries, duplicate rows, or integrity failures occurred. The guard completed in 529.14 seconds with 15.575 GiB peak sampled process RSS.

The audit reconstructed the exact planned conditions, candidate digest and shuffled order; checked pinned questions, answers, categories, source hashes and backend identity; reparsed every criterion vector; and verified every prompt/output token count with the pinned CPU tokenizer. No model inference or grade changes were performed.

These are exploratory grades from the same Qwen3-8B generator, not official BEAM scores. There are only three conversation clusters and 60 questions; no population confidence intervals are reported.

| Slice and arm | Valid n | All criteria passed | Mean criterion fraction | Answer cap hits | Warm answer p50/p95 (s) | Judge p50/p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| E1/direct_text_bf16 | 60 | 19/60 (31.7%) | 34.86% | 23 | 3.14/4.74 | 1.79/2.59 |
| E1/latent_bf16 | 60 | 18/60 (30.0%) | 35.28% | 27 | 4.22/5.27 | 1.72/2.58 |
| E1/latent_int8 | 60 | 17/60 (28.3%) | 34.17% | 25 | 4.19/5.29 | 1.74/2.46 |
| E1/text_bf16 | 60 | 17/60 (28.3%) | 32.08% | 16 | 9.77/18.49 | 1.75/2.53 |
| E6/direct_text_bf16 | 12 | 3/12 (25.0%) | 25.00% | 0 | 2.68/4.21 | 1.72/2.20 |
| E6/latent_bf16 | 12 | 3/12 (25.0%) | 25.00% | 3 | 3.48/5.20 | 1.66/2.18 |
| E6/latent_int8 | 12 | 2/12 (16.7%) | 16.67% | 3 | 3.46/5.23 | 1.67/1.91 |
| E6/text_bf16 | 12 | 3/12 (25.0%) | 29.17% | 0 | 7.22/10.67 | 1.64/1.87 |

The criterion fraction is the mean of per-question fractions, so longer rubrics do not receive extra weight. Each E1 arm has 20 questions per conversation and six per category. Each E6 arm has four per conversation and six per category. Micro-averages, every category/conversation denominator, and all paired comparisons are in the JSON.

| E1 paired comparison (A minus B) | n | A only passes | B only passes | All-pass difference | Criterion-fraction difference |
|---|---:|---:|---:|---:|---:|
| direct_text_bf16 − latent_bf16 | 60 | 4 | 3 | +1.7 pp | -0.42 pp |
| direct_text_bf16 − latent_int8 | 60 | 5 | 3 | +3.3 pp | +0.69 pp |
| direct_text_bf16 − text_bf16 | 60 | 5 | 3 | +3.3 pp | +2.78 pp |
| latent_bf16 − latent_int8 | 60 | 2 | 1 | +1.7 pp | +1.11 pp |
| latent_bf16 − text_bf16 | 60 | 6 | 5 | +1.7 pp | +3.19 pp |
| latent_int8 − text_bf16 | 60 | 6 | 6 | +0.0 pp | +2.08 pp |

The BF16 relay passes one fewer complete rubric than direct text, despite a 0.42 percentage-point higher mean criterion fraction; neither establishes superiority. BF16 versus int8 differs by one fully passing question, with two BF16-only passes and one int8-only pass. All four arms fail every contradiction-resolution and event-ordering full rubric (six questions per category per arm).

All 48 E6 candidates, judge inputs, raw judgments, and criterion vectors exactly match their E1 counterparts. E6 is a repeated update/contradiction category slice with no version-filter intervention. It adds no independent update-policy evidence.

Judge inference consumed 105,463 prompt tokens and generated 14,007 tokens in 524.99 seconds. All 288 judgments stopped at EOS; none hit the 192-token judge cap. The largest prompt plus reserved output was 1,184 tokens, below 8,192. No invalid-output retry path was exercised; the implementation would preserve invalid rows and skip already attempted IDs on resume, so a successful guard alone would not prove every grade valid.

Answer generation separately produced 23,706 host tokens and 13,277 decoded text-note tokens. All 288 warm executions totaled 1441.56 seconds. Shared independent-document cache construction totaled 23.86 seconds, counted once across 60 original questions. The offline BM25 query costs totaled 0.082 seconds, excluding index construction.

Warm answer timing includes condition-specific text-note generation or cache assembly/bridge/quantization and final decoding. It excludes shared cache construction, corpus retrieval/indexing, initial tokenization and scheduling. Judge timing measures prompt prefill and grade decoding; it excludes prompt formatting/tokenization, parsing, file writes and model load. These costs remain separate.

The 128-token answer cap is a material limit, especially for detailed summaries and ten-item requests. Cap subgroup grades are descriptive and cannot isolate the causal effect of truncation. Only three BM25-ranked conversation chunks reached each answer, so this does not evaluate million-token histories held in native memory.

A qualitative sample across ten categories found a plausible judge false negative: judgment f8bf3e60441b602116f17ce4 marks every event-ordering criterion false although the answer mentions translation API integration and error handling across items 1 and 4. This remains unadjudicated; original grades are preserved. Some rubric criteria also specify form without exact factual content, and the judge sees no original conversation. Valid JSON and rubric passing therefore do not establish calibrated factual accuracy.

The companion JSON preserves exact source hashes, all category/conversation grades, paired wins and losses, cap subgroups, token reconciliation, timing definitions, and the full integrity-check result.
