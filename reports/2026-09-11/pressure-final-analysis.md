# Final context-pressure analysis

The matched-trace pressure run completed **384/384 conditions on all 96 predeclared questions**, with no failed, missing, duplicate, or retried records. The guard exited successfully after **675.085 seconds** with **15.604 GiB peak sampled process RSS**. No inference was launched or interrupted during this analysis.

The release is `reports/2026-09-11/context-pressure`. `scripts/build_pressure_report.py --require-complete` passed with no verification issues. All 22 released artifacts passed independent packed/unpacked checksum and size checks. The exported question/corpus files were restored under a separate validation directory with exact original names and hashes. No figures are produced by this exporter, so there was no pressure plot to inspect.

## Results

Only **9/96** iterative-text source traces exceeded 1,536 capped evidence tokens. Their source-token count is an exposure criterion, not a statement that a question requires more context. All 96 remain the primary denominator.

| Replay arm | Correct / 96 | EM | Token F1 | Replay p50 / p95 seconds | Correct / 9 overflow | Overflow p50 / p95 seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Relevance-retained original text | 3 / 96 | 3.125% | 4.878% | 0.534 / 1.609 | 0 / 9 | 0.667 / 0.792 |
| Same retained documents, independent KV BF16 | 1 / 96 | 1.042% | 3.159% | 1.407 / 1.821 | 0 / 9 | 1.675 / 1.856 |
| Same retained documents, independent KV int8 | 2 / 96 | 2.083% | 3.680% | 1.441 / 1.848 | 0 / 9 | 1.760 / 1.925 |
| Rolling summary plus recent original text | 4 / 96 | 4.167% | 5.779% | 0.813 / 13.990 | 0 / 9 | 11.102 / 20.786 |

Compared with retained original text, BF16 lost two correct answers and recovered none; int8 lost one and recovered none. The summary recovered one and lost none. The extra summary answer was **outside the nine-question overflow subset**. These are sparse sample differences, not established population superiority or equivalence.

| Paired contrast against retained text | EM difference, percentage points | Empirical example-bootstrap 95% interval | Median paired replay-time ratio | Mean extra replay seconds |
| --- | ---: | ---: | ---: | ---: |
| Independent KV BF16 | −2.083 | [−5.208, 0] | 2.698× | +0.757 |
| Independent KV int8 | −1.042 | [−3.125, 0] | 2.742× | +0.785 |
| Rolling summary | +1.042 | [0, +3.125] | 1.015× | +2.851 |

The intervals are exploratory fixed-subset resampling estimates, unadjusted for multiple comparisons. Their endpoints do not establish a general accuracy advantage. All overflow comparisons have identical zero correctness and no estimated interval. Median paired ratios are not ratios of the marginal medians in the first table.

## What the budgets actually tested

The retained text and both native cache arms used **identical retained IDs, token streams, and per-round retention decisions on all 96 questions**. Independent BM25 recomputation reproduced every saved selection. Every model call respected the 1,536-token evidence allowance; the largest evidence input was 1,531 tokens, and the largest fixed-prompt/evidence/output reservation was 1,744 tokens, below the 8,192-token backend cap.

The source controller and retained-text replay have **identical final output tokens and scores on 96/96 questions**. Retention removed **no annotated support document IDs on any question**. Complete annotated-ID coverage remained 17/96 overall and 2/9 in the overflow subset. Mean retained support-ID coverage was 53.125% overall and 68.519% on overflow. This is annotation coverage, not proof that the 300-token source cap preserved every answer-bearing sentence: 28/96 retained-text/native cases included a truncated document, including 5/9 overflow cases.

Rolling summaries ran on **39/96 questions**, including **30 questions whose source evidence did not exceed 1,536 tokens**. There were 48 summary calls. The policy reserves 384 tokens for memory and wrappers, leaving 1,152 for recent original passages even before a summary exists; that policy explains why summary work can begin before the full evidence allowance overflows. Mean original-support-ID coverage in the recent text was 40.191%, with a further 12.934% of support IDs having appeared in summarized source passages. The latter is provenance and cannot establish that the final generated memory retained those facts.

The separate support-size audit establishes that **all 96 complete annotated support sets fit within 1,536 tokens**, including wrappers and before paragraph truncation: median 420, maximum 989; capped maximum 965. The replay therefore provides no demonstrated necessary-information overflow case. Combined with the unchanged support-ID coverage after retention, it cannot validate a larger effective latent context, a trained compressor, or the original capacity thesis. Its limited conclusion is that this independently built cache relay showed no accuracy/latency benefit over retained text in this matched-trace, cold-cache retention diagnostic.

## Costs and source references

Replay latency charges token reconstruction, retention, newly retained document-cache builds, cache assembly/recomputation, quantization where applicable, summary generation, and the final answer. Component sums exactly matched recorded replay duration for all 384 conditions. BF16 averaged 0.704 seconds of cache construction and 0.563 seconds of assembly/recomputation per question, before final decoding. Int8 added about 0.0175 seconds of quantization on average. Cold per-question construction is an explicit part of this comparison.

The source iterative-text controller averaged 2.023 seconds of planning plus 0.0104 seconds of retrieval, held identical across replay arms for a given question. The casewise replay-plus-source-component accounting sum has p50 values 2.527 / 3.316 / 3.345 / 2.804 seconds for text / BF16 / int8 / summary. This is **not** measured end-to-end adaptive retrieval under pressure: the queries were chosen by the original larger-context controller.

Original basic RAG achieved 1/96, with original end-to-end p50/p95 0.394/1.355 seconds; original iterative text achieved 3/96, at 2.555/4.142 seconds. Both were 0/9 on the same source-defined overflow subset. These references retain their original retrieval settings and cost definitions; they are not extra pressure-arm replications.

## Audit receipts

- [pressure-final-audit.json](pressure-final-audit.json): all-condition token, independent retrieval-ranking, score, support, truncation, identity, budget, and component-cost reconciliation; PASS with zero failures. Produced by CPU-only [audit_pressure_final.py](audit_pressure_final.py), whose own SHA-256 is recorded.
- [pressure-final-extra.json](pressure-final-extra.json): exact source/replay output agreement, coverage and summary-policy counts, plus independent agreement with every exported metric and paired EM delta.
- [pressure-final-release-audit.json](pressure-final-release-audit.json): 22 artifact checksum checks and exact restoration receipts for `adaptive_musique_test.jsonl` and `adaptive_musique_corpus.jsonl`.
- [adaptive-support-size-audit.json](adaptive-support-size-audit.json): outcome-independent support-size audit, including tokenizer, source, and data hashes.
- Released `summary.json`, `metrics.csv`, `pairs.csv`, `tables.md`, `artifact-manifest.json`, and archived run/source/input files remain under `reports/2026-09-11/context-pressure`.
