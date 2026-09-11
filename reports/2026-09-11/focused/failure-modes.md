# Posthoc failure-mode diagnostics

**Snapshot: COMPLETE.** Descriptive diagnostics only; no causal attribution.

## Public E1 by support-document coverage

| Job | Arm | Nominal hops | Support coverage | N | EM | F1 | Selected truncation | UNKNOWN | Output cap |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |

Supporting IDs do not establish that the answer text survived truncation. JSON includes separate truncation strata and intersected document IDs.

## Matched BF16 latent versus direct text


## Synthetic E2: chain length and actual execution

| Job | Arm | Chain length | Nominal hops | Executed evidence hops | N | EM | F1 | UNKNOWN | Output cap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multihop_docs20 | direct_text / bf16 | 10 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | direct_text / bf16 | 10 | 10 | 10 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | direct_text / bf16 | 10 | 3 | 3 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | direct_text / bf16 | 10 | 5 | 5 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / bf16 | 10 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / bf16 | 10 | 10 | 10 | 3 | 0.0% | 0.0% | 33.3% | 0.0% |
| multihop_docs20 | latent / bf16 | 10 | 3 | 3 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / bf16 | 10 | 5 | 5 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / int8 | 10 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / int8 | 10 | 10 | 10 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / int8 | 10 | 3 | 3 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | latent / int8 | 10 | 5 | 5 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | text / bf16 | 10 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs20 | text / bf16 | 10 | 10 | 10 | 3 | 66.7% | 66.7% | 33.3% | 0.0% |
| multihop_docs20 | text / bf16 | 10 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs20 | text / bf16 | 10 | 5 | 5 | 3 | 33.3% | 33.3% | 33.3% | 0.0% |
| multihop_docs6 | direct_text / bf16 | 2 | 1 | 1 | 4 | 25.0% | 25.0% | 0.0% | 0.0% |
| multihop_docs6 | direct_text / bf16 | 2 | 3 | 3 | 4 | 100.0% | 100.0% | 0.0% | 0.0% |
| multihop_docs6 | latent / bf16 | 2 | 1 | 1 | 4 | 25.0% | 25.0% | 0.0% | 0.0% |
| multihop_docs6 | latent / bf16 | 2 | 3 | 3 | 4 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs6 | latent / int8 | 2 | 1 | 1 | 4 | 25.0% | 25.0% | 0.0% | 0.0% |
| multihop_docs6 | latent / int8 | 2 | 3 | 3 | 4 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs6 | text / bf16 | 2 | 1 | 1 | 4 | 25.0% | 25.0% | 0.0% | 0.0% |
| multihop_docs6 | text / bf16 | 2 | 3 | 3 | 4 | 25.0% | 25.0% | 0.0% | 0.0% |
| multihop_docs40 | direct_text / bf16 | 20 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | direct_text / bf16 | 20 | 10 | 10 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | direct_text / bf16 | 20 | 20 | 20 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | direct_text / bf16 | 20 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs40 | direct_text / bf16 | 20 | 5 | 5 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / bf16 | 20 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / bf16 | 20 | 10 | 10 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / bf16 | 20 | 20 | 20 | 3 | 0.0% | 0.0% | 33.3% | 0.0% |
| multihop_docs40 | latent / bf16 | 20 | 3 | 3 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / bf16 | 20 | 5 | 5 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs40 | latent / int8 | 20 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / int8 | 20 | 10 | 10 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / int8 | 20 | 20 | 20 | 3 | 0.0% | 0.0% | 100.0% | 0.0% |
| multihop_docs40 | latent / int8 | 20 | 3 | 3 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | latent / int8 | 20 | 5 | 5 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs40 | text / bf16 | 20 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs40 | text / bf16 | 20 | 10 | 10 | 3 | 33.3% | 33.3% | 33.3% | 0.0% |
| multihop_docs40 | text / bf16 | 20 | 20 | 20 | 3 | 33.3% | 33.3% | 66.7% | 0.0% |
| multihop_docs40 | text / bf16 | 20 | 3 | 3 | 3 | 0.0% | 0.0% | 33.3% | 0.0% |
| multihop_docs40 | text / bf16 | 20 | 5 | 5 | 3 | 33.3% | 33.3% | 33.3% | 0.0% |
| multihop_docs7 | direct_text / bf16 | 3 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs7 | direct_text / bf16 | 3 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs7 | direct_text / bf16 | 3 | 5 | 4 | 3 | 66.7% | 66.7% | 0.0% | 0.0% |
| multihop_docs7 | latent / bf16 | 3 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs7 | latent / bf16 | 3 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs7 | latent / bf16 | 3 | 5 | 4 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs7 | latent / int8 | 3 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs7 | latent / int8 | 3 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs7 | latent / int8 | 3 | 5 | 4 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs7 | text / bf16 | 3 | 1 | 1 | 3 | 0.0% | 0.0% | 0.0% | 0.0% |
| multihop_docs7 | text / bf16 | 3 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs7 | text / bf16 | 3 | 5 | 4 | 3 | 66.7% | 66.7% | 0.0% | 0.0% |
| multihop_docs10 | direct_text / bf16 | 5 | 1 | 1 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | direct_text / bf16 | 5 | 3 | 3 | 3 | 100.0% | 100.0% | 0.0% | 0.0% |
| multihop_docs10 | direct_text / bf16 | 5 | 5 | 5 | 3 | 66.7% | 66.7% | 0.0% | 0.0% |
| multihop_docs10 | latent / bf16 | 5 | 1 | 1 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | latent / bf16 | 5 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | latent / bf16 | 5 | 5 | 5 | 3 | 66.7% | 66.7% | 0.0% | 0.0% |
| multihop_docs10 | latent / int8 | 5 | 1 | 1 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | latent / int8 | 5 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | latent / int8 | 5 | 5 | 5 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | text / bf16 | 5 | 1 | 1 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | text / bf16 | 5 | 3 | 3 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |
| multihop_docs10 | text / bf16 | 5 | 5 | 5 | 3 | 33.3% | 33.3% | 0.0% | 0.0% |

## E4: same questions at all four ratios

| Job | Nominal hops | Ratio | Matched N | EM | F1 | EM difference vs 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpot_ablations | 3 | 0.0 | 32 | 18.8% | 27.5% | 0.0% |
| hotpot_ablations | 3 | 0.1 | 32 | 28.1% | 33.2% | 9.4% |
| hotpot_ablations | 3 | 0.2 | 32 | 25.0% | 30.1% | 6.2% |
| hotpot_ablations | 3 | 1.0 | 32 | 43.8% | 53.9% | 25.0% |
| musique_ablations | 3 | 0.0 | 32 | 9.4% | 13.8% | 0.0% |
| musique_ablations | 3 | 0.1 | 32 | 12.5% | 14.1% | 3.1% |
| musique_ablations | 3 | 0.2 | 32 | 12.5% | 14.1% | 3.1% |
| musique_ablations | 3 | 1.0 | 32 | 18.8% | 18.8% | 9.4% |

Available-arm denominators, incomplete questions, and matched per-question scores are retained in JSON.

## Limits

- Posthoc descriptive diagnostics only; these strata do not establish causal explanations.
- Supporting-document ID coverage does not prove that answer-bearing text survived truncation or was used.
- Selected truncation means evidence_ids intersect truncated_document_ids; unselected truncated documents do not count.
- UNKNOWN is the exact case-insensitive stripped sentinel; prose abstentions are not counted. Output-cap rate concerns final answers only.
- Only successful repeat-zero records enter accuracy denominators; missing conditions and errors are disclosed, not imputed.
- BEAM EM/F1, if present, describe lexical overlap, not its behavioral rubric score.
- Synthetic E2 keeps chain length, nominal budget, and actual executed evidence hops separate; evidence exhaustion is not adaptive-search failure.
- Executed evidence hops is the source record's nonempty evidence-batch count; direct_text still processes its selected evidence in one final prompt, not multiple agent steps.
- E4 paired comparisons require all four ratios and identical ordered evidence IDs for the same question and condition.
