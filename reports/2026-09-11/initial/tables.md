# Experiment results

Generated 2026-09-11T09:25:26.231093+00:00. Status: **generation_complete**.

These are bounded local Qwen3 experiments. They do not establish the original adaptive multi-agent or BEAM-10M thesis. Incomplete jobs remain visible in receipts and CSVs; headline tables and figures use completed jobs only unless explicitly requested.

## Completion

Completion is scoped to the **5 reported jobs / 3,360 planned conditions** below. The amended original matrix planned 8,120 conditions: 3,360 initial conditions were completed and 4,760 remaining original conditions were canceled. The original resource guard was intentionally interrupted at this controlled boundary; its stopped status does not imply failure of the completed jobs.

Amendment reason: Deadline and relevance amendment: prioritize adaptive global retrieval and context pressure; remove redundant exhausted-context sweeps.

Unused partial runs are excluded from these results and preserved separately: `runs/original-interrupted-multihop`.

| Job | State | Successful / planned | Unresolved errors |
| --- | --- | --- | --- |
| beam_equal_updates | complete | 288 / 288 | 0 |
| hotpot_equal | complete | 512 / 512 | 0 |
| musique_equal | complete | 512 / 512 | 0 |
| private_audit_precision | complete | 1024 / 1024 | 0 |
| updates | complete | 1024 / 1024 | 0 |

## E1

| Dataset / job | Arm / KV | Setting | n | EM | F1 | Warm p50 / p95 (s) | Host / internal tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa / hotpot_equal | direct_text/bf16 | h=3; r=0.2; correct; all | 128 | 39.1% | 47.9% | 0.47 / 0.61 | 3.4 / 0.0 |
| hotpotqa / hotpot_equal | latent/bf16 | h=3; r=0.2; correct; all | 128 | 31.2% | 39.2% | 1.27 / 1.43 | 2.9 / 0.0 |
| hotpotqa / hotpot_equal | latent/int8 | h=3; r=0.2; correct; all | 128 | 31.2% | 39.9% | 1.28 / 1.47 | 3.0 / 0.0 |
| hotpotqa / hotpot_equal | text/bf16 | h=3; r=0.2; correct; all | 128 | 37.5% | 47.1% | 6.16 / 11.58 | 3.0 / 169.7 |
| musique / musique_equal | direct_text/bf16 | h=3; r=0.2; correct; all | 128 | 9.4% | 11.9% | 0.38 / 0.55 | 1.7 / 0.0 |
| musique / musique_equal | latent/bf16 | h=3; r=0.2; correct; all | 128 | 6.2% | 10.5% | 1.22 / 1.38 | 2.1 / 0.0 |
| musique / musique_equal | latent/int8 | h=3; r=0.2; correct; all | 128 | 6.2% | 10.1% | 1.25 / 1.40 | 2.1 / 0.0 |
| musique / musique_equal | text/bf16 | h=3; r=0.2; correct; all | 128 | 14.1% | 18.5% | 5.95 / 12.05 | 2.1 / 167.8 |

## E3

| Dataset / job | Arm / KV | Setting | n | EM | F1 | Warm p50 / p95 (s) | Host / internal tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_private_fact / private_audit_precision | latent/bf16 | h=1; r=0; correct; all | 128 | 100.0% | 100.0% | 0.27 / 0.28 | 6.0 / 0.0 |
| synthetic_private_fact / private_audit_precision | latent/bf16 | h=1; r=0; mismatched; all | 128 | 0.0% | 0.0% | 0.10 / 0.10 | 1.0 / 0.0 |
| synthetic_private_fact / private_audit_precision | latent/bf16 | h=1; r=0; no_context; all | 128 | 0.0% | 0.0% | 0.10 / 0.10 | 1.0 / 0.0 |
| synthetic_private_fact / private_audit_precision | latent/bf16 | h=1; r=0; random; all | 128 | 0.0% | 0.0% | 0.59 / 0.60 | 15.8 / 0.0 |
| synthetic_private_fact / private_audit_precision | latent/bf16 | h=1; r=0; zero; all | 128 | 0.0% | 0.0% | 0.10 / 0.10 | 1.0 / 0.0 |

## E5

| Dataset / job | Arm / KV | Setting | n | EM | F1 | Warm p50 / p95 (s) | Host / internal tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_private_fact / private_audit_precision | latent/bf16 | h=1; r=0; correct; all | 128 | 100.0% | 100.0% | 0.27 / 0.28 | 6.0 / 0.0 |
| synthetic_private_fact / private_audit_precision | latent/int4 | h=1; r=0; correct; all | 128 | 100.0% | 100.0% | 0.27 / 0.28 | 6.0 / 0.0 |
| synthetic_private_fact / private_audit_precision | latent/int8 | h=1; r=0; correct; all | 128 | 100.0% | 100.0% | 0.27 / 0.28 | 6.0 / 0.0 |

## E6

| Dataset / job | Arm / KV | Setting | n | EM | F1 | Warm p50 / p95 (s) | Host / internal tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_updates / updates | direct_text/bf16 | h=2; r=0.2; correct; all | 128 | 100.0% | 100.0% | 0.29 / 0.30 | 5.0 / 0.0 |
| synthetic_updates / updates | direct_text/bf16 | h=2; r=0.2; correct; current | 128 | 100.0% | 100.0% | 0.29 / 0.30 | 5.0 / 0.0 |
| synthetic_updates / updates | latent/bf16 | h=2; r=0.2; correct; all | 128 | 100.0% | 100.0% | 0.57 / 0.58 | 5.0 / 0.0 |
| synthetic_updates / updates | latent/bf16 | h=2; r=0.2; correct; current | 128 | 100.0% | 100.0% | 0.39 / 0.44 | 5.0 / 0.0 |
| synthetic_updates / updates | latent/int8 | h=2; r=0.2; correct; all | 128 | 100.0% | 100.0% | 0.58 / 0.59 | 5.0 / 0.0 |
| synthetic_updates / updates | latent/int8 | h=2; r=0.2; correct; current | 128 | 100.0% | 100.0% | 0.40 / 0.45 | 5.0 / 0.0 |
| synthetic_updates / updates | text/bf16 | h=2; r=0.2; correct; all | 128 | 25.0% | 25.0% | 1.40 / 3.23 | 1.2 / 45.0 |
| synthetic_updates / updates | text/bf16 | h=2; r=0.2; correct; current | 128 | 25.0% | 25.0% | 0.75 / 3.24 | 1.2 / 33.0 |

## E6 by category

| Dataset | Category | Arm / KV | Policy | n | EM |
| --- | --- | --- | --- | --- | --- |
| synthetic_updates | contradiction_resolution | direct_text/bf16 | all | 32 | 100.0% |
| synthetic_updates | knowledge_update | direct_text/bf16 | all | 96 | 100.0% |
| synthetic_updates | contradiction_resolution | direct_text/bf16 | current | 32 | 100.0% |
| synthetic_updates | knowledge_update | direct_text/bf16 | current | 96 | 100.0% |
| synthetic_updates | contradiction_resolution | latent/bf16 | all | 32 | 100.0% |
| synthetic_updates | knowledge_update | latent/bf16 | all | 96 | 100.0% |
| synthetic_updates | contradiction_resolution | latent/bf16 | current | 32 | 100.0% |
| synthetic_updates | knowledge_update | latent/bf16 | current | 96 | 100.0% |
| synthetic_updates | contradiction_resolution | latent/int8 | all | 32 | 100.0% |
| synthetic_updates | knowledge_update | latent/int8 | all | 96 | 100.0% |
| synthetic_updates | contradiction_resolution | latent/int8 | current | 32 | 100.0% |
| synthetic_updates | knowledge_update | latent/int8 | current | 96 | 100.0% |
| synthetic_updates | contradiction_resolution | text/bf16 | all | 32 | 100.0% |
| synthetic_updates | knowledge_update | text/bf16 | all | 96 | 0.0% |
| synthetic_updates | contradiction_resolution | text/bf16 | current | 32 | 100.0% |
| synthetic_updates | knowledge_update | text/bf16 | current | 96 | 0.0% |

## E6 paired version filtering

| Dataset | Arm / KV / control | n paired | Current − all EM (percentage points) | Current-only / all-only correct |
| --- | --- | --- | --- | --- |
| synthetic_updates | direct_text/bf16/correct | 128 | +0.0 | 0 / 0 |
| synthetic_updates | latent/bf16/correct | 128 | +0.0 | 0 / 0 |
| synthetic_updates | latent/int8/correct | 128 | +0.0 | 0 / 0 |
| synthetic_updates | text/bf16/correct | 128 | +0.0 | 0 / 0 |

## BEAM: exploratory local rubric judging

BEAM EM/F1 are lexical proxies and are excluded from the accuracy tables above. The optional judge uses the same model family and is not the official BEAM evaluator. Questions are clustered within selected conversations; no population confidence intervals are reported. Judge inference cost is separate from answer generation.

| Job / experiment | Arm / KV | Hops | Judging state | Valid / candidates | Invalid / pending | Conversations | Criterion fraction | All criteria pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beam_equal_updates / E1 | direct_text/bf16 | 3 | complete | 60 / 60 | 0 / 0 | 3 | 34.9% | 31.7% |
| beam_equal_updates / E1 | latent/bf16 | 3 | complete | 60 / 60 | 0 / 0 | 3 | 35.3% | 30.0% |
| beam_equal_updates / E1 | latent/int8 | 3 | complete | 60 / 60 | 0 / 0 | 3 | 34.2% | 28.3% |
| beam_equal_updates / E1 | text/bf16 | 3 | complete | 60 / 60 | 0 / 0 | 3 | 32.1% | 28.3% |
| beam_equal_updates / E6 | direct_text/bf16 | 3 | complete | 12 / 12 | 0 / 0 | 3 | 25.0% | 25.0% |
| beam_equal_updates / E6 | latent/bf16 | 3 | complete | 12 / 12 | 0 / 0 | 3 | 25.0% | 25.0% |
| beam_equal_updates / E6 | latent/int8 | 3 | complete | 12 / 12 | 0 / 0 | 3 | 16.7% | 16.7% |
| beam_equal_updates / E6 | text/bf16 | 3 | complete | 12 / 12 | 0 / 0 | 3 | 29.2% | 25.0% |

## Interpretation and costs

- E2 changes available evidence through a fixed schedule; it does not measure learned adaptive retrieval. Inspect the direct-text curve before attributing a gain to avoiding evidence-note compression. Structural jobs and chain-length strata have separate figures; connected curves contain identical example cohorts with exact n labels. No changing-cohort aggregate curve is plotted.
- BF16 and int8 latent variants are separate. E5 measures a single storage quantization; multi-hop arms may quantize repeatedly. Attention runs in BF16 after dequantization.
- Warm latency excludes model loading, document-cache precomputation, fixed retrieval/schedule selection, and shared final-question tokenization. The precompute measurement covers the union of documents selected across the experiment grid for an example, not a per-case cold-query latency or full BEAM ingestion.
- Retrieval and schedule timing, truncation, cap-hit rates, cache bytes, and executed hops are in metrics.csv. fraction_document_truncation is the fraction of examples with at least one delivered truncated document; fraction_any_candidate_truncation also counts unused candidates. Headline hop counts are scheduled batches; empty batches do not create new evidence.
- mean_support_recall measures delivered annotation-ID coverage. For synthetic updates, annotations include both old and current versions, so correct current-only filtering can lower this value without losing the required current fact.
- Paired comparisons use common example IDs within a job. Constant-difference bootstrap intervals are left blank rather than implying certainty. Small-subset findings remain exploratory.

## Audit artifacts

Raw prediction ledgers and exact normalized inputs are gzip-compressed without altering their uncompressed bytes. Normalized inputs are verified against generation-manifest hashes; the three normalized BEAM corpora and data license notes are included. Exact executable sources are archived under artifacts/source/ by source-set digest, verified against the executed hashes. When current files differ, matching bytes are recovered from a recorded Git commit; unavailable or mismatching bytes fail the complete-release check. Generation manifests, available summaries/checkpoints, judge ledgers, and resource/data receipts are under artifacts/. artifact-manifest.json records SHA-256 hashes. These retain failures and partial trailing records; analysis excludes malformed records and lists them in summary.json.

![e1_hotpot_equal](figures/e1_hotpot_equal.png)
![e1_musique_equal](figures/e1_musique_equal.png)
![e3_private_audit_precision](figures/e3_private_audit_precision.png)
![e5_private_audit_precision](figures/e5_private_audit_precision.png)
![beam_judge_beam_equal_updates_e1](figures/beam_judge_beam_equal_updates_e1.png)
![beam_judge_beam_equal_updates_e6](figures/beam_judge_beam_equal_updates_e6.png)
