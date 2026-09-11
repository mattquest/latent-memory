# Development and pilot overlap audit

The preserved model ledgers show two pilot questions in each initial public QA cohort (2/128 HotpotQA and 2/128 MuSiQue), the same questions in each declared focused public cohort (2/32), and two pilot probes in the initial BEAM cohort (2/60). The BEAM judge pilot reused these same two probes; its eight judgments are eight candidate responses, not eight distinct questions. All other pilot-to-target joins are empty.

No model runs, generation changes, dataset changes, or headline-cohort changes were made for this audit. Detailed IDs, exact question strings, source/data hashes, archive receipts, and every overlap join appear in [development-overlap-audit.json](development-overlap-audit.json).

## Scope and method

The audit reads the preserved validation-history manifest and verifies packed and unpacked bytes for the archived ledgers/manifests. The ten observed prediction/judge ledgers also match their current raw files byte for byte. Conditions are deduplicated to questions before joining by both example ID and exact question text. Initial membership comes from the completed five-job run; focused membership comes from its declared config and input prefixes. The focused run subsequently completed all 660 declared conditions. Adaptive membership uses all 96 declared test questions. Full synthetic development32/test128 files are checked independently of which development questions were actually executed.

| Preserved ledger | Records | Distinct questions |
|---|---:|---:|
| `adaptive-dev/results.jsonl` | 42 | 6 |
| `context-pressure-dev/results.jsonl` | 24 | 6 |
| `development-v2/beam/results.jsonl` | 8 | 2 |
| `development-v2/hotpot/results.jsonl` | 22 | 2 |
| `development-v2/multihop/results.jsonl` | 38 | 2 |
| `development-v2/musique/results.jsonl` | 22 | 2 |
| `development-v2/updates/results.jsonl` | 32 | 4 |
| `development/multihop/results.jsonl` | 86 | 3 |
| `judge-pilot/judgments.jsonl` | 8 | 2 |
| `pilot-private/results.jsonl` | 96 | 8 |

The earlier stopped development run contains only three observed synthetic multihop questions. Preflight, numerical, tokenizer, and retrieval checks without normalized benchmark prediction ledgers are cataloged separately in the JSON; they are not counted as benchmark model exposure.

## Exact overlapping questions

| Dataset | Example ID | Exact question |
|---|---|---|
| BEAM | `beam-1m-1:abstention:0` | Can you tell me more about my background in psychology, like where I studied or my specialization? |
| BEAM | `beam-1m-1:abstention:1` | What was the feedback or user reaction to the multi-language assistant's dynamic language switching feature in the chat UI? |
| HotpotQA | `5a85f98e5542996432c57152` | Where did the designer of the Van de Graaff generator teach at? |
| HotpotQA | `5ae135fb55429920d523431f` | Scott Workman was an American stuntman and actor credited with a successful show on what smaller TV Network? |
| MuSiQue | `3hop1__241001_568433_51423` | What is the name of the castle in the headquarters of the company that employs Arna Selznick? |
| MuSiQue | `2hop__511454_120259` | When was Lady Godiva's birthplace abolished? |

The public QA overlap also appears in the focused first32 cohorts. There are no ID-only matches with differing question text and no extra exact-question matches under different IDs.

## Development/test disjointness

| Check | Sizes | Shared IDs | Shared exact questions |
|---|---:|---:|---:|
| `synthetic_private_fact` | 32 / 128 | 0 | 0 |
| `synthetic_multihop` | 32 / 128 | 0 | 0 |
| `synthetic_updates` | 32 / 128 | 0 | 0 |
| `adaptive_test96_vs_original_musique128` | 96 / 128 | 0 | 0 |
| `adaptive_test96_vs_adaptive_development6` | 96 / 6 | 0 | 0 |

All ten preserved pilot ledgers have zero overlap with adaptive test96. The adaptive and pressure development runs use the same six development questions. The synthetic pilot ledgers also have zero overlap with the focused synthetic16 questions. These checks establish question disjointness, not disjoint articles, component facts, templates, or model pretraining.

## Descriptive E1 sensitivity

The original headline denominator remains 128 per QA dataset. The table below removes the union of observed pilot IDs and exact question strings, giving 126 questions per arm. It uses existing scored predictions, with no reruns. This is a posthoc descriptive check, not a replacement holdout study or a correction for every possible development effect.

| Dataset | Arm | Original correct / 128 | Correct after exclusion / 126 | EM after exclusion | F1 after exclusion |
|---|---|---:|---:|---:|---:|
| HotpotQA | Direct text | 50 | 50 | 39.68% | 48.64% |
| HotpotQA | Text notes | 48 | 47 | 37.30% | 47.05% |
| HotpotQA | Native relay BF16 | 40 | 39 | 30.95% | 39.03% |
| HotpotQA | Native relay int8 | 40 | 39 | 30.95% | 39.71% |
| MuSiQue | Direct text | 12 | 12 | 9.52% | 12.06% |
| MuSiQue | Text notes | 18 | 18 | 14.29% | 18.83% |
| MuSiQue | Native relay BF16 | 8 | 8 | 6.35% | 10.71% |
| MuSiQue | Native relay int8 | 8 | 8 | 6.35% | 10.22% |

The descriptive ordering is unchanged: HotpotQA direct text > text notes > the two relays; MuSiQue text notes > direct text > the two relays. Both overlapping MuSiQue questions were incorrect under all four E1 arms. In HotpotQA, direct text was incorrect on both overlapping questions; each other arm answered one correctly. No uncertainty or causal superiority claim is added by this sensitivity.

## Source chronology and disclosure

The stopped earlier synthetic development ledger began at 06:40:18 UTC and records an older experiment runner. Revised synthetic development began at 06:45:08 UTC. The public pilots began at 06:46:39 UTC (HotpotQA), 06:47:04 UTC (MuSiQue), and 06:47:26 UTC (BEAM). All revised development jobs record the same four executable SHA256 values as the initial main jobs. This supports that the relevant revision preceded public pilot inference and that these four executable files did not change between those pilots and the initial main run. It does not establish that pilot-exposed examples were unseen or that no human inspection occurred.

Disclosure: “The initial public cohorts retain two HotpotQA questions, two MuSiQue questions, and two BEAM probes used in runtime pilots. The BEAM judge pilot reused the same two probes. These questions also occur in the focused public QA subsets. The executable source hashes match between the revised public pilots and initial main runs. A descriptive E1 check excluding the two questions per QA dataset leaves 126 examples and the same accuracy ordering. Synthetic development/test questions and adaptive test96 versus original MuSiQue128 and adaptive dev6 are disjoint by ID and exact question text.”

## Limits

Counts reflect preserved prediction and judge ledgers, not unrecorded in-flight work or every possible human inspection. Data/tokenizer/retrieval audits can inspect predeclared inputs or annotations without model inference. This audit cannot assess pretraining contamination. The public samples were therefore pilot-exposed benchmark diagnostics, while the narrower question-disjointness claims for the synthetic and adaptive splits are supported. Final-test outcome tuning intent cannot be established from these records alone.
