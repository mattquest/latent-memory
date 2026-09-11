# Focused final analysis — September 11, 2026

The focused matrix completed **660/660 conditions** across all seven jobs. The guard returned zero after **1,556.915 seconds**, with **15.571 GiB peak RSS**. No result errors, duplicate IDs, missing requested conditions, or malformed records were found. The automatic pipeline advanced to the BEAM judge; this audit did not launch or interrupt inference.

Both requested exports passed `--require-complete` into `reports/2026-09-11/focused`: `build_report.py` produced 150 metric strata, 112 paired comparisons and 14 figures; `analyze_failure_modes.py` produced complete descriptive diagnostics. The empty public-E1 section is expected: this matrix contains E2, E4 and E5 only. Judge status is correctly absent for this separate matrix.

An independent CPU audit in `runs/audit_focused_final.py` passed **275 checks** and wrote `runs/focused-final-audit.json`. It verifies exact requested question-condition sets, all dataset identities, duplicate/error absence, final output and prefix/evidence bounds, matched ordered evidence IDs, distinct E2 schedules, identical E4/E5 prefix lengths, overlapping BF16 outputs, recomputation counts, packed-storage fractions, and all **67 archive artifact digests** (including original-byte digests when applicable). Two generated license notes have artifact digests but no original source-file digest, as expected. This is a posthoc descriptive audit, not an additional model evaluation.

All seven jobs used the same archived executed source set, `4450540cd99c0b4d70f19e7cd6563c7e3eb186c872df8231a8a07993886e80d2`, and recorded Git head `95fcc1363fba276c4a4b7c017b6e1c085b533c6d`. The archive matches the actual backend, harness, retrieval and runner hashes. Exact per-job results, dataset and manifest hashes are in the audit JSON.

## E2: small structural cohorts, distinct evidence schedules

The five groups contain **4/3/3/3/3 questions**, exactly 16 disjoint original questions. Their condition counts are **32/36/36/48/60**, totaling **212**. Every curve has a fixed cohort; every arm sees identical ordered selected document IDs at a given budget. No selected schedule is duplicated within a question. Nominal/actual budgets are 1,3 / 1,3 for six documents; 1,3,5 / 1,3,4 for seven; and identical nominal/actual budgets for the remaining groups.

The table gives each group's largest distinct budget. These are exact counts, not evidence of a population ranking.

| Documents / questions | Nominal / actual batches | Correct: direct / notes / BF16 / int8 | Warm p50 seconds: direct / notes / BF16 / int8 |
| --- | --- | --- | --- |
| 6 / 4 | 3 / 3 | 4/4 / 1/4 / 0/4 / 0/4 | 0.37 / 11.54 / 1.04 / 1.07 |
| 7 / 3 | 5 / 4 | 2/3 / 2/3 / 0/3 / 1/3 | 0.38 / 17.08 / 1.49 / 1.52 |
| 10 / 3 | 5 / 5 | 2/3 / 1/3 / 2/3 / 1/3 | 0.44 / 18.96 / 2.15 / 2.13 |
| 20 / 3 | 10 / 10 | 0/3 / 2/3 / 0/3 / 0/3 | 0.57 / 41.23 / 7.17 / 7.16 |
| 40 / 3 | 20 / 20 | 0/3 / 1/3 / 0/3 / 0/3 | 0.83 / 87.34 / 28.63 / 28.63 |

At 1,3,5,10,20 batches in the 40-document group, BF16 correct counts are 0,0,1,0,0; notes 0,0,1,1,1; direct text 0,1,0,0,0. More fixed evidence does not give a monotone accuracy curve. Notes solve some questions the other arms miss, and independent relay has no sustained advantage here. One answer changes a group score by 25–33 percentage points. These fixed schedules do not measure an adaptive-search trajectory or establish the original long-horizon kill criterion.

At twenty batches, all three questions expose all supporting document IDs and all forty documents, with **1,606 evidence tokens** and no selected truncation. The BF16/int8 relays cumulatively recompute **5,456 token positions**. The source keeps historical `block_lengths`, flattens them at each concat, selects near all accumulated boundaries again, and forwards separate contiguous spans. Repeated sparse recomputation and a growing prefix are implementation costs; these measurements do not support the guiding estimate of 100–200 ms per latent hop. BF16 and int8 median warm costs are 28.632 and 28.633 seconds. Their union-of-documents precompute median is a separate 2.440 seconds.

Twenty-batch notes decode a mean 2,305 internal tokens; direct text and both relays decode zero internally. Mean final outputs are 6 / 2.667 / 4.333 / 1 tokens for direct/notes/BF16/int8. UNKNOWN counts are 0/3, 2/3, 1/3 and 3/3 respectively. **No final answer reaches its output cap in any of the 660 conditions.** Separately, 16 of 241 synthetic note-generation turns reach 192 decoded tokens; the saved traces do not record their stop reason. Do not describe all intermediate notes as uncapped.

## E4: recomputation quality and warm cost

Each row has the same 32 questions within its dataset, the same six selected paragraphs, and three scheduled evidence batches. All four ratios match on ordered IDs and prefix lengths. Ratio zero recomputes zero positions; ratio one recomputes exactly the full stored prefix for all 64 questions.

| Dataset | Ratio | Correct / 32 | F1 | Warm p50 / p95 seconds | UNKNOWN / 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 0 | 6 | 27.51% | 0.113 / 0.570 | 14 |
| HotpotQA | .1 | 9 | 33.24% | 0.441 / 0.578 | 15 |
| HotpotQA | .2 | 8 | 30.12% | 0.503 / 0.614 | 17 |
| HotpotQA | 1 | 14 | 53.94% | 0.480 / 0.613 | 9 |
| MuSiQue | 0 | 3 | 13.79% | 0.109 / 0.950 | 19 |
| MuSiQue | .1 | 4 | 14.06% | 0.427 / 0.535 | 25 |
| MuSiQue | .2 | 4 | 14.06% | 0.480 / 0.598 | 26 |
| MuSiQue | 1 | 6 | 18.75% | 0.443 / 0.557 | 26 |

Full causal recomputation has the highest observed EM/F1 in both subsets. Partial recomputation is not monotonic: .1 exceeds .2 on HotpotQA. Full recomputation also has lower median warm latency than .2 in these measurements; a single full forward and multiple sparse forwards have different overhead. Output lengths differ too. This is not proof that more recomputation is generally cheaper, or that a particular partial ratio is optimal.

All supporting IDs were selected for 23/32 HotpotQA and 9/32 MuSiQue questions, unchanged across conditions. Selected documents were truncated on 3/32 and 4/32 respectively. Supporting ID coverage does not prove that answer-bearing text survived truncation or that the model used it. The posthoc E4 JSON preserves all four matched scores per question and reports no incomplete or evidence-mismatched pairs.

## E5: actual packed storage, limited precision effects

All rows use .2 recomputation and the same 32 questions per dataset. BF16/int8/int4 mean final cache payloads are exactly **147,456 / 78,336 / 41,472 bytes per token**, including quantization scale/bias overhead: 100% / 53.125% / 28.125% of BF16. Attention dequantizes to BF16; these are storage measurements, not lower-precision attention or whole-process memory ratios.

| Dataset | Precision | Correct / 32 | F1 | Warm p50 / p95 seconds |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | BF16 | 8 | 30.12% | 0.509 / 0.618 |
| HotpotQA | int8 | 8 | 30.12% | 0.514 / 0.629 |
| HotpotQA | int4 | 8 | 28.33% | 0.514 / 0.630 |
| MuSiQue | BF16 | 4 | 14.06% | 0.480 / 0.595 |
| MuSiQue | int8 | 4 | 14.06% | 0.486 / 0.607 |
| MuSiQue | int4 | 3 | 10.94% | 0.487 / 0.603 |

Int8 final output tokens match BF16 on all **64/64** questions. Int4 changes one output per dataset: HotpotQA `5a8ee3a755429917b4a5be02` changes “Bing Crosby” to UNKNOWN (both EM zero, F1 .5714 to zero); MuSiQue `2hop__176712_8311` changes “1216” to “1253” (correct to incorrect). This observed agreement for int8 is not a statistical equivalence claim. The small timing differences do not establish a quantization speedup.

The E4 .2 and E5 BF16 conditions are intentionally overlapping evaluations. Their output tokens and scores match **64/64**. They count toward executed conditions, but are not independent examples or replications supporting additional accuracy certainty.

## Costs and visual inspection

All focused latency tables report **warm online model work**. They exclude model load, independently precomputed document caches, schedule selection and shared final-question tokenization. Union cache precompute is shared across an example's cases; it must not be presented as per-case cold ingestion. Public median union ingest costs are 0.495 seconds for HotpotQA and 0.485 seconds for MuSiQue. Direct-text warm work includes its joint evidence prefill. No cold end-to-end or production Pareto claim follows from these plots.

All 14 generated PNGs were visually inspected: five primary E2 plots, five chain-length counterparts, two E4 plots and two E5 plots. Text and legends are legible, data agree with raw counts and medians, and each E2 curve shows its fixed n. The E5 panel explicitly labels storage and BF16 dequantization. The seven-document plot uses nominal x=5; any standalone caption must say it executes four batches. The chain-length plots repeat the same five cohorts with explicit chain lengths; they are alternate views, not more observations. No plots were edited after results.

The E2 counts, endpoint medians, forty-document curves, internal-token count and cumulative recomputation figures already inserted in `runs/readme-draft.md` were independently cross-checked against the raw ledgers and source and agree.
