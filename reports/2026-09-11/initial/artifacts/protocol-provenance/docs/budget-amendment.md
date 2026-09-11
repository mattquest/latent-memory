# Experiment budget amendment — 2026-09-11

The user's latest priority is harder global retrieval and memory under context pressure, compared with competent retrieval baselines on accuracy and latency. The original fixed-evidence matrix would delay those experiments and contains nominal hop settings that repeat an already exhausted evidence schedule. This amendment changes allocation before the adaptive test. It selects no questions by predictions, correctness, or other model outcomes. It leaves generation prompts, model settings, and completed records unchanged.

## Original matrix and preservation

`configs/overnight.json` plans **8,120 conditions**. The stopping boundary is after its first five jobs: private audit/precision (1,024), Hotpot equal conditions (512), MuSiQue equal conditions (512), BEAM equal/update-category conditions (288), and synthetic version updates (1,024): **3,360 conditions** in total. These numbers describe the plan; run ledgers and completion manifests establish what actually finished.

The original **4,760-condition remainder is canceled**, not marked complete: synthetic multihop (1,984), Hotpot ablations (448), MuSiQue ablations (448), Hotpot hop sweep (640), MuSiQue hop sweep (640), and BEAM hop sweep (600). Any multihop directory created while stopping at the boundary is preserved under `runs/original-interrupted-multihop`, with its original file bytes, and reported as a partial, canceled cohort. Original manifests, source identities, raw data, and results are preserved. A later amended job is a distinct cohort and does not retroactively complete an original canceled job.

## Amended fixed follow-up: 660 conditions

`scripts/prepare_focused_e2.py` takes exactly the **same first 16 rows** of `data/synthetic_multihop.jsonl`, preserves each JSON record, and partitions them by available document count. Source order within each group remains unchanged. It retains the first original nominal hop cap that produces each distinct executed schedule. At two evidence documents per hop, larger caps after evidence exhaustion would add no evidence, no summary call, and no latent hop.

| Available documents | Questions | Original nominal hop caps retained | Executed evidence hops | Conditions across four arms |
| --- | --- | --- | --- | --- |
| 6 | 4 | 1, 3 | 1, 3 | 32 |
| 7 | 3 | 1, 3, 5 | 1, 3, 4 | 36 |
| 10 | 3 | 1, 3, 5 | 1, 3, 5 | 36 |
| 20 | 3 | 1, 3, 5, 10 | 1, 3, 5, 10 | 48 |
| 40 | 3 | 1, 3, 5, 10, 20 | 1, 3, 5, 10, 20 | 60 |
| Total | 16 | | | **212** |

The four arms remain JSON notes, BF16 independent relay, int8 independent relay, and direct full text. This removes 108 repeated-schedule cells from a 320-condition first16 sweep. Group sizes are small. Plots must separate structural groups or explicitly report changing cohort composition and exact denominators; a pooled line across hop caps would confound hop count with which questions remain in the cohort.

Hotpot and MuSiQue each retain their first **32** public questions, with the original four E4 bridge ratios (0, .1, .2, 1) and all three E5 precisions (BF16, int8, int4): **224 conditions per dataset**. The inexpensive BF16 E4/E5 duplicate is retained to keep the generation protocol intact. These overlapping conditions are not independent replications. The follow-up totals **212 + 224 + 224 = 660 conditions** in `configs/focused-ablation.json`.

Grouped JSONLs and `data/focused_e2_groups.manifest.json` record input/output SHA-256 hashes, source row indices, question IDs, nominal caps, executed hops, and script identity. No outcome fields influence grouping.

## Higher-priority experiments

After the original stopping boundary, the order is real-weight synthetic mechanics preflight, six local development questions, a pressure-replay development check on those six traces, the full disjoint adaptive test, then matched-trace context pressure. The fixed follow-up and local BEAM judging complete the planned report.

- Adaptive global retrieval: **96 questions × 7 arms = 672 test conditions**, preceded by **6 × 7 = 42 development conditions**. The seven arms include basic BM25, query expansion, reranking, iterative full text, exact incremental KV, and independent BF16/int8 relay. Test membership was chosen structurally before model outcomes.
- Matched-trace pressure: **96 × 4 = 384 conditions** at the predeclared 1,536-token evidence budget. It replays every saved iterative-text trace; it is a retention diagnostic using decisions made with a larger window.
- Local BEAM judging: **288 completed answer candidates** from the equal/update-category job. This is a condition-blind local same-generator rubric proxy on three source conversations, not published BEAM accuracy. The canceled large BEAM hop sweep is not judged or implied complete.

All model jobs remain serial under the existing memory, system-resource, and wall-time guard. No system settings, guard limits, or model weights change to meet a deadline.

## Timing estimates and uncertainty

The revised decision uses observed costs, not a promise of completion at a particular time. Completed public three-hop text conditions average about 6.5 seconds and 168–170 internal tokens; synthetic development notes emitted 62, 83, and 146.5 tokens at the first three hops. Measured total text time is roughly .036–.038 seconds per generated token. Longer 192-token notes therefore dominate large static sweeps.

Extrapolating these measurements estimated the original remaining matrix at roughly **6–9.5 additional hours**. Longer-hop timings have not yet been measured, so that range is uncertain. Structural pruning estimates synthetic first16 E2 at **23–29 minutes**, with the complete 660-condition focused follow-up around **29–38 minutes**. Adaptive 96×7 was provisionally budgeted at **45–100 minutes**, pressure 96×4 at **25–110 minutes**, and BEAM judging at **15–45 minutes**. These last ranges lack matching full-run measurements and must be revised from development/early-run telemetry. A 6–7AM finish is plausible only near typical costs; it is not guaranteed. If time runs out, incomplete jobs remain explicit and the user's latest adaptive/context-pressure questions take priority over the deferred static sweep.
