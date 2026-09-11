# September 11, 2026 experiment artifacts

The [repository README](../../README.md) is the findings report. This index describes the supporting artifacts. All four amended generation cohorts and the separate local rubric judge completed. The original long-horizon/context-capacity claim remains unresolved; all annotated adaptive support sets still fit the smaller evidence budget.

| Cohort | Questions | Generation conditions | Report |
| --- | ---: | ---: | --- |
| Initial private audit/precision, HotpotQA, MuSiQue, BEAM, updates | 128 / 128 / 128 / 60 / 128 | 3,360 | [Tables](initial/tables.md), [summary](initial/summary.json), [metrics](initial/metrics.csv), [paired comparisons](initial/paired.csv) |
| Focused synthetic hop curves and public recomputation/precision | 16 / 32 / 32 | 660 | [Tables](focused/tables.md), [summary](focused/summary.json), [metrics](focused/metrics.csv), [paired comparisons](focused/paired.csv) |
| Global adaptive retrieval | 96, disjoint from original MuSiQue questions | 672 | [Tables](adaptive/tables.md), [summary](adaptive/summary.json), [metrics](adaptive/metrics.csv), [paired comparisons](adaptive/pairs.csv) |
| Matched-trace context retention | Same 96 adaptive questions | 384 | [Tables](context-pressure/tables.md), [summary](context-pressure/summary.json), [metrics](context-pressure/metrics.csv) |

These total **5,076 generation conditions**, not 5,076 independent questions. Focused public ablations reuse the first 32 questions from the initial cohorts; pressure reuses all 96 adaptive questions. BEAM adds **288 separate local rubric judgments** over its completed candidates. Its three conversation clusters and exploratory same-model judge are not an official leaderboard evaluation.

The original plan contained 8,120 conditions. Its first 3,360 were retained; the original 4,760-condition remainder was explicitly canceled, then replaced by the focused/adaptive/pressure work. The [budget amendment](../../docs/budget-amendment.md), original configuration, stopping receipt, and excluded partial next-job records are preserved. Completion applies to the declared amended cohorts only.

## Audit trails

Each report includes `artifact-manifest.json` with compressed and source byte counts and SHA-256 hashes. Exact normalized inputs, raw outputs, generation manifests, historical executable source, and resource/guard receipts are under its `artifacts/` directory. Model weights are not committed. Follow each report's data license separately from the project's code license.

- [Initial artifact manifest](initial/artifact-manifest.json) and [data attribution](initial/artifacts/data-inputs/DATA_LICENSES.md).
- [Focused artifact manifest](focused/artifact-manifest.json) and [data attribution](focused/artifacts/data-inputs/DATA_LICENSES.md).
- [Adaptive artifact manifest](adaptive/artifact-manifest.json) and [data attribution](adaptive/DATA_LICENSE.md).
- [Pressure artifact manifest](context-pressure/artifact-manifest.json) and [data attribution](context-pressure/DATA_LICENSE.md).
- [Validation and excluded test history](validation-history/README.md), including original failures, stopped development runs, revised mechanics checks, and canceled partial work. These are not pooled into final scores.
- [Synthetic update-note diagnosis](e6-note-analysis.md), [JSON](e6-note-analysis.json), and [per-condition examples](e6-note-examples.jsonl).
- [Independent adaptive analysis](adaptive-analysis.md), [machine-readable analysis](adaptive-analysis.json), and [per-question support-size audit](adaptive-support-size-audit.json). The complete annotated support sets all fit within 1,536 evidence tokens; corpus size is not proof of necessary-information overflow.
- [Independent pressure analysis](pressure-final-analysis.md), [token/ranking audit](pressure-final-audit.json), and [extra agreement checks](pressure-final-extra.json).
- [Prefix reuse by executed rounds](prefix-reuse-by-rounds.json), a descriptive comparison of different controller-selected question groups.
- [Independent focused analysis](focused-final-analysis.md), [condition/storage audit](focused-final-audit.json), and [CPU audit source](audit_focused_final.py).
- [Independent BEAM audit](beam-final-analysis.md) and [machine-readable analysis](beam-final-analysis.json).
- [Development/pilot overlap audit](development-overlap-audit.md) and [exact joins and sensitivity results](development-overlap-audit.json). The initial public samples retain two pilot-exposed questions/probes per dataset; the focused QA subsets inherit that overlap.
- [Completion and resources](resource-and-completion.json) record all five guarded stages, deliberate original interruption, sampled memory, and final test validation.
- [Release verification](release-verification.json) and [restored input verification](restored-input-verification.json).
- [Environment package versions](environment.json) record the 71 installed distributions without source URLs or private environment variables.
- [Supplementary receipt manifest](supplementary-manifest.json) covers every published report file and the repository README; individual exporter manifests additionally verify unpacked source bytes.

## Restore and reproduce

From the repository root, restore `initial`, `focused`, and `adaptive` inputs with `scripts/restore_report_data.py --report-dir reports/2026-09-11/COHORT`. It verifies every selected archive before writing, preserves original JSONL basenames, reuses identical existing files, and rejects conflicts. The adaptive package includes the development questions and corpus provenance; pressure inputs duplicate its test/corpus bytes.

The README gives the exact model-download, guarded-run, and export commands. `docs/data-and-protocol.md` and `docs/adaptive-data.md` describe rebuilding from public sources. The current code contains recorded maintenance changes after the first five original jobs; their exact executed sources are included rather than replaced with a current-code claim. Deterministic greedy decoding does not guarantee identical wall times or floating-point outputs across dependency or hardware changes.

Percentages, uncertainty intervals, and timing definitions are specific to each cohort. Original fixed-evidence times are warm and exclude shared preparation; adaptive times charge all online per-question work; pressure times are replay costs, with source controller/retrieval costs recorded separately. None includes resident model loading. Recovered basic-RAG failures and actual overflow traces are conditional diagnostics, not independently sampled populations.
