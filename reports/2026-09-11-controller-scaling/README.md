# Controller scaling: limited development screen

Completed September 11, 2026. The [repository README](../../README.md#limited-controller-screen-september-11-follow-up)
contains the full findings, interpretation, costs and caveats. This package keeps
the original 48 end-to-end conditions separate from 96 posthoc final-answer
calls. Both use twelve questions; the prepared 96-question test remains unused.

The original exact-match counts are **0/12, 0/12, 1/12, 1/12** for 8B short,
8B thinking, 14B short and 14B thinking. Thinking retrieves more annotated
support but takes about nine times the paired query latency, without improving
accuracy. Removing one search-command instruction from final synthesis eliminates
five stray search commands but leaves all configurations at **1/12**. This is a
reason to repair and screen the controller/answer pipeline before a large cache
comparison, not a population result about latent memory.

## Results and audit files

| File | Contents |
| --- | --- |
| [tables.md](tables.md) | Original end-to-end headline table and three scientific plots |
| [summary.json](summary.json) | Strict completion checks, all metrics, pairs, trajectories, model/setup metadata |
| [metrics.csv](metrics.csv) | Per-job full-cohort and hop-category metrics |
| [pairs.csv](pairs.csv) | Same-question model and reasoning-budget contrasts |
| [trajectories.csv](trajectories.csv) | Support exposure and repetition by step, with explicit denominators |
| [productivity.json](productivity.json) | Per-question support discovery reconstructed from traces |
| [pilot-diagnostics.json](pilot-diagnostics.json) | Caps, stops, command leakage, support gains and raw-action summaries |
| [per-question.md](per-question.md) | Every question, gold aliases, original predictions, support coverage and rounds |
| [final-diagnostic-summary.json](final-diagnostic-summary.json) | All paired final-prompt predictions and final-stage-only measurements |
| [qualitative-audit.md](qualitative-audit.md) | Exact source references for useful discovery and annotation gaps |
| [artifact-manifest.json](artifact-manifest.json) | 88 packed/unpacked raw-artifact receipts from strict export |
| [supplement-manifest.json](supplement-manifest.json) | Additional diagnostic, guard, preflight, code and chronology receipts |
| [publication-manifest.json](publication-manifest.json) | File hashes for this published package and root README |
| [DATA_LICENSE.md](DATA_LICENSE.md) | MuSiQue attribution and modifications |

`artifacts/runs/` contains each complete original ledger, run manifest, summary,
checkpoint, resource samples, guard receipt and process log as gzip. `artifacts/inputs/`
contains the actual question/corpus bytes per job. `artifacts/source/` preserves
executed generation source. `artifacts/provenance/` preserves the new data
selection, prior-question exclusions and original benchmark source member.
Model weights are excluded; pinned download inventories and hashes are included.

`supplement/runs/controller-scaling/final-diagnostic-v1/` contains all 96 final
calls, paired records, full prompt token IDs, source/checkpoint receipts and
resource guards. A first launch used an invalid model label and failed before
model loading; its five-second guard receipt is retained separately. The two
successful jobs verified and replayed every selected condition. No failed launch
is counted as inference or as a missing evaluated question.

## Reproduction and chronology

The [generation protocol](../../docs/controller-scaling-protocol.md),
[data preparation](../../docs/controller-scaling-data.md),
[final-prompt diagnostic](../../docs/controller-final-diagnostic.md) and
[report contract](../../docs/controller-scaling-report.md) contain exact commands
and configuration fields. The matrix is preserved in
[its gzip archive](artifacts/matrix/controller-scaling-development.json.gz).

The original pipeline source was committed at
`d1d4ffa08c65d97b2ba4587c0683566a3840e712`. The final diagnostic was added at
`8b564da` and used the same generation bytes. These frozen versions matter:
a later one-line parser fix accepts trailing whitespace after an ANSWER finish
line. The old source and before/after hashes are preserved under
`supplement/runs/controller-scaling/postpilot-parser/`. Do not run the repaired
parser and label that execution an exact pilot replay. No existing prediction
or score was retrospectively changed.

A report-only draw call was added to ensure shared-axis tick labels appear in
PNG exports. An interim export occurred while the first final diagnostic was
running; the original reporter was immediately restored, and both diagnostic
manifests pin that original reporter hash. Both versions and the chronology are
preserved under `supplement/runs/controller-scaling/report-rendering/`. Final
strict export used the draw fix before the parser repair; all original generation
sources still matched at export. The draw change does not affect scores,
retrieval, model calls or diagnostic execution.

The strict exporter intentionally rejects original ledgers if the checked-out
generation sources differ from their manifests. To rerun inference or re-export
original ledgers, use an isolated checkout of the frozen generation source and
the exact archived inputs; use the archived report source for identical rendering.
Current repaired code is for new, separately identified runs. Resource guards and
model download/setup instructions remain in the linked protocols. Reproducing
model timings requires appropriate Apple Silicon hardware and does not guarantee
identical wall-clock times on another machine.

The following commands recompute the extra descriptive summaries **from this
published archive**, without model loading or access to original local run paths:

```sh
python reports/2026-09-11-controller-scaling/analyze_pilot.py
python reports/2026-09-11-controller-scaling/analyze_final.py
python reports/2026-09-11-controller-scaling/verify_publication.py
```

The first two scripts preserve all original questions and recompute only
post-inference diagnostics. The verifier checks packed/unpacked receipts, archived
source identities, complete-condition counts and publication file hashes.
Numeric condition IDs are scoped by job; models may share a question/policy case
ID, so cross-job comparisons use `(job_id, case_id)`.

## Limits

This is a small, inspected development subset of public MuSiQue paragraphs,
not a full-Wikipedia or production-store evaluation. Several support annotations
omit required relations, and support-ID exposure does not establish sufficient
text. Nearly every reasoning call hit its 256-token cap. Final synthesis discards
transient controller thought and remains greedy/nonthinking. The posthoc system
ablation changes only that system sentence; it neither preserves thought state nor
measures a revised end-to-end pipeline. No basic-RAG or cache arm ran in this
limited screen, no trained latent planner or compressor was evaluated, and no
held-out superiority, dollar saving or energy reduction is claimed.

The earlier study under `reports/2026-09-11/` remains a separate historical
snapshot. Its README publication receipt refers to the earlier report commit,
not the expanded current README.
