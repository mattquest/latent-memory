# Controller scaling report contract

`scripts/build_controller_scaling_report.py` is a CPU-only export. It consumes a
declared four-job matrix: two pinned checkpoints, each with controller reasoning
budgets 0 and 256. The first development pilot can contain only `iterative_text`;
a later frozen comparison can contain `basic_rag`, `iterative_text`,
`incremental_kv_bf16`, and `relay_kv_bf16`. It never launches inference.

The matrix uses `schema_version: controller-scaling-matrix-v1` and these fields:

- `cohort`: `development` or `frozen_test`.
- `dataset`, `corpus`, `questions_sha256`, `corpus_sha256`, and `data_manifest`.
- `shared_config`: the exact adaptive generation configuration shared by all jobs,
  including paths, limit, seed, arms, round caps, token/document budgets, bridge
  ratio and controller sampling. Omit `max_reasoning_tokens`, `output_dir`,
  `resume`, `max_runtime_seconds`, and `max_case_seconds`; the latter four are
  operational settings. Include default generation settings too.
- `jobs`: four records containing a simple unique `id`, repository-relative
  `run_dir` and `model_path`, pinned `model_revision`, `model_label`, and
  `max_reasoning_tokens`. An optional `protocol` must match that job's declared
  controller policy. Each model label must refer to one checkpoint.
- `provenance_files`: optional additional repository-relative model manifests,
  preparation receipts, or protocol files to archive.

The data-selection manifest must have an `outputs` list of `{path, sha256}` records
pinning the evaluated questions and corpus. The relocatable
`controller-scaling-data-v1` package resolves these paths against the manifest's
directory; historical adaptive manifests use repository-relative paths.
Every listed output is verified and
archived. Dataset partitioning and the decision to freeze a test happen before
running the exporter; an exporter cannot retroactively establish that a cohort
was not used for development.

```sh
.venv/bin/python scripts/build_controller_scaling_report.py \
  --matrix configs/controller-scaling-development.json \
  --output-dir reports/controller-scaling/development \
  --require-complete
```

Every job must have a completed runner summary, exactly one successful row per
expected question/arm/round condition, no historical errors or duplicate rows,
and unchanged generation sources and data. The exporter reconstructs condition
IDs, scores and supporting-ID exposure. It verifies common controller seeds,
reasoning/action token accounting, context and document caps, and exact shared
generation settings. A failed verification withholds headline tables and plots
for the whole matrix while preserving the source ledgers for diagnosis.

Outputs include EM, F1, cold p50/p95, component timings, host/reasoning/action and
forced-control token counts, repeated/invalid stops, final-answer token caps,
per-step new supporting IDs and cumulative annotation coverage. Every cross-model
or reasoning-budget contrast keeps exactly the same question IDs. The full
four-arm design also includes within-job method and round-budget contrasts.

Trajectories retain the original question denominator after early stops: later
new-support gains are zero and cumulative coverage carries forward. The separate
reached-round denominator describes a controller-selected subset. Neither curve
measures intermediate answer accuracy. Different round caps are separate adaptive
runs, and one seed per question is not repeated-sampling uncertainty.

Each run constructs model-specific caches. Cross-model KV transfer is not tested.
Model loading and offline indexing remain separate from cold query latency.
Prefill/decoded tokens and allocator/RSS measurements describe work and memory;
they are not dollar prices or measured energy. Annotations assess delivered
support after inference, without claiming that delivered text was understood or
that a truncated paragraph exposed its relevant passage.

`artifact-manifest.json` records packed and unpacked SHA-256 and byte counts for
gzip archives of raw outputs, manifests, inputs, generation sources, report
sources, selection provenance and optional supporting receipts. Model weights
are excluded. MuSiQue attribution and modifications appear in `DATA_LICENSE.md`.
