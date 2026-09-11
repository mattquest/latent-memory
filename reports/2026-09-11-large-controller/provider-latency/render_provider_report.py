#!/usr/bin/env python3
"""Render only the frozen, sourced controller latency projection artifacts."""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent,
    help="Directory containing frozen JSON inputs; reports are rendered alongside them")
HERE = parser.parse_args().input_dir.resolve()
workloads = json.loads((HERE / "workloads.json").read_text())
projection = json.loads((HERE / "projections.json").read_text())
observations = json.loads((HERE / "published-source-observations.json").read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert projection["workloads_sha256"] == sha(HERE / "workloads.json")
assert projection["profiles_sha256"] == sha(HERE / "profiles.json")
profiles = {p["id"]: p for p in projection["profiles"]}
sources = {s["id"]: s for s in observations["sources"]}
assert len(profiles) == 14 and len(projection["conditions"]) == 408
assert len(projection["summaries"]) == 34 and len(projection["paired_summaries"]) == 17
assert all(p["snapshot_sha256"] == sha(HERE / "published-source-observations.json") for p in profiles.values())
groups = defaultdict(list)
for run in workloads["runs"]:
    for row in run["workloads"]:
        groups[(run["label"], row["arm"])].append(row)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def local_stats(rows):
    values = [row["local_observed_ms"]["cold_end_to_end"] / 1000 for row in rows]
    return {"mean": statistics.mean(values), "p50": quantile(values, .5), "p95": quantile(values, .95)}


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


summary_rows = []
for group in projection["summaries"]:
    profile = profiles[group["profile_id"]]
    local = local_stats(groups[(group["run_label"], group["arm"])])
    row = {"profile_id": profile["id"], "model_id": profile["model_id"], "provider": profile["provider"],
        "screen": "native" if group["run_label"].startswith("native/") else "prior_bounded_thinking",
        "run_label": group["run_label"], "arm": group["arm"], "n_questions": group["conditions"],
        "observed_local_correct_unchanged": group["observed_correct_unchanged"],
        **{"measured_local_" + k + "_s": v for k, v in local.items()},
        "profile_ttft_proxy_s": profile["ttft_s"], "profile_reported_tps": profile["reported_tps"],
        "metric_semantics": profile["metric_semantics"]}
    for field, prefix in [("decode_rate_assumption_s", "scenario_A"), ("inclusive_rate_proxy_s", "scenario_B")]:
        for stat in ("mean", "p50", "p95"):
            row[prefix + "_" + stat + "_s"] = group[field][stat] if group[field] else None
    row.update(source_id=profile["source_id"], source_url=profile["source_url"],
        source_period=sources[profile["source_id"]]["period"], source_snapshot_sha256=profile["snapshot_sha256"],
        projection_sha256=sha(HERE / "projections.json"), workload_sha256=sha(HERE / "workloads.json"))
    assert row["n_questions"] == 12
    summary_rows.append(row)
write_csv("provider-summary.csv", summary_rows)

paired_rows = []
for pair in projection["paired_summaries"]:
    profile = profiles[pair["profile_id"]]
    left_label, left_arm = pair["baseline"].rsplit("/", 1)
    right_label, right_arm = pair["comparison"].rsplit("/", 1)
    left = {row["example_id"]: row for row in groups[(left_label, left_arm)]}
    right = {row["example_id"]: row for row in groups[(right_label, right_arm)]}
    assert set(left) == set(right) == set(pair["example_ids"]) and len(left) == 12
    local_left = {key: row["local_observed_ms"]["cold_end_to_end"] / 1000 for key, row in left.items()}
    local_right = {key: row["local_observed_ms"]["cold_end_to_end"] / 1000 for key, row in right.items()}
    row = {"profile_id": pair["profile_id"], "provider": profile["provider"], "baseline": pair["baseline"],
        "comparison": pair["comparison"], "n_paired_questions": 12,
        "measured_local_mean_paired_difference_s": statistics.mean(local_right[k] - local_left[k] for k in left),
        "measured_local_median_paired_comparison_over_baseline": statistics.median(local_right[k] / local_left[k] for k in left)}
    for field, prefix in [("decode_rate_assumption_s", "scenario_A"), ("inclusive_rate_proxy_s", "scenario_B")]:
        for stat in ("mean_paired_difference_s", "median_paired_comparison_over_baseline"):
            row[prefix + "_" + stat] = pair[field][stat] if pair[field] else None
    row.update(source_url=profile["source_url"], source_snapshot_sha256=profile["snapshot_sha256"],
        example_ids_json=json.dumps(pair["example_ids"], separators=(",", ":")),
        projection_sha256=sha(HERE / "projections.json"))
    paired_rows.append(row)
write_csv("provider-pairs.csv", paired_rows)

index = {(s["profile_id"], s["run_label"], s["arm"]): s for s in projection["summaries"]}
panels = [
    ("Native 14B: basic vs iterative", [("native/14b", "basic_rag"), ("native/14b", "iterative_text")],
        ("Basic RAG", "Iterative"), [("openrouter14b-alibaba", "Alibaba", "A"), ("openrouter14b-alibaba", "Alibaba", "B")]),
    ("Native 122B: basic vs iterative", [("native/122b", "basic_rag"), ("native/122b", "iterative_text")],
        ("Basic RAG", "Iterative"), [("openrouter122b-alibaba", "Alibaba*", "A"), ("openrouter122b-alibaba", "Alibaba*", "B"),
        ("lambda122b-lambda-4xb200-sglang", "4×B200\nSGLang†", "A"), ("lambda122b-lambda-8xh100-vllm", "8×H100\nvLLM†", "A")]),
    ("Prior 8B: short vs bounded thinking", [("prior/8b_short", "iterative_text"), ("prior/8b_thinking", "iterative_text")],
        ("Short", "Thinking"), [("openrouter8b-alibaba", "Model cache‡", "A"), ("openrouter8b-alibaba", "Model cache‡", "B")]),
    ("Prior 14B: short vs bounded thinking", [("prior/14b_short", "iterative_text"), ("prior/14b_thinking", "iterative_text")],
        ("Short", "Thinking"), [("openrouter14b-alibaba", "Alibaba", "A"), ("openrouter14b-alibaba", "Alibaba", "B")]),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
    "axes.labelsize": 10.5, "savefig.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8))
colors = ["#31688e", "#c9772f"]
for ax, (title, contrasts, labels, selected) in zip(axes.flat, panels):
    values = [[local_stats(groups[key])["mean"] for key in contrasts]]
    ticks = ["Measured\nlocal"]
    for profile_id, name, scenario in selected:
        field = "decode_rate_assumption_s" if scenario == "A" else "inclusive_rate_proxy_s"
        values.append([index[(profile_id, *key)][field]["mean"] for key in contrasts])
        ticks.append(name + "\nScenario " + scenario)
    ax.axvspan(-.48, .48, color="#eef1f4", zorder=0)
    ax.axvline(.5, color="#87929b", linestyle=(0, (3, 3)), linewidth=1)
    for arm_index, label in enumerate(labels):
        heights = [v[arm_index] for v in values]
        bars = ax.bar([x + (arm_index - .5) * .34 for x in range(len(values))], heights,
            width=.31, color=colors[arm_index], label=label, edgecolor="white", linewidth=.6, zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in heights], padding=3, fontsize=9)
    ax.set_xticks(range(len(ticks)), ticks)
    ax.set_ylabel("Mean seconds per question")
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    ax.set_xlim(-.55, len(values) - .45)
    ax.set_ylim(0, max(max(v) for v in values) * 1.27)
    ax.yaxis.grid(True, color="#dbe1e6", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle("Controller latency: measured locally and projected conditionally", x=.055, y=.975,
    ha="left", fontweight="bold", fontsize=17)
fig.text(.055, .941, "12 shared questions per condition • actual answers and call trajectories held fixed • no provider inference", fontsize=11)
fig.text(.055, .095, "A: F + max(G − 1, 0) / R.  B: max(F, G / R), per call; both add measured outer host time. G = generated tokens, R = reported rate.", fontsize=10)
fig.text(.055, .074, "F uses reported latency as a TTFT/base-time assumption for OpenRouter. A/B are separate scenarios, not uncertainty bounds or provider measurements.", fontsize=10)
fig.text(.055, .053, "* 122B OpenRouter: stale cache.  † Lambda: 8192 input / 1024 output tokens, concurrency 32; unmatched to these calls. No confidence intervals.", fontsize=10)
fig.text(.055, .032, "‡ The 8B cache is an unlabeled model-level performance series; Alibaba attribution is inferred from its current sole endpoint.", fontsize=10)
fig.subplots_adjust(left=.065, right=.985, top=.885, bottom=.165, hspace=.49, wspace=.20)
fig.savefig(HERE / "provider-latency.png", dpi=180)
plt.close(fig)

f = lambda value: "—" if value is None else f"{value:.3f}"
lines = ["# Conditional provider latency for the two controller screens", "",
    "This analysis keeps the recorded answers, retrieval decisions and token workloads fixed, then substitutes published provider or hardware timing profiles. It is a latency sensitivity analysis; no hosted inference was run and no provider accuracy was measured.", "",
    "There are **12 distinct development questions**, reused in 96 local conditions: 48 native basic/iterative conditions and 48 earlier short/bounded-thinking conditions. The 14 profiles produce 408 condition/profile calculations, 34 summary groups and 17 paired comparisons. These are repeated calculations on the same questions, not 408 independent subjects. All six published Lambda hardware/engine combinations are included below.", "",
    "## Selected means", "",
    "Every entry is a mean in seconds per question over the same 12 questions. “Local” is measured local latency; A and B are projected scenarios. Native and earlier screens use different controller prompts and cannot be treated as one experiment.", "",
    "| Screen and condition | Measured local | Cached profile A | Cached profile B | Local exact matches |",
    "| --- | ---: | ---: | ---: | ---: |"]
for run_label, arm, profile_id, label in [
    ("native/14b", "basic_rag", "openrouter14b-alibaba", "Native 14B basic"),
    ("native/14b", "iterative_text", "openrouter14b-alibaba", "Native 14B iterative"),
    ("native/122b", "basic_rag", "openrouter122b-alibaba", "Native 122B basic"),
    ("native/122b", "iterative_text", "openrouter122b-alibaba", "Native 122B iterative"),
    ("prior/8b_short", "iterative_text", "openrouter8b-alibaba", "Prior 8B short"),
    ("prior/8b_thinking", "iterative_text", "openrouter8b-alibaba", "Prior 8B thinking"),
    ("prior/14b_short", "iterative_text", "openrouter14b-alibaba", "Prior 14B short"),
    ("prior/14b_thinking", "iterative_text", "openrouter14b-alibaba", "Prior 14B thinking")]:
    s = index[(profile_id, run_label, arm)]
    lines.append(f"| {label} | {f(local_stats(groups[(run_label, arm)])['mean'])} | {f(s['decode_rate_assumption_s']['mean'])} | {f(s['inclusive_rate_proxy_s']['mean'])} | {int(s['observed_correct_unchanged'])}/12 |")
lines += ["", "These selected cached profiles use Alibaba for 14B and 122B. For 8B, the source is an unlabeled cached model-level performance series: attribution to Alibaba is inferred from the current sole endpoint and is not a measured historical Alibaba pair. The Alibaba 122B values use a stale cached provider table. Under Lambda's six distinct hardware calibrations, native 122B projected A means span 1.091–7.663 s for basic and 3.920–26.205 s for iterative. These endpoints are different scenarios, not an uncertainty interval or a provider ranking. Some calibrations are slower than the measured local runs.", "",
    "![Measured local means and selected conditional provider scenarios](provider-latency.png)", "",
    "The figure selects cached Alibaba profiles for 14B/122B, the 8B model-level series and two Lambda configurations to keep it readable. The appendix retains every profile and result group; the selection is not a claim that these providers or engines are best.", "",
    "## Token and timing definitions", "",
    "Let H be measured time outside the local model-call windows, F the profile's TTFT (or cached latency used as its proxy), R its reported token rate, and Gᵢ the recorded generated tokens in logical call i. The two calculations are:", "",
    "- **A — decode-rate assumption:** H + Σᵢ[F + max(Gᵢ − 1, 0) / R].",
    "- **B — inclusive-rate proxy:** H + Σᵢ max(F, Gᵢ / R). This is emitted only when the reported throughput convention is ambiguous.", "",
    "A and B are separate what-if scenarios, not confidence intervals, bounds, or measured provider latency. Lambda publishes mean time per output token (TPOT), so R = 1000 / TPOT_ms and only A is used. Taking the inverse of the reported mean TPOT is a constant-rate scenario; it is not a measurement of the mean reciprocal rate across requests.", "",
    "G includes generated reasoning (including a generated end-think delimiter), action text and final-answer tokens. Sampled EOS tokens and forced input controls are excluded. Reasoning and action belong to one logical controller call; a forced phase closure is a separate input segment, not an extra request. All 254 logical calls are preserved. The earlier final decoder did not save EOS IDs; its sampled-token count is derived from its saved stop reason, and this derivation does not affect G.", "",
    "Local means use saved cold end-to-end times. Each query constructs and processes its context afresh; the model and retrieval index are already loaded. They include retrieval, per-case preparation, controller calls, final generation and measured host overhead; they exclude model loading and shared index setup. Native and earlier timing wrappers differ, so their measured means remain separate. The provider scenarios retain only measured outer host work plus the substituted call formula. Within-call host work, EOS termination and forced-closure prefill latency are not modeled separately.", "",
    "No additional input-token/rate term is added: the supplied TTFT already contains unmatched prompt processing and, depending on source, queue/network work. There is no fitted prefill scaling law. Saved prefill counts remain in the workload ledger. Across the native 122B calls, initial prompts contain 762–2125 tokens and generated outputs contain 1–18 tokens; this differs sharply from Lambda's 8192/1024-token, concurrency-32 workload. The exact bounded reasoning/forced closure protocol would require custom server support and may not be reproducible with an ordinary chat API.", "",
    "## Source limits", "",
    "The [8B OpenRouter page](https://openrouter.ai/qwen/qwen3-8b/pricing) and [14B OpenRouter page](https://openrouter.ai/qwen/qwen3-14b-04-28/uptime) provide cached averages of displayed one-week P50 series, with nearby availability displays ending September 10; exact performance-week boundaries and timezone are unknown. The [122B OpenRouter provider table](https://openrouter.ai/qwen/qwen3.5-122b-a10b/benchmarks) was marked crawled last month and exposes no exact aggregation dates. Its profiles are historical sensitivity scenarios, not current service estimates. Retrieval date is September 11, 2026; it is not the performance measurement date.", "",
    "OpenRouter's [provider integration guide](https://openrouter.ai/docs/guides/community/for-providers) and [latency/performance guide](https://openrouter.ai/docs/guides/best-practices/latency-and-performance) use differing throughput descriptions, motivating A/B. Cached pages also describe their latency label ambiguously; using it as F is an assumption. Current unauthenticated endpoint responses had null performance fields, which were not converted to zero or combined with unrelated fresh metrics. Means of the projected question workloads are not provider mean latencies or provider P50/P95 estimates.", "",
    "[Lambda's benchmark](https://lambda.ai/inference-models/qwen/qwen3.5-122b-a10b) supplies separate mean TTFT and TPOT for 512 requests with 8192 input tokens, 1024 output tokens and maximum concurrency 32. It is a hardware calibration, not a measured serverless endpoint for these short requests. Its execution date, precision and exact engine revisions are unspecified. The six configurations remain separate; aggregate cluster token throughput is not used as single-request decode speed.", "",
    "Profiles match explicit model families, not identical checkpoints or execution. The local 122B model is a 5-bit conversion, and provider precision, sampling, templates and reasoning policies may differ. Such differences can change answers and search trajectories. This report retains the unchanged repository exact-match scorer and supplied aliases for local outputs only; it does not transfer that accuracy to a provider. Sparse successes on 12 development questions do not establish quality equivalence.", "",
    "Original KV relay, incremental-cache and pressure experiments are excluded: ordinary chat APIs do not expose their cache operations. These calculations therefore cover the 96 controller-screen conditions, not the earlier thousands of cache experiments.", "",
    "## Workload appendix", "",
    "Counts are totals across 12 questions per row. Initial prefill excludes forced control input; sampled EOS is shown separately from generated text in the machine-readable ledger.", "",
    "| Run / arm | Logical calls | Initial prefill tokens | Forced input tokens | Generated reasoning | Generated action | Generated final |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for g in workloads["groups"]:
    t = g["totals"]
    lines.append(f"| {g['run_label']} / {g['arm']} | {t['logical_model_calls']} | {t['initial_prefill_tokens']} | {t['forced_control_prefill_tokens']} | {t['reasoning_generated_tokens']} | {t['action_generated_tokens']} | {t['final_generated_tokens']} |")
lines += ["", "## All 14 profiles", "",
    "F is TTFT for Lambda and reported cached latency assumed to be TTFT/base time for OpenRouter, in seconds. R is reported tokens/s; for Lambda it is inverse mean TPOT. OpenRouter profiles support A/B; Lambda profiles support A only. The 8B model-level series has inferred Alibaba attribution. Source links identify each retained rate/latency pair.", "",
    "| Profile ID | Model | Provider / configuration | F (s) | R (tok/s) | Scenarios | Source |",
    "| --- | --- | --- | ---: | ---: | --- | --- |"]
for p in profiles.values():
    lines.append(f"| {p['id']} | {p['model_id'].removeprefix('Qwen/')} | {p['provider']} | {f(p['ttft_s'])} | {f(p['reported_tps'])} | {'A only' if p['metric_semantics'] == 'decode_tps_from_tpot' else 'A/B'} | [{p['source_id']}]({p['source_url']}) |")
lines += ["", "## All 34 result groups", "",
    "Each group has n = 12 questions. A/B values are projected **means**; the CSV additionally contains empirical p50/p95 across these same question workloads. Those percentiles describe workload variation, not uncertainty or provider service percentiles. Missing B means the TPOT source supports only A. Repeated local means identify the same local cohort and must not be pooled across profiles.", "",
    "| Profile ID | Run / arm | n | Measured local mean (s) | Projected A mean (s) | Projected B mean (s) |",
    "| --- | --- | ---: | ---: | ---: | ---: |"]
for s in summary_rows:
    lines.append(f"| {s['profile_id']} | {s['run_label']} / {s['arm']} | 12 | {f(s['measured_local_mean_s'])} | {f(s['scenario_A_mean_s'])} | {f(s['scenario_B_mean_s'])} |")
lines += ["", "## Paired comparisons", "",
    "The [paired CSV](provider-pairs.csv) contains all 17 profile-specific contrasts, each matching the same 12 question IDs. Positive mean differences mean the comparison took longer: iterative minus basic in the native screen, and thinking minus short in the prior screen. Ratios are medians of within-question comparison/baseline ratios, not ratios of cohort means. No confidence intervals are inferred from alternative providers or scenario A/B.", "",
    "## Artifacts and reproduction", "",
    "The [summary CSV](provider-summary.csv), [paired CSV](provider-pairs.csv), [workload ledger](workloads.json), [projection ledger](projections.json), [profiles](profiles.json), [curated source observations](published-source-observations.json) and [audit receipt](projection-audit.json) preserve all included computations. Source/result/data hashes are in the workload ledger; projection hashes pin the workload and profile inputs. Full third-party webpages and documentation are not redistributed with this report.", "",
    "To regenerate the tables and figure from the released JSON ledgers, run from the repository root (no original run directories or model weights are needed):", "",
    "```sh", ".venv/bin/python reports/2026-09-11-large-controller/provider-latency/render_provider_report.py", "```", "",
    "To regenerate workloads and projections, first restore the six preserved source runs and normalized inputs as described in the parent report, then copy the released profile receipts and run:", "",
    "```sh", "mkdir -p runs/large-controller/provider-latency",
    "cp reports/2026-09-11-large-controller/provider-latency/profiles.json reports/2026-09-11-large-controller/provider-latency/published-source-observations.json runs/large-controller/provider-latency/",
    ".venv/bin/python scripts/project_controller_latency.py --profiles runs/large-controller/provider-latency/profiles.json",
    ".venv/bin/python reports/2026-09-11-large-controller/provider-latency/render_provider_report.py --input-dir runs/large-controller/provider-latency", "```", "",
    "The renderer uses its own directory by default or an explicit `--input-dir`. It performs no network calls or inference. Regenerating projection inputs after changing the extraction script changes the extractor hash; preserve the released JSON files for exact replay of the reported calculations.", ""]
(HERE / "provider-latency.md").write_text("\n".join(lines))
receipt = {"schema_version": "provider-latency-render-v1", "input_sha256": {name: sha(HERE / name) for name in
    ["workloads.json", "projections.json", "profiles.json", "published-source-observations.json"]},
    "renderer_sha256": sha(Path(__file__)), "output_sha256": {name: sha(HERE / name) for name in
    ["provider-latency.md", "provider-summary.csv", "provider-pairs.csv", "provider-latency.png"]},
    "summary_rows": len(summary_rows), "paired_rows": len(paired_rows), "distinct_questions": 12}
(HERE / "provider-render-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({"summary_rows": len(summary_rows), "paired_rows": len(paired_rows), "figure": str(HERE / "provider-latency.png")}))
