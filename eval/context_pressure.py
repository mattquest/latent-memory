"""Matched-trace retention/interference ablation; not adaptive pressure retrieval.

Only new files implement this diagnostic. Model imports remain in the launcher.
All source questions are included, irrespective of correctness or trace length.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .adaptive_experiments import PROLOGUE, CHAT_END, document_text, final_text, append_record, read_records, write_json
from .real_experiments import answer_scores, clean_prediction, percentile, bootstrap_interval
from .retrieval import terms

PROTOCOL = "matched-trace-pressure-v1"
ARMS = ("retained_text", "retained_kv_bf16", "retained_kv_int8", "rolling_summary")
MEMORY_START = "\n<retained_memory>\n"
MEMORY_END = "\n</retained_memory>\n"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PressureConfig:
    dataset: str
    corpus: str
    source_dir: str
    output_dir: str
    evidence_budget: int = 1536
    summary_budget: int = 384
    max_answer_tokens: int = 48
    max_model_context: int = 8192
    source_rounds: int = 5
    seed: int = 20260914
    expected_questions: int = 96
    max_runtime_seconds: float = 18000
    max_case_seconds: float = 240
    resume: bool = True

    def validate(self):
        if not 0 < self.summary_budget < self.evidence_budget <= 2048:
            raise ValueError("Require 0 < summary budget < evidence budget <= 2048")
        if min(self.max_answer_tokens, self.max_model_context, self.source_rounds, self.expected_questions,
               self.max_runtime_seconds, self.max_case_seconds) <= 0 or self.max_model_context > 8192:
            raise ValueError("Invalid count, runtime or backend context limit")


def rank_retention(question, fragments):
    """BM25 over only delivered text, with sorted terms and content-ID ties."""
    counts = {key: Counter(terms(value["visible_text"])) for key, value in fragments.items()}
    lengths = {key: sum(value.values()) for key, value in counts.items()}
    average = sum(lengths.values()) / max(1, len(lengths)) or 1
    scores = {key: 0.0 for key in counts}
    for term in sorted(set(terms(question))):
        df = sum(term in value for value in counts.values())
        idf = math.log(1 + (len(counts) - df + .5) / (df + .5))
        for key in sorted(counts):
            frequency = counts[key].get(term, 0)
            if frequency:
                scores[key] += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * lengths[key] / average))
    return sorted(scores, key=lambda key: (-scores[key], key)), scores


def retain(question, fragments, budget):
    ranking, scores = rank_retention(question, fragments)
    chosen, used = [], 0
    for key in ranking:
        if used + len(fragments[key]["tokens"]) <= budget:
            chosen.append(key)
            used += len(fragments[key]["tokens"])
    # Presentation follows arrival order; selection alone uses relevance.
    chosen_set = set(chosen)
    return [key for key in fragments if key in chosen_set], scores


def prepare_replay(backend, corpus, source_row, source_config):
    """Reconstruct and verify exact source token prefixes, without using labels."""
    prologue = backend.encode(PROLOGUE)
    fragments, rounds, accumulated = {}, [], []
    for event in source_row["trace"]:
        if event["event"] != "retrieval":
            continue
        new = []
        for key in event["new_document_ids"]:
            if key in fragments or key not in corpus:
                raise ValueError("Repeated or unknown source trace document")
            original = backend.encode(document_text(corpus[key]))
            tokens = original[:source_config["max_document_tokens"]]
            if not tokens:
                raise ValueError("Empty source document fragment")
            fragments[key] = {"tokens": tokens, "visible_text": backend.decode_tokens(tokens),
                              "truncated": len(tokens) < len(original), "original_tokens": len(original)}
            accumulated.append(key)
            new.append(key)
        if event["accumulated_document_ids"] != accumulated:
            raise ValueError("Source trace accumulated-document order mismatch")
        full = prologue + [token for key in accumulated for token in fragments[key]["tokens"]]
        if sha(json.dumps(full).encode()) != event["evidence_token_ids_sha256"]:
            raise ValueError("Source trace token hash mismatch; source tokenizer/prompt/cap changed")
        if sum(len(fragments[key]["tokens"]) for key in accumulated) != event["evidence_tokens"]:
            raise ValueError("Source trace evidence token count mismatch")
        rounds.append({"round": event["round"], "new_document_ids": new})
    if not rounds or accumulated != source_row["evidence_ids"]:
        raise ValueError("Missing/inconsistent retrieval trace")
    return fragments, rounds


def run_pressure_case(backend, corpus, example, source_row, source_config, arm, config):
    if arm not in ARMS:
        raise ValueError("Unsupported pressure arm")
    backend.synchronize()
    started = time.perf_counter()
    timing = defaultdict(float)
    counts = Counter()
    calls, retention_trace, summary_trace = [], [], []
    fragments = {}

    def check_time():
        if time.perf_counter() - started > config.max_case_seconds:
            raise TimeoutError("Pressure case runtime budget exceeded")

    def model_call(kind, fn):
        check_time()
        backend.synchronize()
        before = time.perf_counter()
        result = fn()
        backend.synchronize()
        timing[kind] += (time.perf_counter() - before) * 1000
        return result

    before = time.perf_counter()
    prepared, rounds = prepare_replay(backend, corpus, source_row, source_config)
    prologue = backend.encode(PROLOGUE)
    final_suffix = backend.encode(final_text(example["question"]))
    memory_start, memory_end = backend.encode(MEMORY_START), backend.encode(MEMORY_END)
    timing["trace_validation_and_tokenization_ms"] += (time.perf_counter() - before) * 1000
    inner_summary_cap = config.summary_budget - len(memory_start) - len(memory_end)
    if inner_summary_cap < 16:
        raise ValueError("Summary budget is too small after wrapper tokens")
    native = arm.startswith("retained_kv_")
    cache_blocks, prefix, selected = {}, None, []
    peak_stored_bytes = 0
    if native:
        prefix = model_call("cache_build_ms", lambda: backend.prefill(prologue))
        counts["cache_prefill_tokens"] += len(prologue)
    memory, recent, summarized_ids = [], [], []

    def memory_tokens():
        return memory_start + memory + memory_end if memory else []

    def generate(evidence, suffix, maximum, kind, kv=None):
        if len(evidence) > config.evidence_budget:
            raise AssertionError("Evidence input budget exceeded")
        total = len(prologue) + len(evidence) + len(suffix) + maximum
        if total > min(config.max_model_context, getattr(backend, "max_context", 8192)):
            raise ValueError("Complete pressure model context reservation exceeds cap")
        calls.append({"kind": kind, "evidence_input_tokens": len(evidence), "fixed_prompt_question_tokens": len(prologue) + len(suffix),
                      "output_reservation_tokens": maximum, "total_context_reservation_tokens": total})
        prompt = suffix if kv is not None else prologue + evidence + suffix
        result = model_call(kind + "_ms", lambda: backend.decode(kv, prompt, max_tokens=maximum))
        counts[kind + "_prefill_tokens"] += len(prompt)
        counts[kind + "_generated_tokens"] += len(result)
        counts["max_evidence_input_tokens"] = max(counts["max_evidence_input_tokens"], len(evidence))
        counts["max_model_context_tokens"] = max(counts["max_model_context_tokens"], total)
        calls[-1]["generated_tokens"] = len(result)
        calls[-1]["stop_reason"] = getattr(backend, "last_decode_stats", {}).get("stop_reason")
        return result

    for event in rounds:
        check_time()
        before = time.perf_counter()
        for key in event["new_document_ids"]:
            fragments[key] = prepared[key]
        if arm != "rolling_summary":
            selected, scores = retain(example["question"], fragments, config.evidence_budget)
            timing["retention_ms"] += (time.perf_counter() - before) * 1000
            retention_trace.append({"round": event["round"], "retained_document_ids": selected.copy(),
                                    "retained_original_tokens": sum(len(fragments[key]["tokens"]) for key in selected),
                                    "ranking_scores": scores})
            if native:
                cache_blocks = {key: block for key, block in cache_blocks.items() if key in selected}
                for key in selected:
                    if key in cache_blocks:
                        continue
                    block = model_call("cache_build_ms", lambda key=key: backend.prefill(fragments[key]["tokens"]))
                    counts["cache_prefill_tokens"] += len(fragments[key]["tokens"])
                    counts["document_cache_builds"] += 1
                    if arm == "retained_kv_int8":
                        block = model_call("quantization_ms", lambda block=block: backend.quantize(block, bits=8))
                    cache_blocks[key] = block
                    del block
                peak_stored_bytes = max(peak_stored_bytes, prefix.nbytes + sum(block.nbytes for block in cache_blocks.values()))
        else:
            recent.extend(event["new_document_ids"])
            expired = []
            # Reserve the configured summary slot even before the first summary.
            recent_budget = config.evidence_budget - config.summary_budget
            while recent and sum(len(fragments[key]["tokens"]) for key in recent) > recent_budget:
                expired.append(recent.pop(0))
            batches, batch, size = [], [], 0
            for key in expired:
                length = len(fragments[key]["tokens"])
                if length > recent_budget:
                    raise ValueError("A capped source paragraph exceeds summary input allowance")
                if size + length > recent_budget:
                    batches.append(batch)
                    batch, size = [], 0
                batch.append(key)
                size += length
            if batch:
                batches.append(batch)
            timing["retention_ms"] += (time.perf_counter() - before) * 1000
            for batch in batches:
                evidence = memory_tokens() + [token for key in batch for token in fragments[key]["tokens"]]
                suffix = backend.encode("\nQuestion to support later: " + example["question"] +
                    "\nTask: Update the retained factual memory using only the previous memory and documents above. "
                    "Preserve exact names, numbers, relations, and unresolved links relevant to this question. "
                    "Do not answer the question or invent missing facts. Write concise factual notes only.\n" + CHAT_END)
                output = generate(evidence, suffix, inner_summary_cap, "summary_generation")
                memory = list(output)
                summarized_ids.extend(batch)
                summary_trace.append({"round": event["round"], "source_document_ids": batch,
                                      "memory_token_ids": memory.copy(), "memory_text": backend.decode_tokens(memory),
                                      "memory_token_ids_sha256": sha(json.dumps(memory).encode()),
                                      "evidence_input_tokens": len(evidence)})
            selected = recent.copy()
            retention_trace.append({"round": event["round"], "retained_document_ids": selected.copy(),
                                    "summary_tokens_with_wrappers": len(memory_tokens()),
                                    "retained_original_tokens": sum(len(fragments[key]["tokens"]) for key in selected)})
    evidence = (memory_tokens() if arm == "rolling_summary" else []) + [token for key in selected for token in fragments[key]["tokens"]]
    if native:
        blocks = [prefix, *(cache_blocks[key] for key in selected)]
        prefix = model_call("cache_assembly_ms", lambda: backend.concat(blocks, bridge_ratio=source_config["bridge_ratio"]))
        counts["bridge_recomputed_tokens"] += int(prefix.metadata.get("bridge_tokens_recomputed", 0))
        if tuple(prefix.token_ids) != tuple(prologue + evidence):
            raise AssertionError("Native and text retained token streams differ")
    output = generate(evidence, final_suffix, config.max_answer_tokens, "final_generation", prefix if native else None)
    raw = backend.decode_tokens(output)
    backend.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    timing["other_replay_ms"] = max(0.0, elapsed - sum(timing.values()))
    total_source_tokens = sum(len(item["tokens"]) for item in prepared.values())
    support = set(example.get("supporting_context_ids", []))
    controller_ms = source_row["component_ms"].get("controller_ms", 0)
    source_search_ms = source_row["component_ms"].get("retrieval_ms", 0)
    prediction = clean_prediction(raw)
    return {"status": "ok", "protocol_version": PROTOCOL, "id": case_id(example["id"], arm, config),
            "example_id": example["id"], "arm": arm, "question": example["question"], "answers": example["answers"],
            "category": example.get("category"), "source_result_id": source_row["id"],
            "source_trace_sha256": sha(json.dumps(source_row["trace"], sort_keys=True).encode()),
            "prediction": prediction, "raw_output": raw, "output_token_ids": output,
            "scores": answer_scores(prediction, example["answers"]), "counts": dict(counts), "component_ms": dict(timing),
            "replay_cold_ms": elapsed, "source_controller_ms": controller_ms, "source_retrieval_ms": source_search_ms,
            "replay_plus_source_controller_and_retrieval_ms": elapsed + controller_ms + source_search_ms,
            "timing_note": "Replay plus selected source components is an accounting sum, not end-to-end adaptive pressure latency.",
            "evidence_budget": config.evidence_budget, "cumulative_source_tokens": total_source_tokens,
            "trace_overflow": total_source_tokens > config.evidence_budget, "retained_evidence_tokens": len(evidence),
            "retained_evidence_token_ids_sha256": sha(json.dumps(evidence).encode()),
            "retained_document_ids": selected, "summarized_source_document_ids": summarized_ids,
            "retained_truncated_document_ids": [key for key in selected if prepared[key]["truncated"]],
            "retained_original_support_id_coverage": len(support & set(selected)) / len(support) if support else None,
            "summarized_source_support_id_coverage": len(support & set(summarized_ids)) / len(support) if support else None,
            "support_coverage_note": "ID overlap only; truncated text or a generated summary may omit the supporting fact.",
            "retention_trace": retention_trace, "summary_trace": summary_trace, "model_calls": calls,
            "cache": {"peak_retained_block_bytes": peak_stored_bytes, "final_attention_prefix_bytes": prefix.nbytes if native else None,
                      "stored_precision": "int8" if arm.endswith("int8") else "bf16" if native else None,
                      "attention_precision": "bf16", "cache_policy": "eager build on retained entry; evicted blocks released; no cross-question reuse" if native else "no native retained cache"},
            "memory": backend.memory_stats()}


def case_id(example_id, arm, config):
    return sha(json.dumps([PROTOCOL, example_id, arm, config.evidence_budget, config.summary_budget]).encode())[:24]


def aggregate(records, references, config, stop_reason, planned):
    latest = {row["id"]: row for row in records}
    successful = [row for row in latest.values() if row["status"] == "ok"]
    groups, metrics = defaultdict(list), []
    for row in successful:
        groups[(row["arm"], "all")].append(row)
        if row["trace_overflow"]:
            groups[(row["arm"], "trace_overflow")].append(row)
    for (arm, subset), values in sorted(groups.items()):
        coverages = [row["retained_original_support_id_coverage"] for row in values
                     if row["retained_original_support_id_coverage"] is not None]
        metrics.append({"arm": arm, "subset": subset, "n": len(values),
                        "exact_match": statistics.mean(row["scores"]["exact_match"] for row in values),
                        "f1": statistics.mean(row["scores"]["f1"] for row in values),
                        "replay_p50_ms": percentile([row["replay_cold_ms"] for row in values], .5),
                        "replay_p95_ms": percentile([row["replay_cold_ms"] for row in values], .95),
                        "mean_source_controller_ms": statistics.mean(row["source_controller_ms"] for row in values),
                        "mean_source_retrieval_ms": statistics.mean(row["source_retrieval_ms"] for row in values),
                        "mean_retained_tokens": statistics.mean(row["retained_evidence_tokens"] for row in values),
                        "mean_retained_original_support_id_coverage": statistics.mean(coverages) if coverages else None})
    by_id = defaultdict(dict)
    for row in successful:
        by_id[row["example_id"]][row["arm"]] = row
    paired = []
    for arm in ARMS[1:]:
        for subset in ("all", "trace_overflow"):
            pairs = [(mapping["retained_text"], mapping[arm]) for mapping in by_id.values()
                     if "retained_text" in mapping and arm in mapping and (subset == "all" or mapping[arm]["trace_overflow"])]
            if not pairs:
                continue
            differences = [right["scores"]["exact_match"] - left["scores"]["exact_match"] for left, right in pairs]
            paired.append({"arm": arm, "reference": "retained_text", "subset": subset, "n": len(pairs),
                           "em_difference": statistics.mean(differences),
                           "paired_bootstrap_ci95": bootstrap_interval(differences, config.seed) if len(set(differences)) > 1 else None})
    reference_groups, reference_metrics = defaultdict(list), []
    for row in references:
        reference_groups[(row["case"]["arm"], "all")].append(row)
        if row.get("controller_trace_overflow"):
            reference_groups[(row["case"]["arm"], "trace_overflow")].append(row)
    for (arm, subset), values in sorted(reference_groups.items()):
        reference_metrics.append({"arm": arm, "subset": subset, "n": len(values),
                                  "exact_match": statistics.mean(row["scores"]["exact_match"] for row in values),
                                  "f1": statistics.mean(row["scores"]["f1"] for row in values),
                                  "original_end_to_end_p50_ms": percentile([row["cold_end_to_end_ms"] for row in values], .5),
                                  "original_end_to_end_p95_ms": percentile([row["cold_end_to_end_ms"] for row in values], .95),
                                  "role": "Original source condition; original retrieval/evidence budget and end-to-end cost"})
    complete = stop_reason == "complete" and len(successful) == planned and all(row["status"] == "ok" for row in latest.values())
    return {"protocol_version": PROTOCOL, "status": "complete" if complete else "INCOMPLETE",
            "stop_reason": stop_reason, "planned_runs": planned, "successful_runs": len(successful),
            "errors": sum(row["status"] != "ok" for row in latest.values()), "metrics": metrics, "paired": paired,
            "source_references": references, "reference_metrics": reference_metrics, "limitations": [
                "Shared-trace retention/interference ablation; controller decisions were made in the original larger-context run.",
                "All questions are included; overflow is defined by source-token exposure, not answer correctness.",
                "Evidence budget counts source tokens, KV slots and summary wrappers. Fixed prompts/questions and output reservations are additional, explicitly logged.",
                "Int8 reduces stored bytes, not attention positions. No trained latent compressor or latent query controller is tested.",
                "Most questions may have sufficient support within the smaller window. Overflow does not prove necessary information exceeds context.",
                "Support-ID coverage is annotation provenance, not proof truncated text or generated notes preserve the facts.",
                "Basic RAG and larger-context adaptive references retain their original retrieval settings and measured end-to-end costs."]}


def report_tables(summary):
    lines = ["# Matched-trace context pressure", "", f"Status: **{summary['status']}**.", "",
             "All source questions are retained. The secondary overflow subset is defined by cumulative source-token exposure, not correctness.", "",
             "| Arm | Subset | n | EM | F1 | Replay p50 / p95 (s) | Mean source planning / search (s) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary["metrics"]:
        lines.append(f"| {row['arm']} | {row['subset']} | {row['n']} | {row['exact_match']:.1%} | {row['f1']:.1%} | "
                     f"{row['replay_p50_ms']/1000:.2f} / {row['replay_p95_ms']/1000:.2f} | "
                     f"{row['mean_source_controller_ms']/1000:.2f} / {row['mean_source_retrieval_ms']/1000:.2f} |")
    lines += ["", "## Original references", "", "These timings include original search and planning; replay timings above exclude those separately listed costs.", "",
              "| Original arm | Subset | n | EM | Original end-to-end p50 / p95 (s) |", "| --- | --- | --- | --- | --- |"]
    for row in summary["reference_metrics"]:
        lines.append(f"| {row['arm']} | {row['subset']} | {row['n']} | {row['exact_match']:.1%} | "
                     f"{row['original_end_to_end_p50_ms']/1000:.2f} / {row['original_end_to_end_p95_ms']/1000:.2f} |")
    lines += ["", "## Limits", "", *["- " + note for note in summary["limitations"]], ""]
    return "\n".join(lines)


def run_pressure_suite(backend, config):
    config.validate()
    if "mock" in str(getattr(backend, "name", "")).lower() or not hasattr(backend, "concat"):
        raise ValueError("Real native backend required for scored pressure runs")
    source_dir, directory = Path(config.source_dir), Path(config.output_dir)
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    source_summary = json.loads((source_dir / "summary.json").read_text())
    if source_summary.get("stop_reason") != "complete":
        raise ValueError("Source adaptive run must be complete before replay")
    source_raw = (source_dir / "results.jsonl").read_bytes()
    dataset_raw, corpus_raw = Path(config.dataset).read_bytes(), Path(config.corpus).read_bytes()
    identity = source_manifest["identity"]
    if sha(dataset_raw) != identity["questions_sha256"] or sha(corpus_raw) != identity["corpus_sha256"]:
        raise ValueError("Pressure inputs differ from source adaptive dataset/corpus")
    if backend.metadata() != identity["backend"]:
        raise ValueError("Backend/tokenizer identity differs from source run")
    source_config = identity["config"]
    if config.max_answer_tokens != source_config["max_answer_tokens"]:
        raise ValueError("Use the same final answer cap as the source run")
    questions = [json.loads(line) for line in dataset_raw.splitlines() if line.strip()]
    if len(questions) != config.expected_questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Must replay the full expected question set")
    corpus = {row["id"]: {key: row[key] for key in ("id", "title", "text")}
              for row in (json.loads(line) for line in corpus_raw.splitlines() if line.strip())}
    source_rows = {row["id"]: row for row in (json.loads(line) for line in source_raw.splitlines() if line.strip())}
    selected, references = {}, []
    for row in source_rows.values():
        if row["status"] != "ok":
            raise ValueError("Source contains unresolved failed cases")
        c = row["case"]
        if c["arm"] == "iterative_text" and c["max_rounds"] == config.source_rounds:
            if row["example_id"] in selected:
                raise ValueError("Duplicate source controller trace")
            selected[row["example_id"]] = row
        if c["arm"] == "basic_rag" or (c["arm"] == "iterative_text" and c["max_rounds"] == config.source_rounds):
            references.append({key: row[key] for key in ("id", "example_id", "case", "scores", "cold_end_to_end_ms", "evidence_ids")})
    if set(selected) != {q["id"] for q in questions}:
        raise ValueError("One completed iterative-text trace is required for every question")
    basic_ids = [row["example_id"] for row in references if row["case"]["arm"] == "basic_rag"]
    if len(basic_ids) != len(questions) or set(basic_ids) != set(selected):
        raise ValueError("One completed Basic RAG reference is required for every question")
    for example in questions:
        if selected[example["id"]]["question"] != example["question"]:
            raise ValueError("Source controller question differs from the pinned dataset")
    for row in references:
        row["controller_trace_overflow"] = selected[row["example_id"]]["counts"]["retrieved_evidence_tokens"] > config.evidence_budget
    directory.mkdir(parents=True, exist_ok=True)
    identity_config = asdict(config)
    for key in ("output_dir", "resume", "max_runtime_seconds"):
        identity_config.pop(key)
    root = Path(__file__).resolve().parents[1]
    run_identity = {"protocol": PROTOCOL, "config": identity_config, "source_manifest_sha256": sha((source_dir / "manifest.json").read_bytes()),
                    "source_results_sha256": sha(source_raw), "dataset_sha256": sha(dataset_raw), "corpus_sha256": sha(corpus_raw),
                    "backend": backend.metadata(), "source_sha256": {name: sha((root / name).read_bytes()) for name in
                    ("eval/context_pressure.py", "scripts/run_context_pressure.py", "eval/adaptive_experiments.py", "eval/real_experiments.py", "eval/retrieval.py", "engine/backends_mlx.py")}}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if not config.resume or json.loads(manifest_path.read_text())["identity"] != run_identity:
            raise ValueError("Pressure run identity changed; use a new output directory")
    else:
        archives = []
        for name, raw in (("source-results.jsonl", source_raw), ("questions.jsonl", dataset_raw), ("corpus.jsonl", corpus_raw)):
            payload = gzip.compress(raw, mtime=0)
            (directory / (name + ".gz")).write_bytes(payload)
            archives.append({"artifact": name + ".gz", "source_sha256": sha(raw), "artifact_sha256": sha(payload)})
        write_json(manifest_path, {"identity": run_identity, "source_manifest": source_manifest, "archives": archives,
                                  "data_license": {"name": "CC-BY-4.0", "attribution": "Trivedi et al. (2022), MuSiQue",
                                                   "source": "https://github.com/StonyBrookNLP/musique",
                                                   "modifications": "See data/adaptive_musique.manifest.json and docs/adaptive-data.md"}})
    path = directory / "results.jsonl"
    records = read_records(path, repair_trailing=True)
    completed = {row["id"] for row in records if row["status"] == "ok"}
    planned, started, stop_reason = len(questions) * len(ARMS), time.monotonic(), "complete"
    try:
        for index, question in enumerate(questions):
            arms = list(ARMS)
            random.Random(config.seed + index).shuffle(arms)
            for arm in arms:
                identifier = case_id(question["id"], arm, config)
                if identifier in completed:
                    continue
                if time.monotonic() - started > config.max_runtime_seconds:
                    stop_reason = "runtime_budget"
                    break
                backend.clear_cache()
                try:
                    row = run_pressure_case(backend, corpus, question, selected[question["id"]], source_config, arm, config)
                except Exception as error:
                    row = {"status": "error", "id": identifier, "example_id": question["id"], "arm": arm,
                           "error_type": type(error).__name__, "error": str(error)}
                    append_record(path, row)
                    records.append(row)
                    raise
                append_record(path, row)
                records.append(row)
                completed.add(identifier)
                write_json(directory / "checkpoint.json", {"completed": len(completed), "planned": planned, "last_id": identifier})
                print(json.dumps({"completed": len(completed), "planned": planned, "arm": arm, "example_id": question["id"],
                                  "overflow": row["trace_overflow"], "replay_ms": round(row["replay_cold_ms"]), "em": row["scores"]["exact_match"]}), flush=True)
            if stop_reason != "complete":
                break
    except BaseException:
        stop_reason = "failed_or_interrupted"
        raise
    finally:
        summary = aggregate(records, references, config, stop_reason, planned)
        write_json(directory / "summary.json", summary)
        (directory / "tables.md").write_text(report_tables(summary))
    return summary
