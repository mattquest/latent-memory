"""Reproducible real-model diagnostic experiments for the KV relay prototype.

These experiments use a fixed, disclosed evidence schedule and one unadapted
Qwen3 model. They do not implement the trained planner/verifier or establish
the proposed BEAM-scale, adaptive multi-agent thesis. Every scored answer is
decoded by the real backend; unavailable functionality fails visibly.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import statistics
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "fixed-evidence-qwen3-v3-bm25-json"
SYSTEM = ("Use only the supplied documents to answer the question. "
          "Documents may contain arbitrary private facts. Do not guess. "
          "Give only the short answer, without explanation. If the documents "
          "do not establish the answer, answer UNKNOWN. Later explicit updates "
          "supersede earlier versions of the same fact. If equally authoritative "
          "current sources disagree, answer CONFLICT.")
PROLOGUE = ("<|im_start|>system\n" + SYSTEM +
            "<|im_end|>\n<|im_start|>user\nDocuments:\n")
BEAM_SYSTEM = ("Answer the user's question using only the supplied conversation excerpts. "
               "Respect the user's latest explicit preferences, corrections, and updated facts. "
               "Use chronology and source roles when reconciling facts. If the excerpts do not "
               "contain requested information, say what is unknown without guessing. Distinguish "
               "conflicting claims when they cannot be resolved. Give a concise, useful answer "
               "with enough detail to address every part of the question.")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    output_dir: str
    experiments: tuple[str, ...] = ("E1", "E3", "E4", "E5", "E6")
    limit: int = 20
    seed: int = 42
    hops: tuple[int, ...] = (1, 3, 5, 10, 20)
    equal_hops: int = 3
    bridge_ratios: tuple[float, ...] = (0.0, 0.1, 0.2, 1.0)
    precisions: tuple[str, ...] = ("bf16", "int8", "int4")
    bridge_ratio: float = 0.2
    relay_precision: str = "int8"
    latent_precisions: tuple[str, ...] = ()
    max_context_tokens: int = 4096
    max_document_tokens: int = 512
    max_answer_tokens: int = 48
    max_relay_tokens: int = 192
    max_runtime_seconds: float = 18000
    evidence_per_hop: int = 2
    schedule: str = "lexical"
    resume: bool = True
    repeats: int = 1

    def validate(self) -> None:
        if not set(self.experiments) <= {"E1", "E2", "E3", "E4", "E5", "E6"}:
            raise ValueError("Experiments must be E1 through E6")
        if self.schedule not in {"lexical", "source", "oracle"}:
            raise ValueError("schedule must be lexical, source, or oracle")
        if min(self.limit, self.max_context_tokens, self.max_document_tokens,
               self.max_answer_tokens, self.max_relay_tokens, self.equal_hops,
               self.evidence_per_hop, self.repeats, *self.hops) < 1:
            raise ValueError("Counts and token budgets must be positive")
        if any(not 0 <= x <= 1 for x in (*self.bridge_ratios, self.bridge_ratio)):
            raise ValueError("Recompute ratios must be within [0, 1]")
        if not set(self.precisions) <= {"bf16", "int8", "int4"}:
            raise ValueError("Unsupported precision")
        if self.relay_precision not in {"bf16", "int8", "int4"}:
            raise ValueError("Unsupported relay precision")
        if not set(self.latent_precisions) <= {"bf16", "int8", "int4"}:
            raise ValueError("Unsupported latent precision")
        if len(self.latent_precisions) != len(set(self.latent_precisions)):
            raise ValueError("Duplicate latent precision")


@dataclass(frozen=True)
class Case:
    experiment: str
    arm: str
    hops: int = 1
    control: str = "correct"
    bridge_ratio: float = 0.2
    precision: str = "bf16"
    update_policy: str = "all"
    repeat: int = 0


@dataclass
class Counts:
    host_output_tokens: int = 0
    internal_decoded_tokens: int = 0
    query_prefill_tokens: int = 0
    ingest_prefill_tokens: int = 0
    bridge_recomputed_tokens: int = 0
    evidence_tokens: int = 0
    max_prefix_tokens: int = 0


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(c for c in value if c not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_scores(prediction: str, answers: list[str]) -> dict[str, float]:
    """Standard SQuAD/Hotpot-style exact match and token overlap F1."""
    p = normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0
    for answer in answers:
        a = normalize_answer(answer)
        best_em = max(best_em, float(p == a))
        pt, at = p.split(), a.split()
        if p in {"yes", "no", "noanswer"} or a in {"yes", "no", "noanswer"}:
            f1 = float(p == a)
        elif not pt or not at:
            f1 = float(pt == at)
        else:
            common = sum((collections.Counter(pt) & collections.Counter(at)).values())
            f1 = 2 * common / (len(pt) + len(at))
        best_f1 = max(best_f1, f1)
    return {"exact_match": best_em, "f1": best_f1}


def score_example(prediction: str, example: dict) -> dict[str, float]:
    """Opaque private codes are exact strings, not punctuation-normalized prose."""
    if example.get("dataset", "").startswith("synthetic_"):
        expected = example["answers"]
        if all(re.fullmatch(r"[0-9]{6}", answer) for answer in expected):
            matched = float(prediction.strip() in expected)
            return {"exact_match": matched, "f1": matched}
        if all(answer.casefold() == "conflict" for answer in expected):
            matched = float(prediction.strip().casefold() == "conflict")
            return {"exact_match": matched, "f1": matched}
    return answer_scores(prediction, example["answers"])


def wilson_interval(successes: float, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * q
    lo = math.floor(i)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (i - lo)


def bootstrap_interval(values: list[float], seed: int = 42, draws: int = 2000) -> list[float]:
    if not values:
        return [0.0, 1.0]
    if min(values) == max(values):
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def clean_prediction(text: str) -> str:
    """Strip generated chat wrappers; preserve other text for strict scoring."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        text = text.split(marker, 1)[0]
    return text.strip()


def read_dataset(path: str, limit: int | None = None, *,
                 include_timing_sidecars: bool = False) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            value.setdefault("evidence", value.get("contexts", []))
            if not all(k in value for k in ("id", "question", "answers", "evidence")):
                raise ValueError(f"Missing dataset fields at {path}:{line_number}")
            if not isinstance(value["answers"], list) or not value["answers"]:
                raise ValueError(f"answers must be a nonempty list at line {line_number}")
            if not value["evidence"]:
                raise ValueError(f"No evidence at line {line_number}")
            examples.append(value)
            if limit and len(examples) >= limit:
                break
    if len({x["id"] for x in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    if not examples:
        raise ValueError("Dataset is empty")
    if include_timing_sidecars:
        _attach_timing_sidecars(examples, Path(path))
    return examples


def _attach_timing_sidecars(examples: list[dict], dataset_path: Path) -> None:
    """Hydrate cost-only measurements after semantic dataset identity checks.

    Optional absent receipts do not affect answers. Present receipts must match
    their semantic source and canonical example hashes. Never merge arbitrary
    sidecar fields into model inputs, case definitions, or retrieval ranking.
    """
    directory = dataset_path.resolve().parent
    cache = {}

    def safe_file(name):
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or "/" in name or "\\" in name or Path(name).is_absolute()):
            raise ValueError("Timing receipt references must be safe basenames")
        path = directory / name
        if path.resolve().parent != directory:
            raise ValueError("Timing receipt reference escapes dataset directory")
        return path

    def measurement(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Timing measurements must be finite nonnegative numbers")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Timing measurements must be finite nonnegative numbers")
        return float(value)

    for example in examples:
        metadata = example.get("metadata", {})
        name = metadata.get("timing_sidecar")
        if name is None:
            continue
        receipt_path = safe_file(name)
        if not receipt_path.exists():
            metadata["timing_receipt_status"] = "missing_optional_sidecar"
            continue
        if name not in cache:
            raw = receipt_path.read_bytes()
            receipt = json.loads(raw)
            if receipt.get("schema_version") != 1:
                raise ValueError("Unsupported timing receipt schema")
            semantic_path = safe_file(receipt.get("semantic_dataset"))
            semantic_digest = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
            if semantic_digest != receipt.get("semantic_dataset_sha256"):
                raise ValueError("Timing receipt semantic dataset hash mismatch")
            cache[name] = receipt, hashlib.sha256(raw).hexdigest()
        receipt, receipt_digest = cache[name]
        query = receipt.get("queries", {}).get(example["id"])
        if not isinstance(query, dict):
            raise ValueError("Timing receipt lacks the requested example")
        semantic_digest = hashlib.sha256(json.dumps(
            example, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if semantic_digest != query.get("semantic_example_sha256"):
            raise ValueError("Timing receipt semantic example hash mismatch")
        metadata["retrieval_seconds"] = measurement(query.get("retrieval_seconds"))
        metadata["index_preparation_seconds"] = measurement(receipt.get("index_preparation_seconds"))
        metadata["timing_provenance"] = {"sidecar": name, "sidecar_sha256": receipt_digest}


def schedule_evidence(example: dict, hops: int, config: ExperimentConfig,
                      update_policy: str = "all") -> list[list[dict]]:
    """Fixed retrieval; public ranking never uses answers/support annotations."""
    documents = list(example["evidence"])
    if update_policy == "current":
        documents = [d for d in documents if not d.get("superseded_by")
                     and not d.get("superseded", False)
                     and d.get("is_current", True) and d.get("active", True)]
    if config.schedule == "oracle":
        groups = example.get("hop_context_ids")
        if not groups:
            raise ValueError("oracle schedule requires explicit hop_context_ids")
        by_id = {d["id"]: d for d in documents}
        schedule = [[by_id[did] for did in group if did in by_id] for group in groups]
        return (schedule + [[] for _ in range(hops)])[:hops]
    if config.schedule == "lexical":
        from .retrieval import BM25Index
        if documents:
            documents = [document for document, _ in BM25Index(documents).search(
                example["question"], k=len(documents), max_per_source=len(documents))]
    return [documents[i * config.evidence_per_hop:(i + 1) * config.evidence_per_hop]
            for i in range(hops)]


def document_text(document: dict) -> str:
    # IDs used in public benchmarks can contain answers; expose titles and text only.
    title = document.get("title", "")
    return f"\n<document>\n{title}\n{document['text']}\n</document>\n"


def _is_rubric_task(example: dict) -> bool:
    return bool(example.get("metadata", {}).get("requires_rubric_judge"))


def prologue_text(example: dict) -> str:
    if _is_rubric_task(example):
        return "<|im_start|>system\n" + BEAM_SYSTEM + "<|im_end|>\n<|im_start|>user\nConversation excerpts:\n"
    return PROLOGUE


def suffix_text(question: str, rubric_task: bool = False) -> str:
    instruction = ("Give a concise answer addressing all requested details using the excerpts."
                   if rubric_task else "Answer only with the short answer. If unavailable answer UNKNOWN.")
    return ("\nQuestion: " + question + "\n" + instruction +
            "<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


def cases_for(config: ExperimentConfig, example: dict) -> list[Case]:
    cases: list[Case] = []
    r = config.bridge_ratio
    arms = [("text", "bf16"), *[("latent", precision) for precision in
             (config.latent_precisions or (config.relay_precision,))], ("direct_text", "bf16")]
    if "E1" in config.experiments:
        cases += [Case("E1", arm, config.equal_hops, bridge_ratio=r,
                       precision=precision) for arm, precision in arms]
    if "E2" in config.experiments:
        cases += [Case("E2", arm, hops, bridge_ratio=r,
                       precision=precision) for hops in config.hops for arm, precision in arms]
    if "E3" in config.experiments:
        cases += [Case("E3", "latent", config.equal_hops, control=control, bridge_ratio=r)
                  for control in ("correct", "mismatched", "zero", "random", "no_context")]
    if "E4" in config.experiments:
        cases += [Case("E4", "latent", config.equal_hops, bridge_ratio=ratio)
                  for ratio in config.bridge_ratios]
    if "E5" in config.experiments:
        cases += [Case("E5", "latent", config.equal_hops, bridge_ratio=r, precision=precision)
                  for precision in config.precisions]
    if "E6" in config.experiments and ("update" in example.get("category", "").lower()
                                      or "contradiction" in example.get("category", "").lower()
                                      or "update" in example.get("dataset", "").lower()):
        policies = ("all", "current") if has_version_filter(example) else ("all",)
        cases += [Case("E6", arm, config.equal_hops, bridge_ratio=r, update_policy=policy,
                       precision=precision) for arm, precision in arms for policy in policies]
    return [dataclasses.replace(case, repeat=repeat)
            for repeat in range(config.repeats) for case in cases]


def has_version_filter(example: dict) -> bool:
    return any(any(key in document for key in
                   ("active", "is_current", "superseded_by", "superseded"))
               for document in example.get("evidence", []))


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(x) for x in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def case_id(example_id: str, case: Case) -> str:
    return hashlib.sha256(json.dumps([example_id, dataclasses.asdict(case)],
                                    sort_keys=True).encode()).hexdigest()[:24]


def _sync(backend: Any) -> None:
    backend.synchronize()


def _block_stats(block: Any) -> dict:
    if block is None:
        return {"seq_len": 0, "nbytes": 0}
    result = {"seq_len": int(block.seq_len), "nbytes": int(block.nbytes)}
    for key in ("quant", "quantization", "metadata", "precision"):
        value = getattr(block, key, None)
        if value is not None:
            result[key] = _jsonable(value)
    return result


def _bridge_count(block: Any) -> int:
    metadata = getattr(block, "metadata", {})
    if isinstance(metadata, dict):
        return int(metadata.get("bridge_tokens_recomputed",
                                metadata.get("recomputed_tokens", metadata.get("bridge_recomputed_tokens", 0))))
    return 0


@dataclass
class PreparedExample:
    tokens: dict[str, list[int]]
    blocks: dict[str, Any]
    ingest_ms: float
    ingest_tokens: int
    prologue_tokens: list[int]
    prologue_block: Any
    truncated_ids: list[str] = field(default_factory=list)


def prepare_example(backend: Any, example: dict, config: ExperimentConfig) -> PreparedExample:
    tokens, blocks, truncated = {}, {}, []
    # Per-document caps precede retrieval. The total context cap applies after
    # ranking, so source order cannot silently remove high-ranked documents.
    for doc in example["evidence"]:
        raw = backend.encode(document_text(doc))
        capped = raw[:config.max_document_tokens]
        if len(capped) < len(raw):
            truncated.append(doc["id"])
        tokens[doc["id"]] = capped
    prologue = backend.encode(prologue_text(example))
    provisional = PreparedExample(tokens, {}, 0, 0, prologue, None, truncated)
    wanted = set()
    for case in cases_for(config, example):
        _, selected = _select_schedule(example, provisional, case, config)
        wanted.update(d["id"] for d in selected)
    _sync(backend)
    start = time.perf_counter()
    prologue_block = backend.prefill(prologue)
    for did, ids in tokens.items():
        if ids and did in wanted:
            blocks[did] = backend.prefill(ids)
    _sync(backend)
    ingest_ms = (time.perf_counter() - start) * 1000
    return PreparedExample(tokens, blocks, ingest_ms,
                           len(prologue) + sum(len(tokens[did]) for did in wanted),
                           prologue, prologue_block, truncated)


def _decode(backend: Any, prefix: Any, prompt: list[int], limit: int,
            counts: Counts, internal: bool = False) -> tuple[str, list[int]]:
    counts.query_prefill_tokens += len(prompt)
    counts.max_prefix_tokens = max(counts.max_prefix_tokens,
                                  (prefix.seq_len if prefix is not None else 0) + len(prompt))
    output = backend.decode(prefix, prompt, max_tokens=limit)
    if internal:
        counts.internal_decoded_tokens += len(output)
    else:
        counts.host_output_tokens += len(output)
    return backend.decode_tokens(output), output


def _concat(backend: Any, blocks: list[Any], ratio: float, counts: Counts) -> Any:
    merged = backend.concat(blocks, bridge_ratio=ratio)
    counts.bridge_recomputed_tokens += _bridge_count(merged)
    return merged


def _select_schedule(example: dict, prepared: PreparedExample, case: Case,
                     config: ExperimentConfig) -> tuple[list[list[dict]], list[dict]]:
    schedule = schedule_evidence(example, case.hops, config, case.update_policy)
    # Deduplicate documents globally so nominal hops never manufacture new evidence.
    seen = set()
    filtered, selected = [], []
    used_tokens = 0
    for group in schedule:
        valid = []
        for doc in group:
            did = doc["id"]
            if did not in seen and prepared.tokens.get(did):
                seen.add(did)
                if used_tokens + len(prepared.tokens[did]) > config.max_context_tokens:
                    continue
                used_tokens += len(prepared.tokens[did])
                valid.append(doc)
                selected.append(doc)
        filtered.append(valid)
    return filtered, selected


def run_case(backend: Any, example: dict, donor: dict, prepared: PreparedExample,
             case: Case, config: ExperimentConfig, seed: int) -> dict:
    selection_started = time.perf_counter()
    schedule, selected = _select_schedule(example, prepared, case, config)
    schedule_selection_ms = (time.perf_counter() - selection_started) * 1000
    if case.experiment == "E3":
        if example.get("dataset") != "synthetic_private_fact" or len(selected) != 1:
            raise ValueError("E3 currently requires single-document synthetic_private_fact examples; "
                             "multi-document donor layouts are not matched by this protocol")
        if donor.get("dataset") != "synthetic_private_fact" or len(donor["evidence"]) != 1:
            raise ValueError("E3 donor must be a single-document synthetic_private_fact example")
    counts = Counts(evidence_tokens=sum(len(prepared.tokens[d["id"]]) for d in selected))
    scheduled_evidence_tokens = counts.evidence_tokens
    delivered = [] if case.control == "no_context" else selected
    if case.control == "no_context":
        counts.evidence_tokens = 0
    suffix = backend.encode(suffix_text(example["question"], _is_rubric_task(example)))
    trace, cache_stats = [], {}
    donor_block = None
    donor_preparation_ms = 0.0
    if case.control == "mismatched":
        donor_documents = donor["evidence"]
        donor_ids = backend.encode("".join(document_text(d) for d in donor_documents))
        target = counts.evidence_tokens
        pad = backend.encode(" ")[0]
        donor_ids = (donor_ids + [pad] * max(0, target - len(donor_ids)))[:target]
        if not donor_ids:
            raise ValueError("Mismatched audit requires nonempty evidence")
        _sync(backend)
        start = time.perf_counter()
        donor_block = backend.prefill(donor_ids)
        _sync(backend)
        donor_preparation_ms = (time.perf_counter() - start) * 1000
    _sync(backend)
    started = time.perf_counter()
    if case.arm == "direct_text":
        prompt = prepared.prologue_tokens + [t for d in selected for t in prepared.tokens[d["id"]]] + suffix
        raw, output = _decode(backend, None, prompt, config.max_answer_tokens, counts)
    elif case.arm == "text":
        note = ""
        for index, group in enumerate(schedule):
            if not group:
                trace.append({"hop": index + 1, "evidence_ids": [], "skipped": "no_new_evidence"})
                continue
            evidence = "".join(backend.decode_tokens(prepared.tokens[d["id"]]) for d in group)
            user = (f"Question: {example['question']}\nPrevious evidence JSON:\n{note or '{}'}\n"
                    f"New documents:\n{evidence}\nReturn only a compact JSON object "
                    'with the shape {"facts":["subject | relation | object"]}. '
                    "Prioritize newly relevant facts and merge useful earlier facts. Preserve exact "
                    "names, codes, relationships, dates and explicit updates. Omit unrelated facts, "
                    "unsupported conclusions, headings, Markdown and the question itself. "
                    "Use brief fact strings and finish the JSON within "
                    f"{config.max_relay_tokens} tokens. Do not guess missing links or answer the question.")
            prompt = backend.chat_prompt(user, system=("Maintain compact factual evidence for another agent. "
                                                       "Return valid JSON only. Never invent facts or links."))
            note, ids = _decode(backend, None, prompt, config.max_relay_tokens, counts, internal=True)
            note = clean_prediction(note)
            trace.append({"hop": index + 1, "evidence_ids": [d["id"] for d in group],
                          "relay_text": note, "decoded_tokens": len(ids)})
        prompt = prepared.prologue_tokens + backend.encode(note) + suffix
        raw, output = _decode(backend, None, prompt, config.max_answer_tokens, counts)
    else:
        counts.ingest_prefill_tokens = prepared.ingest_tokens
        if case.experiment in {"E1", "E2", "E6"}:
            relay = prepared.prologue_block
            for index, group in enumerate(schedule):
                if not group:
                    trace.append({"hop": index + 1, "evidence_ids": [], "skipped": "no_new_evidence"})
                    continue
                relay = _concat(backend, [relay] + [prepared.blocks[d["id"]] for d in group],
                                case.bridge_ratio, counts)
                instruction = backend.encode("\nRetain the evidence above for the question: " +
                                             example["question"] + "\n")
                relay = backend.append(relay, instruction)
                counts.query_prefill_tokens += len(instruction)
                counts.max_prefix_tokens = max(counts.max_prefix_tokens, relay.seq_len)
                trace.append({"hop": index + 1, "evidence_ids": [d["id"] for d in group],
                              "relay_tokens": relay.seq_len, "decoded_tokens": 0})
                if case.precision != "bf16":
                    relay = backend.quantize(relay, bits=int(case.precision.removeprefix("int")))
        elif case.control == "no_context":
            relay = prepared.prologue_block
        elif case.experiment == "E3":
            evidence_blocks = [prepared.blocks[d["id"]] for d in selected]
            if not evidence_blocks:
                raise ValueError("Cache experiment has no selected evidence")
            evidence_relay = _concat(backend, evidence_blocks, case.bridge_ratio, counts)
            if case.control == "mismatched":
                evidence_relay = donor_block
            elif case.control in {"zero", "random"}:
                evidence_relay = backend.perturb(evidence_relay, case.control, seed=seed)
            relay = _concat(backend, [prepared.prologue_block, evidence_relay],
                            0.0, counts)
        else:
            relay = _concat(backend, [prepared.prologue_block] +
                            [prepared.blocks[d["id"]] for d in selected],
                            case.bridge_ratio, counts)
        if case.precision != "bf16" and case.experiment not in {"E1", "E2", "E6"}:
            relay = backend.quantize(relay, bits=int(case.precision.removeprefix("int")))
        cache_stats = _block_stats(relay)
        raw, output = _decode(backend, relay, suffix, config.max_answer_tokens, counts)
    _sync(backend)
    latency_ms = (time.perf_counter() - started) * 1000
    prediction = clean_prediction(raw)
    support = set(example.get("supporting_context_ids", []))
    chosen = {d["id"] for d in delivered}
    return {
        "status": "ok", "id": case_id(example["id"], case), "example_id": example["id"],
        "dataset": example.get("dataset", Path(config.dataset).stem),
        "category": example.get("category", "unspecified"), "case": dataclasses.asdict(case),
        "example_metadata": example.get("metadata", {}),
        "seed": seed, "question": example["question"], "answers": example["answers"],
        "prediction": prediction, "raw_output": raw, "output_token_ids": output,
        "generation_stop": getattr(backend, "last_decode_stats", {}).get("stop_reason"),
        "metric_role": "lexical_overlap_only_not_behavior_accuracy" if _is_rubric_task(example) else "qa_accuracy",
        "scores": score_example(prediction, example),
        "latency_ms": latency_ms,
        "cold_ingest_ms": prepared.ingest_ms if case.arm == "latent" else 0.0,
        "cache_precompute_scope": "prologue_and_union_of_scheduled_documents_for_all_cases_of_this_example",
        "retrieval_ms": (example.get("metadata", {}).get("retrieval_seconds", 0.0) * 1000
                         if "retrieval_seconds" in example.get("metadata", {}) else None),
        "schedule_selection_ms": schedule_selection_ms,
        "evidence_schedule_method": ("BM25_query_only_source_order_ties_zero_score_tail"
                                     if config.schedule == "lexical" else config.schedule),
        "donor_preparation_ms": donor_preparation_ms,
        "donor_id": donor["id"] if case.control == "mismatched" else None,
        "counts": dataclasses.asdict(counts), "cache": cache_stats,
        "evidence_ids": [d["id"] for d in delivered],
        "scheduled_evidence_ids": [d["id"] for d in selected],
        "scheduled_evidence_tokens": scheduled_evidence_tokens,
        "evidence_hops_executed": 0 if case.control == "no_context" else sum(bool(group) for group in schedule),
        "update_evaluation_mode": ("explicit_supersession_filter" if has_version_filter(example)
                                   else "category_answering_only") if case.experiment == "E6" else None,
        "truncated_document_ids": prepared.truncated_ids,
        "context_budget_tokens": config.max_context_tokens,
        "support_recall": len(support & chosen) / len(support) if support else None,
        "trace": trace,
        "memory": _jsonable(backend.memory_stats()) if hasattr(backend, "memory_stats") else {},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def summarize(records: list[dict], seed: int = 42) -> dict:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    failures = [r for r in records if r.get("status") != "ok"]
    for record in records:
        if record.get("status") == "ok":
            case = {k: v for k, v in record["case"].items() if k != "repeat"}
            key = json.dumps([record["dataset"], case], sort_keys=True)
            grouped[key].append(record)
    groups = []
    for key, rows in sorted(grouped.items()):
        dataset, case = json.loads(key)
        # Repetitions inform timing only; never inflate the accuracy sample size.
        by_example: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            by_example[row["example_id"]].append(row)
        primary = [min(group, key=lambda r: r["case"]["repeat"]) for group in by_example.values()]
        em = [r["scores"]["exact_match"] for r in primary]
        f1 = [r["scores"]["f1"] for r in primary]
        timing = [r["latency_ms"] for r in rows]
        groups.append({
            "dataset": dataset, "case": case, "n_examples": len(primary), "n_runs": len(rows),
            "metric_role": primary[0].get("metric_role", "qa_accuracy"),
            "exact_match": statistics.mean(em), "exact_match_ci95": wilson_interval(sum(em), len(em)),
            "f1": statistics.mean(f1), "f1_bootstrap_ci95": bootstrap_interval(f1, seed),
            "warm_latency_p50_ms": percentile(timing, 0.5),
            "warm_latency_p95_ms": percentile(timing, 0.95),
            "warm_latency_mean_ms": statistics.mean(timing),
            "cold_ingest_p50_ms": percentile([r["cold_ingest_ms"] for r in rows], 0.5),
            "mean_counts": {name: statistics.mean(r["counts"][name] for r in rows)
                            for name in dataclasses.asdict(Counts())},
            "mean_evidence_hops_executed": statistics.mean(r["evidence_hops_executed"] for r in rows),
            "fraction_examples_truncated": statistics.mean(bool(r["truncated_document_ids"]) for r in primary),
        })
    # Paired differences use only common examples, with bootstrap over examples.
    paired = []
    index = {}
    for record in records:
        if record.get("status") == "ok" and record["case"]["repeat"] == 0:
            c = record["case"]
            comparison_precision = "mixed_arm_precision" if c["experiment"] in {"E1", "E2", "E6"} else c["precision"]
            key = (record["dataset"], c["experiment"], c["hops"], c["bridge_ratio"],
                   comparison_precision, c["update_policy"], record["example_id"])
            index.setdefault(key, {})[(c["arm"], c["control"], c["precision"])] = record
    comparisons: dict[tuple, list[float]] = collections.defaultdict(list)
    for key, arms in index.items():
        for correct_key, correct in arms.items():
            if correct_key[:2] != ("latent", "correct"):
                continue
            for arm_key, comparator in arms.items():
                if arm_key == correct_key:
                    continue
                comp_key = key[:-1] + ("_".join(correct_key) + "_minus_" + "_".join(arm_key),)
                comparisons[comp_key].append(correct["scores"]["exact_match"] - comparator["scores"]["exact_match"])
    for key, differences in sorted(comparisons.items()):
        paired.append({"dataset": key[0], "experiment": key[1], "hops": key[2],
                       "bridge_ratio": key[3], "precision": key[4], "update_policy": key[5],
                       "comparison": key[6], "n_paired_examples": len(differences),
                       "exact_match_difference": statistics.mean(differences),
                       "paired_bootstrap_ci95": bootstrap_interval(differences, seed)})
    return {"protocol_version": PROTOCOL_VERSION, "n_successful_runs": sum(len(x) for x in grouped.values()),
            "n_failed_runs": len(failures), "groups": groups, "paired_comparisons": paired,
            "limitations": [
                "One unadapted Qwen3 model; no trained role adapters, verifier, or latent retriever.",
                "Fixed evidence schedule; nominal hops with no new evidence are skipped and disclosed.",
                "E1/E2 compare a decoded evidence-note chain to causal KV appends, not trained role-equivalent agents.",
                "Warm latency excludes cold document prefill and model loading; cold prefill is reported separately.",
                "E3 is answer-level private-evidence sensitivity, not a reproduction of an external causal audit.",
                "Small diagnostic subsets do not establish BEAM 1M/10M performance or the original thesis.",
                "Percentile estimates from small samples and bootstrap intervals at empirical ceilings are unstable.",
            ]}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result, valid_bytes = [], 0
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            # A killed process may leave one partial trailing record; remove only
            # that uncommitted suffix before resuming append operations.
            with path.open("r+b") as stream:
                stream.truncate(valid_bytes)
            break
        valid_bytes += len(line)
    return result


def machine_metadata() -> dict:
    packages = {}
    for name in ("mlx", "mlx-lm", "numpy", "transformers", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    repository = Path(__file__).resolve().parents[1]
    code_hashes = {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for relative in ("engine/backends_mlx.py", "eval/real_experiments.py", "eval/retrieval.py",
                                    "scripts/run_experiments.py")
                   if (path := repository / relative).exists()}
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": sys.version,
            "packages": packages, "git_head": git.stdout.strip() if git.returncode == 0 else None,
            "git_dirty": dirty.stdout.splitlines() if dirty.returncode == 0 else None,
            "pid": os.getpid(), "code_sha256": code_hashes}


def run_suite(backend: Any, config: ExperimentConfig) -> dict:
    config.validate()
    backend_name = str(getattr(backend, "name", ""))
    if "mock" in backend_name.lower() or not hasattr(backend, "synchronize"):
        raise ValueError("Real experiments require the native MLX backend, never the mock")
    examples = read_dataset(config.dataset, config.limit)
    if "E3" in config.experiments and len(examples) < 2:
        raise ValueError("E3 requires at least two examples for mismatched caches")
    if "E3" in config.experiments and any(example.get("dataset") != "synthetic_private_fact"
                                          or len(example["evidence"]) != 1 for example in examples):
        raise ValueError("E3 currently supports only single-document synthetic_private_fact examples")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"
    digest = hashlib.sha256(Path(config.dataset).read_bytes()).hexdigest()
    config_identity = dataclasses.asdict(config)
    for key in ("resume", "max_runtime_seconds", "output_dir"):
        config_identity.pop(key)
    metadata = backend.metadata() if callable(getattr(backend, "metadata", None)) else getattr(backend, "metadata", {})
    machine = machine_metadata()
    identity = _jsonable({"config": config_identity, "dataset_sha256": digest,
                          "protocol_version": PROTOCOL_VERSION, "backend": metadata,
                          "source_code_sha256": machine["code_sha256"]})
    if manifest_path.exists():
        if not config.resume:
            raise FileExistsError(f"Existing run directory: {output_dir}")
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Resume config/dataset differs; choose a new output directory")
    else:
        _atomic_json(manifest_path, {"identity": identity, "backend": _jsonable(metadata),
                     "machine": machine, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "schedule_note": "oracle (diagnostic)" if config.schedule == "oracle" else "no support labels used for ranking"})
    # Cost receipts are loaded only after raw dataset/config identity validation.
    _attach_timing_sidecars(examples, Path(config.dataset))
    records = load_records(results_path)
    completed = {row["id"] for row in records if row.get("status") == "ok"}
    suite_started = time.monotonic()
    planned = sum(len(cases_for(config, example)) for example in examples)
    stop_reason = "complete"
    try:
        for index, example in enumerate(examples):
            cases = [case for case in cases_for(config, example) if case_id(example["id"], case) not in completed]
            if not cases:
                continue
            if time.monotonic() - suite_started >= config.max_runtime_seconds:
                stop_reason = "runtime_budget"
                break
            # Deterministic interleaving reduces systematic arm-order warming effects.
            random.Random(config.seed + index).shuffle(cases)
            prepared = prepare_example(backend, example, config)
            donor = examples[(index + 1) % len(examples)]
            if "E3" in config.experiments:
                # A mismatched donor must have a different reference answer.
                options = [d for d in examples if not ({normalize_answer(a) for a in d["answers"]} &
                                                       {normalize_answer(a) for a in example["answers"]})]
                if not options:
                    raise ValueError("No donor with disjoint answers for E3")
                donor = options[index % len(options)]
            for case in cases:
                if time.monotonic() - suite_started >= config.max_runtime_seconds:
                    stop_reason = "runtime_budget"
                    break
                seed = config.seed + index * 1009 + case.repeat
                try:
                    row = run_case(backend, example, donor, prepared, case, config, seed)
                except Exception as exc:
                    row = {"status": "error", "id": case_id(example["id"], case),
                           "example_id": example["id"], "case": dataclasses.asdict(case),
                           "error_type": type(exc).__name__, "error": str(exc)}
                    with results_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    records.append(row)
                    raise
                with results_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(row)
                completed.add(row["id"])
                print(json.dumps({"completed": len(completed), "planned": planned,
                                  "example": example["id"], "case": dataclasses.asdict(case),
                                  "em": row["scores"]["exact_match"], "latency_ms": round(row["latency_ms"]),
                                  "prediction": row["prediction"]}), flush=True)
                _atomic_json(output_dir / "checkpoint.json", {"completed": len(completed), "planned": planned,
                             "last_id": row["id"], "elapsed_seconds": time.monotonic() - suite_started})
            del prepared
            if hasattr(backend, "clear_cache"):
                backend.clear_cache()
            if stop_reason != "complete":
                break
    except BaseException:
        stop_reason = "failed_or_interrupted"
        raise
    finally:
        summary = summarize(records, config.seed)
        summary.update({"stop_reason": stop_reason, "planned_runs": planned,
                        "completed_unique_runs": len(completed),
                        "elapsed_seconds_this_invocation": time.monotonic() - suite_started})
        _atomic_json(output_dir / "summary.json", summary)
    return summary
