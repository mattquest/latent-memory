"""Small, reproducible evaluation data; this module never loads model weights.

Gold answers, supporting IDs, and oracle schedules are scoring metadata. Retrieval
and model prompts must see only question plus evidence text/ordinary metadata.
See docs/data-and-protocol.md for benchmark and diagnostic boundaries.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260911


def load_examples(path: str | Path, limit: int | None = None,
                  split: str | None = None) -> list[dict[str, Any]]:
    """Read normalized JSONL and validate required fields, retaining source order."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("id", "dataset", "question", "answers", "evidence"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number}: missing {key}")
            if not isinstance(row["answers"], list) or not row["answers"]:
                raise ValueError(f"{path}:{line_number}: answers must be nonempty list")
            ids = [str(item["id"]) for item in row["evidence"]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path}:{line_number}: duplicate evidence IDs")
            if not all(isinstance(item["text"], str) for item in row["evidence"]):
                raise ValueError(f"{path}:{line_number}: evidence text must be string")
            if split is not None and row.get("split") != split:
                continue
            if limit is not None and len(examples) >= limit:
                break
            examples.append(row)
    return examples


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Atomic deterministic UTF-8 JSONL writer with a checksum receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return {"path": str(path), "examples": count, "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample(rows: list[dict[str, Any]], count: int, seed: int,
                  id_key: str = "id") -> list[dict[str, Any]]:
    """Hash-ranked sample, independent of source ordering or Python random versions."""
    def key(row):
        return hashlib.sha256(f"{seed}:{row[id_key]}".encode()).hexdigest()
    return sorted(rows, key=key)[:count]


def _base(kind: str, index: int, seed: int, split: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "id": f"{kind}-{seed}-{index:05d}",
            "dataset": f"synthetic_{kind}", "split": split, "category": kind,
            "metadata": {"seed": seed, "synthetic": True,
                         "protocol": "controlled_diagnostic",
                         "source": "locally generated fictional facts",
                         "generator_version": 1}}


def generate_synthetic(kind: str, count: int = 128, seed: int = DEFAULT_SEED,
                       split: str = "test") -> list[dict[str, Any]]:
    """Generate private lookup, 2–20-link reasoning, or version/conflict probes.

    Uses only fictional entities and random labels; no user's data is inspected.
    Each example gets an independent deterministic RNG. Dev should use another
    seed, so inspecting development examples does not expose test private facts.
    """
    if kind not in ("private_fact", "multihop", "updates"):
        raise ValueError(f"Unknown synthetic kind: {kind}")
    rows = []
    for index in range(count):
        rng = random.Random(f"latent-memory-v1:{kind}:{seed}:{index}")
        row = _base(kind, index, seed, split)
        # Fixed-width identifiers and values help make audit cache lengths similar.
        entity = f"K{rng.randrange(100000, 1000000)}"
        answer = str(rng.randrange(100000, 1000000))
        prefix = row["id"]
        if kind == "private_fact":
            evidence = [{"id": f"{prefix}:fact", "title": "Private registry",
                         "text": f"Private registry entry: the access number for record {entity} is {answer}."}]
            row.update(question=f"What is the access number for record {entity}? Answer with only the six-digit number.",
                       answers=[answer], evidence=evidence,
                       supporting_context_ids=[evidence[0]["id"]],
                       hop_context_ids=[[evidence[0]["id"]]])
            row["metadata"].update(answer_space=900000, receiver_private_content=True)
        elif kind == "multihop":
            links = (2, 3, 5, 10, 20)[index % 5]
            labels = rng.sample(range(100000, 1000000), links)
            labels = [f"D{number}" for number in labels]
            labels[0] = entity
            supports = []
            for offset in range(links - 1):
                supports.append({"id": f"{prefix}:link:{offset:02d}",
                                 "title": f"Depot {labels[offset]}",
                                 "text": f"Depot {labels[offset]} points to depot {labels[offset + 1]}."})
            supports.append({"id": f"{prefix}:link:{links-1:02d}",
                             "title": f"Depot {labels[-1]}",
                             "text": f"Depot {labels[-1]} is the final depot. Its access number is {answer}."})
            distractors = []
            for offset in range(max(4, links)):
                label = f"X{rng.randrange(100000, 1000000)}"
                false_answer = str(rng.randrange(100000, 1000000))
                distractors.append({"id": f"{prefix}:distractor:{offset:02d}",
                                    "title": f"Depot {label}",
                                    "text": f"Depot {label} is the final depot. Its access number is {false_answer}."})
            evidence = supports + distractors
            rng.shuffle(evidence)
            row.update(question=f"Starting at depot {entity}, follow the depot links to the final depot. What is its access number? Answer with only the six-digit number.",
                       answers=[answer], evidence=evidence,
                       supporting_context_ids=[item["id"] for item in supports],
                       hop_context_ids=[[item["id"]] for item in supports])
            row["metadata"].update(chain_length=links, n_distractors=len(distractors),
                                   oracle_schedule_warning="Gold chain order: diagnostic use only")
        else:
            old_answer = answer
            while answer == old_answer:
                answer = str(rng.randrange(100000, 1000000))
            old_id, new_id = f"{prefix}:v1", f"{prefix}:v2"
            if index % 4 == 3:
                category = "contradiction_resolution"
                evidence = [
                    {"id": old_id, "title": "Clerk A report", "version": 1, "active": True,
                     "text": f"Clerk A reports that the access number for record {entity} is {old_answer}. Neither clerk has higher authority."},
                    {"id": new_id, "title": "Clerk B report", "version": 1, "active": True,
                     "text": f"Clerk B reports that the access number for record {entity} is {answer}. This conflicts with Clerk A; neither report is marked as a correction."}]
                question = f"Do the equally authoritative reports agree on the access number for record {entity}? Answer only AGREEMENT or CONFLICT."
                answers = ["CONFLICT"]
                row["metadata"]["conflicting_values"] = [old_answer, answer]
            else:
                category = "knowledge_update"
                evidence = [
                    {"id": old_id, "title": "Registry revision 1", "version": 1,
                     "active": False, "superseded_by": new_id,
                     "text": f"On 2026-01-01, record {entity} had access number {old_answer}."},
                    {"id": new_id, "title": "Registry revision 2", "version": 2,
                     "active": True, "supersedes": old_id,
                     "text": f"On 2026-02-01, record {entity} was updated. Its current access number is {answer}. This replaces the earlier access number."}]
                question = f"What is the current access number for record {entity}? Answer with only the six-digit number."
                answers = [answer]
                row["metadata"].update(old_answer=old_answer, current_answer=answer,
                                       active_evidence_ids=[new_id])
            row.update(question=question, answers=answers, evidence=evidence,
                       supporting_context_ids=[item["id"] for item in evidence],
                       hop_context_ids=[[old_id], [new_id]], category=category)
            row["metadata"]["evidence_order"] = "chronological; conflicts have equal authority"
        rows.append(row)
    return rows


def normalize_hotpot(row: dict[str, Any]) -> dict[str, Any]:
    context = row["context"]
    if isinstance(context, dict):
        context = zip(context["title"], context["sentences"])
    evidence = [{"id": f"{row['_id']}:{i}", "title": title, "text": "".join(sentences),
                 "sentences": sentences}
                for i, (title, sentences) in enumerate(context)]
    supporting = row["supporting_facts"]
    if isinstance(supporting, dict):
        supporting = list(zip(supporting["title"], supporting["sent_id"]))
    supporting_titles = {fact[0] for fact in supporting}
    return {"schema_version": SCHEMA_VERSION, "id": row["_id"], "dataset": "hotpotqa",
            "split": "validation_distractor", "question": row["question"],
            "answers": [row["answer"]], "evidence": evidence,
            "supporting_context_ids": [item["id"] for item in evidence if item["title"] in supporting_titles],
            "category": row.get("type", "unknown"),
            "metadata": {"source_id": row["_id"], "level": row.get("level"),
                         "supporting_facts": list(supporting), "protocol": "public_dev_subsample",
                         "license": "CC-BY-SA-4.0", "source": "https://hotpotqa.github.io/"}}


def normalize_musique(row: dict[str, Any]) -> dict[str, Any]:
    evidence = [{"id": f"{row['id']}:{paragraph['idx']}", "title": paragraph["title"],
                 "text": paragraph["paragraph_text"]}
                for paragraph in row["paragraphs"]]
    answers = list(dict.fromkeys([row["answer"]] + row.get("answer_aliases", [])))
    return {"schema_version": SCHEMA_VERSION, "id": row["id"], "dataset": "musique",
            "split": "validation_answerable", "question": row["question"],
            "answers": answers, "evidence": evidence,
            "supporting_context_ids": [f"{row['id']}:{p['idx']}" for p in row["paragraphs"] if p["is_supporting"]],
            "category": row["id"].split("__")[0],
            "metadata": {"source_id": row["id"], "answerable": row.get("answerable", True),
                         "question_decomposition": row.get("question_decomposition", []),
                         "protocol": "public_dev_subsample", "license": "CC-BY-4.0",
                         "source": "https://github.com/StonyBrookNLP/musique"}}


def flatten_beam_chat(chat: list[Any], conversation_id: str) -> list[dict[str, Any]]:
    """Flatten official nested batches, preserving chronology and every message."""
    evidence = []
    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            if "content" in item and "role" in item:
                index = len(evidence)
                evidence.append({"id": f"{conversation_id}:{index}",
                                 "title": f"Message {item.get('id', index)} ({item['role']})",
                                 "text": item["content"], "role": item["role"],
                                 "source_message_id": item.get("id"),
                                 "time_anchor": item.get("time_anchor"), "chronological_index": index})
            elif "turns" in item:
                visit(item["turns"])
            else:
                raise ValueError(f"Unknown BEAM chat object keys: {list(item)}")
        else:
            raise ValueError(f"Unknown BEAM chat value: {type(item)}")
    visit(chat)
    return evidence
