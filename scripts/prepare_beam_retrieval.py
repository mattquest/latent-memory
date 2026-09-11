#!/usr/bin/env python3
"""Retrieve from complete BEAM histories with BM25 and Qwen token chunking.

No gold/reference/rubric fields are read by the retriever. They are copied to
the resulting examples only for later scoring, never indexed or prompted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.datasets import sha256_file, write_jsonl
from eval.retrieval import BM25Index


def semantic_example_sha256(example):
    """Hash the immutable normalized row, never measurement timestamps."""
    return hashlib.sha256(json.dumps(example, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare_conversation(number, data_dir, tokenizer, *, chunk_tokens=256,
                         overlap_tokens=32, top_k=20, clock=time.perf_counter):
    """Write semantic inputs and separate timing measurements for one corpus.

    Combined datasets retain this conversation's safe basename sidecar reference.
    No elapsed measurements or their volatile hashes enter semantic JSONL rows.
    """
    start = clock()
    corpus_path = data_dir / f"beam_1m_{number}_corpus.jsonl"
    probes_path = data_dir / f"beam_1m_{number}_probes.jsonl"
    corpus = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    documents, corpus_tokens = [], 0
    for entry in corpus:
        tokens = tokenizer.encode(entry["text"], add_special_tokens=False)
        corpus_tokens += len(tokens)
        for offset in range(0, len(tokens), chunk_tokens - overlap_tokens):
            piece = tokens[offset:offset + chunk_tokens]
            documents.append({
                "id": f"{entry['id']}:tokens:{offset}",
                "title": f"{entry['title']}; date {entry.get('time_anchor', 'unknown')}",
                "text": tokenizer.decode(piece),
                "source_message_id": entry.get("source_message_id", entry["id"]),
                "chronological_index": entry.get("chronological_index"),
                "role": entry.get("role"), "token_start": offset,
                "token_count": len(piece),
            })
            if offset + chunk_tokens >= len(tokens):
                break
    index = BM25Index(documents)
    preparation_seconds = clock() - start
    examples, query_timings = [], {}
    output = data_dir / f"beam_1m_{number}_retrieved.jsonl"
    timing_path = output.with_suffix(".timings.json")
    for line in probes_path.read_text().splitlines():
        probe = json.loads(line)
        query_start = clock()
        hits = index.search(probe["question"], k=top_k)
        retrieval_seconds = clock() - query_start
        evidence = [{**item, "retrieval_score": score} for item, score in hits]
        examples.append({**probe, "evidence": evidence,
                         "metadata": {**probe["metadata"],
                                      "retrieval": "BM25 fixed ranking over full conversation",
                                      "corpus_qwen_tokens": corpus_tokens,
                                      "corpus_chunks": len(documents),
                                      "corpus_sha256": sha256_file(corpus_path),
                                      "chunk_tokens": chunk_tokens,
                                      "overlap_tokens": overlap_tokens,
                                      "max_chunks_per_message": 2,
                                      "gold_used_for_retrieval": False,
                                      "timing_sidecar": timing_path.name}})
        query_timings[probe["id"]] = {
            "semantic_example_sha256": semantic_example_sha256(examples[-1]),
            "retrieval_seconds": retrieval_seconds}
    receipt = write_jsonl(output, examples)
    timing = {"schema_version": 1, "semantic_dataset": output.name,
              "semantic_dataset_sha256": receipt["sha256"],
              "index_preparation_seconds": preparation_seconds,
              "queries": query_timings}
    temporary = timing_path.with_suffix(timing_path.suffix + ".tmp")
    temporary.write_text(json.dumps(timing, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(timing_path)
    receipt.update(corpus_qwen_tokens=corpus_tokens, corpus_chunks=len(documents),
                   timing_sidecar=timing_path.name, timing_sidecar_sha256=sha256_file(timing_path))
    output.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--conversations", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.overlap_tokens < args.chunk_tokens or not 1 <= args.top_k <= 32:
        parser.error("invalid chunk/overlap/top-k configuration")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 2**31 - 1
    for number in args.conversations:
        receipt = prepare_conversation(number, args.data_dir, tokenizer,
                                       chunk_tokens=args.chunk_tokens,
                                       overlap_tokens=args.overlap_tokens, top_k=args.top_k)
        print(json.dumps(receipt), flush=True)
    # A fixed balanced sample for the more expensive hop sweep: one hash-ranked
    # probe per category per conversation. Categories are used only to sample
    # the evaluation set; neither rubrics nor answers affect retrieval.
    import collections
    import hashlib
    combined, strata = [], collections.defaultdict(list)
    for number in args.conversations:
        path = args.data_dir / f"beam_1m_{number}_retrieved.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            combined.append(row)
            strata[(number, row["category"])].append(row)
    sample = [min(rows, key=lambda row: hashlib.sha256(
        ("20260911:" + row["id"]).encode()).hexdigest())
        for _, rows in sorted(strata.items())]
    for name, rows in (("beam_1m_retrieved.jsonl", combined),
                       ("beam_1m_hop_sample.jsonl", sample)):
        receipt = write_jsonl(args.data_dir / name, rows)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
