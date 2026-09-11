#!/usr/bin/env python3
"""Prepare bounded, checksummed, public/synthetic experiment datasets.

No API keys, model inference, remote code execution, pickle, or personal data.
Network fetches are HTTPS-only, byte-limited, and atomically saved under data/.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.datasets import (DEFAULT_SEED, flatten_beam_chat, generate_synthetic,
                           normalize_hotpot, normalize_musique, sha256_file,
                           stable_sample, write_jsonl)

BEAM_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"
MUSIQUE_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
HOTPOT_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
MUSIQUE_URL = "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"


class Downloader:
    def __init__(self, root: Path, max_total_mb: int):
        self.root = root
        self.remaining = max_total_mb * 1024 * 1024
        self.receipts = []

    def fetch(self, url: str, relative: str, max_mb: int = 100) -> Path:
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS downloads are allowed")
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            receipt = {"url": url, "path": str(path), "cached": True,
                       "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if receipt["bytes"] > max_mb * 1024 * 1024:
                raise ValueError(f"Cached file exceeds per-file limit: {path}")
            self.receipts.append(receipt)
            return path
        limit = min(max_mb * 1024 * 1024, self.remaining)
        if limit <= 0:
            raise ValueError("Download budget exhausted")
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "latent-memory-data/1.0"})
        size = 0
        try:
            for attempt in range(4):
                try:
                    response = urllib.request.urlopen(request, timeout=45)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            with response, temporary.open("wb") as output:
                length = response.headers.get("Content-Length")
                if length and int(length) > limit:
                    raise ValueError(f"Download exceeds budget: {length} > {limit}")
                while chunk := response.read(min(1024 * 1024, limit - size + 1)):
                    size += len(chunk)
                    self.remaining -= len(chunk)
                    if size > limit:
                        raise ValueError(f"Download exceeded {limit} bytes: {url}")
                    output.write(chunk)
                final_url = response.url
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self.receipts.append({"url": url, "resolved_url": final_url, "path": str(path),
                              "bytes": size, "sha256": sha256_file(path), "cached": False})
        print(f"Downloaded {path}: {size / 1024 / 1024:.1f} MiB", flush=True)
        return path


def prepare_synthetic(root, count, seed):
    receipts = []
    for kind in ("private_fact", "multihop", "updates"):
        receipts.append(write_jsonl(root / f"synthetic_{kind}.jsonl", generate_synthetic(kind, count, seed)))
        receipts.append(write_jsonl(root / f"synthetic_{kind}_dev.jsonl", generate_synthetic(kind, min(32, count), seed + 1, "dev")))
    return receipts


def prepare_hotpot(root, downloader, count, seed):
    # The canonical CMU origin can be unavailable. Its cached JSON remains
    # supported. Otherwise use the publisher's immutable Parquet snapshot.
    raw_path = root / "raw/hotpot_dev_distractor_v1.json"
    revision = "1908d6afbbead072334abe2965f91bd2709910ab"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        source_url = HOTPOT_URL
    else:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("HotpotQA's pinned Parquet source needs pyarrow. Use a Python environment with pyarrow, or put official JSON at data/raw/hotpot_dev_distractor_v1.json.") from exc
        source_url = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{revision}/distractor/validation-00000-of-00001.parquet"
        raw_path = downloader.fetch(source_url, "raw/hotpot_dev_distractor.parquet", 60)
        raw = parquet.read_table(raw_path).to_pylist()
        for row in raw:
            row["_id"] = row.pop("id")
    normalized = [normalize_hotpot(row) for row in stable_sample(raw, count, seed, "_id")]
    digest = sha256_file(raw_path)
    for row in normalized:
        row["metadata"].update(download_source=source_url,
                               source_revision=revision if source_url != HOTPOT_URL else None,
                               sample_method="SHA-256 rank of seed and source question ID",
                               source_population=len(raw), sample_seed=seed,
                               source_sha256=digest)
    return [write_jsonl(root / "hotpotqa_dev_sample.jsonl", normalized)]


def prepare_musique(root, downloader, count, seed):
    archive = downloader.fetch(MUSIQUE_URL, "raw/musique.zip", 600)
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.endswith("musique_ans_v1.0_dev.jsonl") or name.endswith("musique_ans_dev.jsonl")]
        if len(names) != 1:
            raise ValueError(f"Expected one answerable MuSiQue dev file; found {names}")
        info = zipped.getinfo(names[0])
        if info.file_size > 120 * 1024 * 1024:
            raise ValueError("MuSiQue dev file exceeds uncompressed size limit")
        with zipped.open(names[0]) as handle:
            raw = [json.loads(line) for line in handle if line.strip()]
    normalized = [normalize_musique(row) for row in stable_sample(raw, count, seed)]
    return [write_jsonl(root / "musique_dev_sample.jsonl", normalized)]


def prepare_beam(root, downloader, conversations):
    """Store full 1M histories once; probes reference corpora, without oracle cuts."""
    receipts = []
    for number in conversations:
        base = f"https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/{BEAM_REVISION}/chats/1M/{number}"
        chat_path = downloader.fetch(f"{base}/chat.json", f"raw/beam_1m_{number}_chat.json", 15)
        probe_path = downloader.fetch(f"{base}/probing_questions/probing_questions.json", f"raw/beam_1m_{number}_probes.json", 1)
        conversation_id = f"beam-1m-{number}"
        corpus = flatten_beam_chat(json.loads(chat_path.read_text()), conversation_id)
        corpus_path = root / f"beam_1m_{number}_corpus.jsonl"
        receipts.append(write_jsonl(corpus_path, corpus))
        probes = []
        for category, questions in json.loads(probe_path.read_text()).items():
            for index, question in enumerate(questions):
                answer = next((question[key] for key in ("answer", "ideal_answer", "ideal_response", "expected_answer", "expected_compliance", "ideal_summary") if question.get(key)), None)
                if not answer:
                    raise ValueError(f"No reference answer in {category}: {list(question)}")
                probes.append({"schema_version": 1, "id": f"{conversation_id}:{category}:{index}",
                               "dataset": "beam", "split": "1M", "category": category,
                               "question": question["question"], "answers": [answer], "evidence": [],
                               "corpus_path": corpus_path.name,
                               "metadata": {"conversation_id": conversation_id, "source_revision": BEAM_REVISION,
                                            "source": base, "license": "CC-BY-SA-4.0",
                                            "protocol": "public_conversation_subsample",
                                            "requires_rubric_judge": True,
                                            "rubric": question.get("rubric", []),
                                            "source_annotation": question,
                                            "corpus_messages": len(corpus)}})
        receipts.append(write_jsonl(root / f"beam_1m_{number}_probes.jsonl", probes))
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["synthetic", "hotpotqa", "musique", "beam"], default=["synthetic"])
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--beam-conversations", nargs="+", type=int, default=[1])
    parser.add_argument("--max-download-mb", type=int, default=800)
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.max_download_mb <= 1024:
        parser.error("count must be positive; max-download-mb must be in 1..1024")
    if any(number < 1 or number > 35 for number in args.beam_conversations):
        parser.error("BEAM 1M conversation IDs must be in 1..35")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloader = Downloader(args.output_dir, args.max_download_mb)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "schema_version": 1, "count_requested": args.count, "sources": [], "outputs": [], "errors": {}}
    for dataset in args.datasets:
        try:
            if dataset == "synthetic":
                rows = prepare_synthetic(args.output_dir, args.count, args.seed)
            elif dataset == "hotpotqa":
                rows = prepare_hotpot(args.output_dir, downloader, args.count, args.seed)
            elif dataset == "musique":
                rows = prepare_musique(args.output_dir, downloader, args.count, args.seed)
            else:
                rows = prepare_beam(args.output_dir, downloader, args.beam_conversations)
            manifest["outputs"].extend(rows)
            print(f"Prepared {dataset}: {len(rows)} outputs", flush=True)
        except Exception as exc:
            manifest["errors"][dataset] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED {dataset}: {exc}", flush=True)
    manifest["sources"] = downloader.receipts
    manifest_path = args.output_dir / ("manifest-" + "-".join(args.datasets) + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)
    return bool(manifest["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
