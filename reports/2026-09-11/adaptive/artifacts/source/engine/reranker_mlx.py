"""Bounded official Qwen3-Reranker-0.6B scoring, without text generation.

Prompt and two-logit scoring follow the official Transformers example:
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/e61197ed45024b0ed8a2d74b80b4d909f1255473/README.md
The default 2048-token cap is an explicit local experiment bound, below the
official example's 8192. Documents are scored separately; no padding or shared
query KV cache is used. All per-query tokenization/forward work belongs in cold
query latency. Loading and checksum verification belong in reported setup cost.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
# Received from this exact official revision; bind the receipt to actual files.
CHECKPOINT_SHA256 = {
    "README.md": "5bba8c734f6dd3ae48317b4139317e45a7fce48fc55e15670b23a0dd15492ab6",
    "chat_template.jinja": "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d",
    "config.json": "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0",
    "generation_config.json": "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based '
          'on the Query and the Instruct provided. Note that the answer can only be '
          '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def probability_yes(no_logit, yes_logit):
    """Two-class softmax in host float precision, stable for extreme logits."""
    no_logit, yes_logit = float(no_logit), float(yes_logit)
    if not math.isfinite(no_logit) or not math.isfinite(yes_logit):
        raise ValueError("Reranker returned a nonfinite yes/no logit")
    difference = yes_logit - no_logit
    if difference >= 0:
        return 1 / (1 + math.exp(-difference))
    value = math.exp(difference)
    return value / (1 + value)


def prompt_tokens(encode, question, document, max_context):
    """Official prefix/body/suffix tokenization; right-truncate only the body.

The query must fit intact before any document truncation. Corpus IDs and all
question gold metadata are excluded; the document input is title plus text.
"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Reranking requires a nonempty question")
    if not isinstance(document, dict) or not isinstance(document.get("text"), str):
        raise ValueError("Reranking requires document mappings with text")
    title = document.get("title", "")
    if not isinstance(title, str):
        raise ValueError("Document title must be a string")
    prefix, suffix = list(encode(PREFIX)), list(encode(SUFFIX))
    header = f"<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: "
    available = max_context - len(prefix) - len(suffix)
    if len(encode(header)) >= available:
        raise ValueError("Reranker context leaves no space after the complete query")
    text = (title + "\n" if title else "") + document["text"]
    body = list(encode(header + text))
    result = prefix + body[:available] + suffix
    return result, {"input_tokens": len(result),
                    "untruncated_input_tokens": len(prefix) + len(body) + len(suffix),
                    "truncated": len(body) > available}


def _verify_local_checkpoint(path, revision):
    """Require pinned official provenance and verify files before model loading."""
    manifest_path = path / "download-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pinned download manifest required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo_id") != MODEL_ID or manifest.get("revision") != revision:
        raise ValueError("Reranker checkpoint provenance differs from requested official revision")
    entries = manifest.get("files", [])
    if len(entries) != len(CHECKPOINT_SHA256) or {entry["path"] for entry in entries} != set(CHECKPOINT_SHA256):
        raise ValueError("Reranker manifest must describe the exact pinned checkpoint file set")
    if {file.name for file in path.glob("model*.safetensors")} != {"model.safetensors"}:
        raise ValueError("Unexpected model shards would alter the pinned reranker weights")
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Reranker manifest paths must remain within its model directory")
        if entry["sha256"] != CHECKPOINT_SHA256[entry["path"]]:
            raise ValueError(f"Reranker manifest checksum differs from pinned official revision: {relative}")
        target = path / relative
        if target.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Reranker file size mismatch: {relative}")
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Reranker file checksum mismatch: {relative}")
    config = json.loads((path / "config.json").read_text())
    expected = {"model_type": "qwen3", "hidden_size": 1024,
                "num_hidden_layers": 28, "tie_word_embeddings": True,
                "vocab_size": 151669}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint architecture differs from official Qwen3-Reranker-0.6B")
    return manifest


class MLXQwen3Reranker:
    """Official BF16 checkpoint with one bounded causal forward per document.

MLX's memory limit is global to the process: setting 32 GiB here covers both
this checkpoint and an already loaded generator, rather than allocating an
additional allowance. No wired-memory setting is changed. Injected backends
permit CPU-only prompt/scoring tests without importing MLX or loading weights.
"""

    def __init__(self, model_path="models/Qwen3-Reranker-0.6B", *,
                 revision=REVISION, max_context=2048, memory_limit_gb=32,
                 max_documents=20, backend=None):
        if not isinstance(max_context, int) or not 0 < max_context <= 8192:
            raise ValueError("Reranker context must be an integer in (0, 8192]")
        if not isinstance(max_documents, int) or not 0 < max_documents <= 20:
            raise ValueError("Reranker document cap must be an integer in (0, 20]")
        if not 0 < memory_limit_gb <= 32:
            raise ValueError("Combined process memory limit must be in (0, 32] GiB")
        if revision != REVISION:
            raise ValueError("This experiment supports only its pinned official reranker revision")
        self.manifest = None
        if backend is None:
            local = Path(model_path).resolve()
            self.manifest = _verify_local_checkpoint(local, revision)
            from engine.backends_mlx import MLXBackend
            backend = MLXBackend(str(local), revision=revision, max_context=max_context,
                                 memory_limit_gb=memory_limit_gb)
        self.backend = backend
        self.max_context = min(max_context, backend.max_context)
        self.max_documents = max_documents
        self.revision = revision
        self.no_token_ids = list(backend.encode("no"))
        self.yes_token_ids = list(backend.encode("yes"))
        if len(self.no_token_ids) != 1 or len(self.yes_token_ids) != 1:
            raise ValueError("Official reranker requires single-token yes and no labels")
        if self.no_token_ids == self.yes_token_ids:
            raise ValueError("Reranker yes/no token IDs must differ")
        self.last_stats = {}

    def score(self, question, documents):
        if not isinstance(documents, (list, tuple)) or len(documents) > self.max_documents:
            raise ValueError(f"Score at most {self.max_documents} documents per query")
        scores, stats = [], []
        self.last_stats = {}
        for document in documents:
            tokens, document_stats = prompt_tokens(self.backend.encode, question, document,
                                                    self.max_context)
            logits = self.backend.next_logits(None, tokens)
            no = logits[self.no_token_ids[0]]
            yes = logits[self.yes_token_ids[0]]
            no = no.item() if hasattr(no, "item") else no
            yes = yes.item() if hasattr(yes, "item") else yes
            scores.append(probability_yes(no, yes))
            stats.append(document_stats)
        self.backend.synchronize()
        self.last_stats = {"input_tokens": sum(s["input_tokens"] for s in stats),
                           "documents": len(documents),
                           "truncated_documents": sum(s["truncated"] for s in stats),
                           "per_document_input_tokens": [s["input_tokens"] for s in stats],
                           "per_document_untruncated_input_tokens": [s["untruncated_input_tokens"] for s in stats],
                           "per_document_truncated": [s["truncated"] for s in stats],
                           "generated_tokens": 0}
        return scores

    def synchronize(self):
        self.backend.synchronize()

    def metadata(self):
        return {"model_id": MODEL_ID, "revision": self.revision,
                "checkpoint_manifest": self.manifest,
                "backend": self.backend.metadata(), "max_context": self.max_context,
                "max_documents": self.max_documents, "batch_size": 1,
                "scoring": "sigmoid(float(yes_logit)-float(no_logit))",
                "no_token_id": self.no_token_ids[0], "yes_token_id": self.yes_token_ids[0],
                "instruction": INSTRUCTION, "input_document": "title + newline + text",
                "truncation": "right-truncate body; preserve framing and complete question",
                "query_prefix_cache": False, "generated_tokens": 0,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
