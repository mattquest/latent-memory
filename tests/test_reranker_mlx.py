import json
import math

import pytest

from engine.reranker_mlx import (
    CHECKPOINT_SHA256, INSTRUCTION, MODEL_ID, PREFIX, REVISION, SUFFIX, MLXQwen3Reranker,
    _verify_local_checkpoint, probability_yes, prompt_tokens,
)


def encode(text):
    return list(text.encode())


class ScoringBackend:
    max_context = 2048

    def __init__(self):
        self.calls = []
        self.syncs = 0

    def encode(self, text):
        return {"no": [0], "yes": [1]}.get(text, encode(text))

    def next_logits(self, prefix, tokens):
        self.calls.append((prefix, tuple(tokens)))
        # The unrelated vocabulary logit must not affect the two-label score.
        return [2.0, 4.0 if b"RELEVANT-FIXTURE" in bytes(tokens) else -2.0, 10000.0]

    def synchronize(self):
        self.syncs += 1

    def metadata(self):
        return {"backend": "cpu-recording-fixture"}


def test_official_prompt_has_no_corpus_or_gold_metadata():
    document = {"id": "SECRET-ID", "title": "China", "text": "Beijing is its capital.",
                "answer": "SECRET-GOLD", "supporting": True}
    tokens, stats = prompt_tokens(encode, "Capital?", document, 2048)
    expected = (PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: Capital?\n<Document>: " +
                "China\nBeijing is its capital." + SUFFIX)
    assert bytes(tokens).decode() == expected
    assert "SECRET" not in expected
    assert not stats["truncated"]
    assert stats["input_tokens"] == stats["untruncated_input_tokens"] == len(tokens)


def test_document_truncation_preserves_complete_query_and_assistant_suffix():
    question = "Who is the mayor?"
    tokens, stats = prompt_tokens(encode, question, {"text": "z" * 10000}, 600)
    assert len(tokens) == 600
    decoded = bytes(tokens).decode()
    assert decoded.startswith(PREFIX)
    assert f"<Query>: {question}\n<Document>: " in decoded
    assert decoded.endswith(SUFFIX)
    assert stats["truncated"]
    assert stats["untruncated_input_tokens"] > stats["input_tokens"]
    with pytest.raises(ValueError, match="complete query"):
        prompt_tokens(encode, "q" * 1000, {"text": "z"}, 600)


def test_two_label_score_is_stable_and_not_full_vocabulary_softmax():
    assert probability_yes(3, 3) == 0.5
    assert probability_yes(-10000, 10000) == 1.0
    assert probability_yes(10000, -10000) == 0.0
    assert probability_yes(1, 2) == pytest.approx(math.exp(2) / (math.exp(1) + math.exp(2)))
    assert probability_yes(-2, 7) == pytest.approx(probability_yes(998, 1007))
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="nonfinite"):
            probability_yes(bad, 1)


def test_document_order_cold_forwards_and_actual_token_statistics():
    backend = ScoringBackend()
    reranker = MLXQwen3Reranker(backend=backend, max_context=600)
    documents = [{"title": "A", "text": "RELEVANT-FIXTURE"}, {"text": "z" * 1000}]
    scores = reranker.score("What?", documents)
    assert scores == pytest.approx([probability_yes(2, 4), probability_yes(2, -2)])
    assert all(prefix is None for prefix, _ in backend.calls)
    assert len(backend.calls) == 2 and backend.syncs == 1
    assert reranker.last_stats["input_tokens"] == sum(len(tokens) for _, tokens in backend.calls)
    assert reranker.last_stats["per_document_truncated"] == [False, True]
    assert reranker.last_stats["documents"] == 2
    assert reranker.last_stats["generated_tokens"] == 0
    assert reranker.metadata()["revision"] == REVISION
    # Calling again does full independent prefill, with fresh per-call counters.
    reranker.score("What?", documents[:1])
    assert len(backend.calls) == 3
    assert reranker.last_stats["documents"] == 1


def test_safety_bounds_and_label_validation_happen_without_model_loading():
    for kwargs in ({"max_context": 8193}, {"memory_limit_gb": 33}, {"max_documents": 21},
                   {"revision": "unverified-revision"}):
        with pytest.raises(ValueError):
            MLXQwen3Reranker(backend=ScoringBackend(), **kwargs)
    backend = ScoringBackend()
    reranker = MLXQwen3Reranker(backend=backend, max_documents=2)
    with pytest.raises(ValueError, match="at most 2"):
        reranker.score("Question", [{"text": "doc"}] * 3)
    assert not backend.calls
    backend.encode = encode
    with pytest.raises(ValueError, match="single-token"):
        MLXQwen3Reranker(backend=backend)


def test_missing_or_foreign_checkpoint_manifest_fails_before_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest required"):
        _verify_local_checkpoint(tmp_path, REVISION)
    (tmp_path / "download-manifest.json").write_text(json.dumps({
        "repo_id": "Qwen/Qwen3-0.6B", "revision": REVISION, "files": []}))
    with pytest.raises(ValueError, match="provenance"):
        _verify_local_checkpoint(tmp_path, REVISION)


def test_modified_receipt_and_extra_model_shards_cannot_change_pinned_weights(tmp_path):
    receipt = {"repo_id": MODEL_ID, "revision": REVISION, "files": [
        {"path": name, "sha256": digest, "size_bytes": 0}
        for name, digest in CHECKPOINT_SHA256.items()]}
    receipt["files"][0]["sha256"] = "forged-checksum"
    manifest = tmp_path / "download-manifest.json"
    manifest.write_text(json.dumps(receipt))
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="differs from pinned official revision"):
        _verify_local_checkpoint(tmp_path, REVISION)
    (tmp_path / "model-extra.safetensors").touch()
    with pytest.raises(ValueError, match="Unexpected model shards"):
        _verify_local_checkpoint(tmp_path, REVISION)
