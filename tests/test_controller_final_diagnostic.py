"""CPU checks of saved-evidence synthesis diagnostic; no model loading."""
import copy
import json

import pytest

from scripts.diagnose_controller_final import (
    PROLOGUE, SEARCH_RULE, SYSTEM, claim_output, document_text, file_sha,
    final_only_system, final_text, generate_final, ids_sha, reconstruct, verify_checkpoint,
)


class BytesBackend:
    def encode(self, text):
        return list(text.encode())


@pytest.fixture
def saved_case():
    backend = BytesBackend()
    corpus = {"a": {"id": "a", "title": "Alpha", "text": "The stored location is Somewhere."},
              "b": {"id": "b", "title": "Beta", "text": "Different evidence."}}
    config = {"max_document_tokens": 32, "max_answer_tokens": 48, "max_model_context_tokens": 8192}
    prefix, trace, evidence_count, seen = backend.encode(PROLOGUE), [], 0, []
    for i, doc in enumerate(corpus.values(), 1):
        fragment = backend.encode(document_text(doc))[:32]
        prefix += fragment
        evidence_count += len(fragment)
        seen.append(doc["id"])
        trace.append({"event": "retrieval", "round": i, "new_document_ids": [doc["id"]],
                      "accumulated_document_ids": seen.copy(), "evidence_tokens": evidence_count,
                      "evidence_token_ids_sha256": ids_sha(prefix)})
        trace.append({"event": "decision", "raw_action": "SECRET_CONTROLLER_ANSWER",
                      "controller_decode": {"raw_reasoning": "SECRET_THOUGHT"}})
    question = "Where is Alpha?"
    row = {"question": question, "answers": ["SECRET_GOLD"], "trace": trace,
           "evidence_ids": seen, "delivered_truncated_document_ids": seen.copy(),
           "counts": {"retrieved_documents": 2, "retrieved_evidence_tokens": evidence_count,
                      "final_prefill_tokens": len(prefix) + len(backend.encode(final_text(question)))}}
    return backend, row, corpus, config


def test_exact_segmented_reconstruction_and_single_sentence_ablation(saved_case):
    backend, row, corpus, config = saved_case
    prompts, evidence = reconstruct(backend, row, corpus, config)
    documents = [token for doc in corpus.values() for token in backend.encode(document_text(doc))[:32]]
    assert prompts["original"] == backend.encode(PROLOGUE) + documents + backend.encode(final_text(row["question"]))
    assert bytes(prompts["original"]).replace(SEARCH_RULE.encode(), b"", 1) == bytes(prompts["final_only_system"])
    assert final_only_system() == SYSTEM.replace(SEARCH_RULE, "", 1)
    assert evidence["evidence_tokens"] == 64
    for tokens in prompts.values():
        assert b"SECRET" not in bytes(tokens)  # no gold, controller output, or thought enters synthesis


@pytest.mark.parametrize("change", ["document", "order", "count", "hash", "truncation", "prefill"])
def test_reconstruction_rejects_changed_evidence_or_ledger(saved_case, change):
    backend, row, corpus, config = copy.deepcopy(saved_case)
    if change == "document":
        corpus["a"]["title"] = "CORRUPTED"
    elif change == "order":
        row["evidence_ids"].reverse()
    elif change == "count":
        row["trace"][0]["evidence_tokens"] += 1
    elif change == "hash":
        row["trace"][0]["evidence_token_ids_sha256"] = "wrong"
    elif change == "truncation":
        row["delivered_truncated_document_ids"] = []
    else:
        row["counts"]["final_prefill_tokens"] += 1
    with pytest.raises(ValueError, match="Reconstructed"):
        reconstruct(backend, row, corpus, config)


def test_output_cannot_overwrite_or_overlap_original_run(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    (source / "results.jsonl").write_text("preserve this")
    for target in (source, source / "child", tmp_path):
        with pytest.raises(ValueError):
            claim_output(target, [source])
    assert (source / "results.jsonl").read_text() == "preserve this"
    fresh = tmp_path / "diagnosis"
    claim_output(fresh, [source])
    with pytest.raises(ValueError, match="fresh diagnostic"):
        claim_output(fresh, [source])


def test_checkpoint_requires_pinned_weight_hash_and_exact_shard_inventory(tmp_path):
    model = tmp_path / "models" / "Tiny"
    model.mkdir(parents=True)
    shard = model / "model-00001.safetensors"
    shard.write_bytes(b"fake CPU fixture")
    receipt = {"revision": "pinned", "weights_dtype": "bfloat16", "files": [
        {"path": shard.name, "size_bytes": shard.stat().st_size, "sha256": file_sha(shard)}]}
    (model / "download-manifest.json").write_text(json.dumps(receipt))
    job = {"model_path": "models/Tiny", "model_revision": "pinned"}
    assert verify_checkpoint(job, tmp_path)["files"][0]["matched_download_sha256"]
    shard.write_bytes(b"different bytes!")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_checkpoint(job, tmp_path)


def test_final_decode_is_fresh_greedy_only_and_records_prompt_ids():
    class Backend:
        last_decode_stats = {"sampled_tokens": 2}
        def clear_cache(self):
            pass
        def synchronize(self):
            pass
        def decode(self, prefix, tokens, max_tokens):
            assert prefix is None and tokens == [1, 2] and max_tokens == 48
            return [65]
        def decode_tokens(self, output):
            return "A"
        def memory_stats(self):
            return {"fixture": True}
    result = generate_final(Backend(), [1, 2], 48)
    assert result["prompt_token_ids"] == [1, 2]
    assert result["output_token_ids"] == [65]
    assert result["work"] == {"prefill_tokens": 2, "generated_tokens": 1,
                              "sampled_tokens_including_eos": 2, "context_reservation_tokens": 50}
