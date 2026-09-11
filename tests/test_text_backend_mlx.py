"""Tiny CPU mechanics and pure startup guards; no downloads or real models."""
import json

import numpy as np
import pytest

from engine.text_backend_mlx import GIB, MLXTextBackend, validate_memory_budget

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
qwen3 = pytest.importorskip("mlx_lm.models.qwen3")
from mlx.utils import tree_flatten


class Tokenizer:
    eos_token_ids = {511}
    def encode(self, text, **kwargs):
        return list(text.encode())
    def decode(self, ids, **kwargs):
        return " ".join(str(token) for token in ids)
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking is False
        return self.encode("NATIVE:" + "|".join(message["role"] + ":" + message["content"] for message in messages) + ":ASSISTANT")


@pytest.fixture
def backend():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(29)
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=2, rms_norm_eps=1e-6,
        vocab_size=512, num_key_value_heads=1, max_position_embeddings=512,
        rope_theta=10000.0, head_dim=32, tie_word_embeddings=False)
    model = qwen3.Model(args)
    nn.quantize(model, group_size=32, bits=4)
    yield MLXTextBackend(model=model, tokenizer=Tokenizer(), config=vars(args),
                          max_context=512, memory_limit_gib=1, prefill_batch_size=7)
    mx.set_default_device(previous)


def test_model_precision_preserved_native_template_and_fresh_cache(backend):
    before = {name: np.asarray(value).copy() for name, value in tree_flatten(backend.model.parameters())}
    messages = [{"role": "system", "content": "Answer shortly"}, {"role": "user", "content": "Where?"}]
    result = backend.generate(messages, 4)
    assert bytes(result["prompt_token_ids"]).startswith(b"NATIVE:")
    assert result["stats"]["native_cache_types"] == ["KVCache", "KVCache"]
    assert result["stats"]["generated_tokens"] <= 4
    assert result["stats"]["context_reservation_tokens"] == len(result["prompt_token_ids"]) + 4
    repeat = backend.generate(messages, 4)
    assert repeat["output_token_ids"] == result["output_token_ids"]
    assert "mlx.core.uint32" in backend.metadata()["parameter_dtypes"]
    assert backend.metadata()["wired_limit_modified"] is False
    for name, value in tree_flatten(backend.model.parameters()):
        np.testing.assert_array_equal(np.asarray(value), before[name])


def test_sampling_is_repeatable_with_explicit_seed(backend):
    args = dict(max_tokens=8, temperature=.7, top_p=.8, top_k=20, seed=21, presence_penalty=1.5)
    messages = [{"role": "user", "content": "Continue"}]
    first = backend.generate(messages, **args)
    assert first["output_token_ids"] == backend.generate(messages, **args)["output_token_ids"]
    assert first["stats"]["sampling"]["seed"] == 21
    with pytest.raises(ValueError, match="explicit reproducible seed"):
        backend.generate(messages, max_tokens=8, temperature=.7)


def test_eos_and_output_cap_have_exact_sampling_counts(backend, monkeypatch):
    calls = []
    def forward(tokens, cache):
        calls.append(tuple(tokens))
        return mx.array([[10.0 if index == 511 else -10.0 for index in range(512)]])
    monkeypatch.setattr(backend, "_forward", forward)
    result = backend.generate([{"role": "user", "content": "x"}], 32)
    assert result["output_token_ids"] == []
    assert result["stats"]["sampled_token_ids"] == [511]
    assert result["stats"]["sampled_tokens_including_eos"] == 1
    assert result["stats"]["stop_reason"] == "eos"
    assert len(calls) == 1


def test_presence_penalty_uses_unique_generated_tokens_not_prompt(backend, monkeypatch):
    # Token65 (A, present in the template) initially wins. It must not be
    # penalized until generated. Thereafter B wins once, then A wins the tie.
    def forward(tokens, cache):
        logits = [-100.0] * 512
        logits[65], logits[66] = 3.0, 2.5
        return mx.array([logits])
    monkeypatch.setattr(backend, "_forward", forward)
    result = backend.generate([{"role": "user", "content": "AAAAA"}], 3, presence_penalty=1.0)
    assert result["output_token_ids"] == [65, 66, 65]
    assert result["stats"]["stop_reason"] == "max_tokens"
    assert result["stats"]["sampled_tokens_including_eos"] == 3


def test_full_native_template_output_reservation_precedes_forward(backend, monkeypatch):
    monkeypatch.setattr(backend, "max_context", 24)
    def forbidden(*args, **kwargs):
        raise AssertionError("Model must not run for an oversized reservation")
    monkeypatch.setattr(backend, "_forward", forbidden)
    with pytest.raises(ValueError, match="Complete prompt/output reservation"):
        backend.generate([{"role": "user", "content": "Large"}], 24)


def test_native_model_cache_factory_used_without_assuming_only_kv(backend, monkeypatch):
    seen = []
    class ArraysFixture:
        state = []
    class KVFixture:
        state = []
    def factory(model):
        cache = [ArraysFixture(), KVFixture()]
        seen.append(cache)
        return cache
    def forward(tokens, cache):
        assert cache is seen[-1]
        return mx.array([[10.0 if index == 511 else -10.0 for index in range(512)]])
    monkeypatch.setattr(backend, "_make_cache", factory)
    monkeypatch.setattr(backend, "_forward", forward)
    messages = [{"role": "user", "content": "x"}]
    first = backend.generate(messages, 2)
    backend.generate(messages, 2)
    assert seen[0] is not seen[1]
    assert first["stats"]["native_cache_types"] == ["ArraysFixture", "KVFixture"]


def budget(**changes):
    return validate_memory_budget(**{**dict(memory_limit_gib=88, min_available_gib=12,
        start_reserve_gib=4, checkpoint_bytes=79 * GIB,
        device_info={"max_recommended_working_set_size": 107 * GIB}, available_bytes=96 * GIB), **changes})


def test_startup_budget_accepts_room_for_checkpoint_and_floor():
    assert budget()["estimated_startup_bytes"] == 83 * GIB


@pytest.mark.parametrize("changes,match", [
    ({"memory_limit_gib": 105}, "104"),
    ({"device_info": {"max_recommended_working_set_size": 80 * GIB}}, "working set"),
    ({"checkpoint_bytes": 85 * GIB}, "startup reserve exceed"),
    ({"available_bytes": 94 * GIB}, "Insufficient available RAM"),
    ({"min_available_gib": -1}, "nonnegative"),
])
def test_startup_guard_rejects_unsafe_bounds(changes, match):
    with pytest.raises(ValueError, match=match):
        budget(**changes)


def test_local_config_rejects_custom_model_code_before_load(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "model_file": "custom.py"}))
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="custom model code"):
        MLXTextBackend(tmp_path, revision="pinned")
