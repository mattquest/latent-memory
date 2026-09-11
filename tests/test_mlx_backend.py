"""Small random Qwen3 mechanics tests on CPU; never download or load 8B weights."""

from dataclasses import replace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
qwen3 = pytest.importorskip("mlx_lm.models.qwen3")

from engine.backends_mlx import MLXBackend, MLXKVBlock


class TinyTokenizer:
    eos_token_ids = {255}

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode(errors="replace")


@pytest.fixture(scope="module")
def backend():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=3,
        intermediate_size=128, num_attention_heads=2, rms_norm_eps=1e-6,
        vocab_size=256, num_key_value_heads=1, max_position_embeddings=512,
        rope_theta=10000.0, head_dim=32, tie_word_embeddings=False,
    )
    result = MLXBackend(model=qwen3.Model(args), tokenizer=TinyTokenizer(),
                        max_context=512, memory_limit_gb=1, prefill_batch_size=7)
    yield result
    mx.set_default_device(previous)


def as_numpy(array):
    return np.array(array.astype(mx.float32))


def blocks(backend):
    tokens = list(range(10, 26))
    return tokens, backend.prefill(tokens[:8]), backend.prefill(tokens[8:])


def test_native_bfloat16_cache_and_byte_count(backend):
    block = backend.prefill([1, 2, 3])
    assert isinstance(block, MLXKVBlock)
    assert block.keys[0].shape == (1, 1, 3, 32)
    assert all(a.dtype == mx.bfloat16 for a in block.arrays())
    assert block.nbytes == 3 * 1 * 3 * 32 * 2 * 2
    assert backend.metadata()["role_adapters"] is None


def test_append_matches_joint_causal_forward_without_mutating_source(backend):
    tokens, left, right = blocks(backend)
    before = [as_numpy(a).copy() for a in left.arrays()]
    appended = backend.append(left, list(right.token_ids))
    joint = backend.prefill(tokens)
    for a, j in zip(appended.arrays(), joint.arrays()):
        np.testing.assert_allclose(as_numpy(a), as_numpy(j), atol=0.02, rtol=0.01)
    for a, original in zip(left.arrays(), before):
        np.testing.assert_array_equal(as_numpy(a), original)
    actual = backend.next_logits(appended, [32])
    expected = backend.next_logits(None, tokens + [32])
    np.testing.assert_allclose(as_numpy(actual), as_numpy(expected), atol=0.02, rtol=0.01)


def test_concat_rephases_every_key_once_and_leaves_values_unchanged(backend):
    tokens, left, right = blocks(backend)
    merged = backend.concat([left, right])
    joint = backend.prefill(tokens)
    # Layer zero K depends only on the same token embedding and its position.
    # Hence concatenation must agree with full-prefill layer zero after RoPE.
    np.testing.assert_allclose(as_numpy(merged.keys[0]), as_numpy(joint.keys[0]),
                               atol=0.02, rtol=0.01)
    for layer in range(left.n_layers):
        np.testing.assert_array_equal(as_numpy(merged.values[layer][:, :, 8:, :]),
                                      as_numpy(right.values[layer]))
    # Upper layers retain genuinely independent rather than joint attention.
    assert np.max(np.abs(as_numpy(merged.keys[1]) - as_numpy(joint.keys[1]))) > 0.1


@pytest.mark.parametrize("offset", [12, 6000])
def test_rope_composition_error_is_small_relative_to_key_scale(backend, offset):
    # BF16 roundoff scales with key magnitude; a fixed absolute threshold does
    # not distinguish phase errors. Compare normalized RMS with an unshifted
    # negative control at both short and long offsets.
    rope = backend.model.layers[0].self_attn.rope
    raw = (mx.random.normal((1, 1, 24, 32), key=mx.random.key(19)) * 16).astype(mx.bfloat16)
    local = rope(raw)
    expected = as_numpy(rope(raw, offset=offset))
    shifted = as_numpy(backend._shift_keys(local, 0, offset))
    denominator = np.linalg.norm(expected)
    relative_error = np.linalg.norm(shifted - expected) / denominator
    negative_control = np.linalg.norm(as_numpy(local) - expected) / denominator
    assert relative_error < 0.01
    assert relative_error < negative_control * 0.1


def test_full_bridge_is_identical_to_joint_prefill(backend):
    tokens, left, right = blocks(backend)
    bridged = backend.concat([left, right], bridge_ratio=1)
    joint = backend.prefill(tokens)
    for a, j in zip(bridged.arrays(), joint.arrays()):
        np.testing.assert_array_equal(as_numpy(a), as_numpy(j))
    assert bridged.metadata["bridge_tokens_recomputed"] == len(tokens)


def test_partial_bridge_recomputes_selected_tokens_only(backend):
    tokens, left, right = blocks(backend)
    original = backend.concat([left, right])
    bridged = backend.concat([left, right], bridge_ratio=0.25)
    selected = backend._bridge_positions((8, 8), 0.25)
    assert selected == [6, 7, 8, 9]
    assert bridged.metadata["bridge_tokens_recomputed"] == 4
    remaining = [i for i in range(len(tokens)) if i not in selected]
    for a, o in zip(bridged.arrays(), original.arrays()):
        np.testing.assert_array_equal(as_numpy(a)[:, :, remaining, :],
                                      as_numpy(o)[:, :, remaining, :])
    assert np.max(np.abs(as_numpy(bridged.keys[1]) - as_numpy(original.keys[1]))) > 0.1


@pytest.mark.parametrize("bits,max_relative_error", [(8, 0.02), (4, 0.15)])
def test_packed_quantization_has_real_reduced_storage(backend, bits, max_relative_error):
    source = backend.prefill(list(range(30)))
    quant = backend.quantize(source, bits=bits, group_size=32)
    restored = backend.dequantize(quant)
    assert quant.keys[0][0].dtype == mx.uint32
    assert quant.nbytes < source.nbytes
    assert quant.nbytes == sum(a.nbytes for a in quant.arrays())
    expected_ratio = (bits / 8 + 4 / 32) / 2  # bf16 scale + bias per group
    assert quant.nbytes / source.nbytes == expected_ratio
    for orig, actual in zip(source.arrays(), restored.arrays()):
        error = np.linalg.norm(as_numpy(orig) - as_numpy(actual)) / np.linalg.norm(as_numpy(orig))
        assert error < max_relative_error
    assert restored.seq_len == source.seq_len


def test_zero_and_random_audits_preserve_structure_and_random_moments(backend):
    source = backend.prefill(list(range(30)))
    zero = backend.perturb(source, "zero")
    random = backend.perturb(source, "random", seed=7)
    repeat = backend.perturb(source, "random", seed=7)
    assert zero.token_ids == random.token_ids == source.token_ids
    assert all(np.count_nonzero(as_numpy(a)) == 0 for a in zero.arrays())
    for original, noise, duplicate in zip(source.arrays(), random.arrays(), repeat.arrays()):
        o, n = as_numpy(original), as_numpy(noise)
        np.testing.assert_array_equal(n, as_numpy(duplicate))
        np.testing.assert_allclose(n.mean(axis=(2, 3)), o.mean(axis=(2, 3)), atol=0.001)
        np.testing.assert_allclose(n.std(axis=(2, 3)), o.std(axis=(2, 3)), atol=0.003)
        assert not np.array_equal(o, n)
    with pytest.raises(ValueError, match="undo an audit"):
        backend.concat([random], bridge_ratio=1)


def test_audit_intervention_guard_survives_concat_append_and_quantization(backend):
    source = backend.prefill([1, 2, 3])
    zero = backend.perturb(source, "zero")
    random = backend.perturb(source, "random", seed=9)
    merged = backend.concat([source, zero, random])
    assert merged.metadata["cache_intervention"] == "mixed"
    assert merged.metadata["cache_interventions"] == ["random", "zero"]
    appended = backend.append(merged, [4])
    quantized = backend.quantize(appended, bits=4, group_size=32)
    merged_again = backend.concat([source, quantized])
    with pytest.raises(ValueError, match="undo an audit"):
        backend.concat([merged_again], bridge_ratio=1)


def test_decode_is_bounded_greedy_and_does_not_mutate_prefix(backend):
    prefix = backend.prefill([1, 2, 3])
    before = [as_numpy(a).copy() for a in prefix.arrays()]
    out = backend.decode(prefix, [4], max_tokens=3)
    assert out == backend.decode(prefix, [4], max_tokens=3)
    assert len(out) <= 3
    for a, original in zip(prefix.arrays(), before):
        np.testing.assert_array_equal(as_numpy(a), original)
    assert backend.decode(prefix, [4], max_tokens=0) == []
    with pytest.raises(ValueError, match="context cap"):
        backend.decode(prefix, [4], max_tokens=509)


def test_rejects_invalid_inputs_and_claimed_adapters(backend):
    with pytest.raises(ValueError, match="At least one"):
        backend.prefill([])
    with pytest.raises(ValueError, match="vocabulary"):
        backend.prefill([256])
    with pytest.raises(ValueError, match="No trained role adapters"):
        backend.prefill([1], adapter="reader")
    with pytest.raises(TypeError, match="MLXKVBlock"):
        backend.concat([object()])
    source = backend.prefill([1])
    other = replace(source, metadata={"backend_identity": "other-weights"})
    with pytest.raises(ValueError, match="different model"):
        backend.decode(other, [2], max_tokens=1)


def test_hidden_state_is_last_hidden_vector(backend):
    hidden = backend.hidden_state([1, 2, 3])
    assert hidden.shape == (64,)
    assert np.isfinite(hidden).all()
