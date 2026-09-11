#!/usr/bin/env python3
"""Numerical checks on actual model weights; separate from scored test data."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.backends_mlx import MLXBackend


def main():
    backend = MLXBackend("models/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
    mx = backend.mx
    a = backend.encode("Private registry. Record Alpha points to record Beta.\n")
    b = backend.encode("Record Beta has the access code CEDAR17.\n")
    q = backend.encode("What is the access code reached from Alpha? Answer:")
    left, right = backend.prefill(a), backend.prefill(b)
    original = mx.array(left.keys[0])
    joint = backend.prefill(a + b)
    appended = backend.append(left, b)
    full = backend.concat([left, right], bridge_ratio=1)
    partial = backend.concat([left, right], bridge_ratio=0.2)
    raw = backend.concat([left, right], bridge_ratio=0)
    joint_logits = backend.next_logits(joint, q)
    append_logits = backend.next_logits(appended, q)
    full_logits = backend.next_logits(full, q)
    def diff(x, y):
        return float(mx.max(mx.abs(x.astype(mx.float32) - y.astype(mx.float32))).item())
    shifted_error = diff(raw.keys[0][:, :, len(a):, :], joint.keys[0][:, :, len(a):, :])
    target = joint.keys[0][:, :, len(a):, :].astype(mx.float32)
    shifted = raw.keys[0][:, :, len(a):, :].astype(mx.float32)
    unshifted = right.keys[0].astype(mx.float32)
    def nrmse(value):
        return float(mx.sqrt(mx.sum((value - target)**2) / mx.sum(target**2)).item())
    shifted_nrmse, unshifted_nrmse = nrmse(shifted), nrmse(unshifted)
    checks = {
        "model": backend.metadata(),
        "append_vs_joint_next_logits_max_abs": diff(append_logits, joint_logits),
        "append_vs_joint_next_token_same": int(mx.argmax(append_logits)) == int(mx.argmax(joint_logits)),
        "full_bridge_vs_joint_next_logits_max_abs": diff(full_logits, joint_logits),
        "full_bridge_vs_joint_keys_max_abs": max(diff(x, y) for x, y in zip(full.keys, joint.keys)),
        "source_cache_mutation_max_abs": diff(original, left.keys[0]),
        "rope_shift_vs_joint_layer0_keys_max_abs": shifted_error,
        "rope_joint_layer0_key_max_abs": float(mx.max(mx.abs(target)).item()),
        "rope_shift_vs_joint_layer0_nrmse": shifted_nrmse,
        "rope_unshifted_vs_joint_layer0_nrmse": unshifted_nrmse,
        "partial_bridge_recomputed": partial.metadata["bridge_tokens_recomputed"],
        "partial_bridge_selected_fraction": partial.metadata["bridge_ratio_actual"],
        "native_bf16_bytes_per_token": joint.nbytes / joint.seq_len,
        "precision": {},
    }
    for bits in (8, 4):
        quantized = backend.quantize(raw, bits=bits)
        restored = backend.dequantize(quantized)
        checks["precision"][str(bits)] = {
            "bytes_per_token": quantized.nbytes / quantized.seq_len,
            "storage_fraction": quantized.nbytes / raw.nbytes,
            "dtype_after_dequantization": str(restored.keys[0].dtype),
        }
    checks["memory"] = backend.memory_stats()
    print(json.dumps(checks, indent=2), flush=True)
    assert checks["full_bridge_vs_joint_next_logits_max_abs"] == 0
    assert checks["full_bridge_vs_joint_keys_max_abs"] == 0
    assert checks["source_cache_mutation_max_abs"] == 0
    assert checks["append_vs_joint_next_token_same"]
    assert checks["append_vs_joint_next_logits_max_abs"] < 1
    # Composition of post-RoPE BF16 keys incurs two rounding steps. Absolute
    # error grows with key magnitude: use a scale-aware 1% relative norm
    # threshold and require at least a 10x improvement over no repositioning.
    assert shifted_nrmse < 0.01
    assert shifted_nrmse < unshifted_nrmse / 10
    assert 0 < checks["partial_bridge_selected_fraction"] < 1
    print("All real-weight numerical checks passed.", flush=True)


if __name__ == "__main__":
    main()
