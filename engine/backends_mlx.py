"""Native Qwen3 KV relay on MLX, using one shared base without role adapters.

Use MLXKVBlock and this backend's concat/quantize APIs: the NumPy store's block
format is intentionally separate. Keys are post-RoPE; concat applies a constant
phase shift for each new block offset. Stock causal attention then makes all
prefix tokens visible to the receiver. Bridge recomputation reruns selected
contiguous boundary spans against the preceding causal cache. This heuristic
is not a reproduction of CacheBlend token selection; 100% is full joint prefill.
MLX imports are deferred so mock-backend imports still work on other platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from engine.backends import Backend


@dataclass(frozen=True)
class MLXKVBlock:
    """Snapshot: native arrays [1, kv_heads, tokens, head_dim], positions 0..N-1.

    Quantized layers are (packed_uint32, scale, bias) tuples. ``nbytes`` counts
    real KV arrays including quantization overhead, excluding Python metadata.
    Original token IDs permit bridge recomputation without decoding a message.
    """

    keys: tuple[Any, ...]
    values: tuple[Any, ...]
    token_ids: tuple[int, ...]
    block_lengths: tuple[int, ...]
    chunk_id: str
    quant: str = "bf16"
    quant_bits: int | None = None
    group_size: int = 64
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("A block needs matching, nonempty K/V layers")
        if not self.token_ids or sum(self.block_lengths) != len(self.token_ids):
            raise ValueError("Block lengths must cover nonempty token IDs")
        if self.quant_bits not in (None, 4, 8):
            raise ValueError("Supported cache precision: bf16, 8-bit, 4-bit")
        for k, v in zip(self.keys, self.values):
            ka, va = (k[0], v[0]) if self.quant_bits else (k, v)
            if ka.shape != va.shape or len(ka.shape) != 4:
                raise ValueError("K/V must have matching four-dimensional shapes")
            if ka.shape[0] != 1 or ka.shape[2] != self.seq_len:
                raise ValueError("K/V token dimension does not match metadata")

    @property
    def seq_len(self):
        return len(self.token_ids)

    @property
    def token_count(self):
        return self.seq_len

    @property
    def n_layers(self):
        return len(self.keys)

    def arrays(self):
        if self.quant_bits:
            return tuple(a for layers in (self.keys, self.values)
                         for layer in layers for a in layer)
        return self.keys + self.values

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays())


class MLXBackend(Backend):
    """Local bf16 dense Qwen3, greedy bounded decoding; no implicit downloads.

    Model/tokenizer injection supports small CPU correctness tests. Production
    use requires an already downloaded directory. No generate() call or wired
    memory override is used. There are no trained role adapters in this backend.
    """

    name = "mlx-qwen3-shared-base"

    def __init__(self, model_path="models/Qwen3-8B", *, revision=None,
                 max_context=8192, memory_limit_gb=32, prefill_batch_size=256,
                 model=None, tokenizer=None):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        if not 0 < max_context <= 8192 or prefill_batch_size <= 0:
            raise ValueError("Context must be in (0, 8192]; prefill batch size must be positive")
        if not 0 < memory_limit_gb <= 32:
            raise ValueError("Memory limit must be in (0, 32] GiB")
        self.mx = mx
        self._cache_type = KVCache
        self.prefill_batch_size = prefill_batch_size
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        self.revision = revision
        self.model_path = str(Path(model_path).resolve())
        self.last_decode_stats = {}
        mx.set_memory_limit(self.memory_limit_bytes)
        mx.set_cache_limit(512 * 1024**2)
        if model is None:
            from mlx_lm import load
            local = Path(model_path)
            if not local.is_dir() or not (local / "config.json").is_file():
                raise FileNotFoundError(f"Local model not present: {local}")
            config = json.loads((local / "config.json").read_text())
            self._validate_config(config)
            model, tokenizer = load(str(local.resolve()),
                                    tokenizer_config={"trust_remote_code": False})
        else:
            config = vars(model.args).copy()
            self._validate_config(config)
        if tokenizer is None:
            raise ValueError("A tokenizer is required")
        self.model, self.tokenizer, self.config = model, tokenizer, config
        model.set_dtype(mx.bfloat16)
        model.eval()
        mx.eval(model.parameters())
        self.synchronize()
        self.max_context = min(max_context, config["max_position_embeddings"])
        self._identity = hashlib.sha256(json.dumps(
            {"path": self.model_path, "revision": revision, "config": config},
            sort_keys=True).encode()).hexdigest()[:24]

    @staticmethod
    def _validate_config(config):
        if config.get("model_type") != "qwen3":
            raise ValueError("Native relay currently supports dense Qwen3 only")
        if config.get("rope_scaling"):
            raise ValueError("RoPE relay requires unscaled standard Qwen3 RoPE")
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("This experiment requires unquantized bf16 weights")

    def _role(self, adapter):
        if adapter != "base":
            raise ValueError("No trained role adapters are installed; use adapter='base'")

    def _tokens(self, tokens):
        result = tuple(int(t) for t in tokens)
        if not result:
            raise ValueError("At least one token is required")
        if any(t < 0 or t >= self.vocab_size for t in result):
            raise ValueError("Token ID outside model vocabulary")
        return result

    def _check_length(self, length):
        if length > self.max_context:
            raise ValueError(f"Requested {length} tokens exceeds context cap {self.max_context}")

    def _check_block(self, block):
        if not isinstance(block, MLXKVBlock):
            raise TypeError("Use MLXKVBlock and MLXBackend.concat, not NumPy relay blocks")
        if block.n_layers != len(self.model.layers):
            raise ValueError("Relay layer count differs from this model")
        if block.metadata.get("backend_identity") != self._identity:
            raise ValueError("Relay was created by a different model/configuration")

    @property
    def vocab_size(self):
        return self.config["vocab_size"]

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def chat_tokens(self, system, user, add_generation_prompt=True):
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        return list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt,
            enable_thinking=False))

    def chat_prompt(self, user, system="Answer accurately using the provided evidence."):
        return self.chat_tokens(system, user)

    def _cache(self, prefix=None, end=None):
        caches = [self._cache_type() for _ in self.model.layers]
        if prefix is not None:
            prefix = self.dequantize(prefix)
            end = prefix.seq_len if end is None else end
            if end:
                for c, k, v in zip(caches, prefix.keys, prefix.values):
                    # Separate handles protect snapshots from later indexed writes.
                    c.state = (self.mx.array(k[:, :, :end, :]),
                               self.mx.array(v[:, :, :end, :]))
        return caches

    def _forward(self, tokens, cache, *, logits=False):
        """Bound prompt batches and avoid vocabulary logits at every position."""
        mx, h = self.mx, None
        for start in range(0, len(tokens), self.prefill_batch_size):
            x = mx.array([tokens[start:start + self.prefill_batch_size]], dtype=mx.int32)
            h = self.model.model(x, cache=cache)
            mx.eval(h, [c.state for c in cache])
        if logits:
            h = h[:, -1:, :]
            if self.config["tie_word_embeddings"]:
                result = self.model.model.embed_tokens.as_linear(h)
            else:
                result = self.model.lm_head(h)
            mx.eval(result)
            return result[0, -1, :]
        return h

    def _block(self, cache, tokens, lengths, metadata=None):
        keys, values = zip(*(c.state for c in cache))
        keys = tuple(k.astype(self.mx.bfloat16) for k in keys)
        values = tuple(v.astype(self.mx.bfloat16) for v in values)
        self.mx.eval(keys, values)
        return MLXKVBlock(keys, values, tuple(tokens), tuple(lengths),
            hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()[:24],
            metadata={"backend_identity": self._identity, **(metadata or {})})

    def prefill(self, tokens, adapter="base"):
        self._role(adapter)
        tokens = self._tokens(tokens)
        self._check_length(len(tokens))
        cache = self._cache()
        self._forward(tokens, cache)
        return self._block(cache, tokens, (len(tokens),), {"prefill": "independent_causal"})

    def append(self, prefix, tokens, adapter="base"):
        """Transform new tokens against a prior cache, returning a new snapshot."""
        self._role(adapter)
        if prefix is None:
            return self.prefill(tokens)
        self._check_block(prefix)
        tokens = self._tokens(tokens)
        self._check_length(prefix.seq_len + len(tokens))
        cache = self._cache(prefix)
        self._forward(tokens, cache)
        return self._block(cache, prefix.token_ids + tokens,
            prefix.block_lengths + (len(tokens),),
            {**prefix.metadata, "last_append_tokens": len(tokens)})

    def next_logits(self, prefix, prompt_tokens):
        """Evaluated next-token logits for real numerical cache diagnostics."""
        if prefix is not None:
            self._check_block(prefix)
        tokens = self._tokens(prompt_tokens)
        self._check_length((prefix.seq_len if prefix else 0) + len(tokens))
        return self._forward(tokens, self._cache(prefix), logits=True)

    def decode(self, prefix, prompt_tokens, adapter="base", max_tokens=64):
        self._role(adapter)
        if max_tokens < 0:
            raise ValueError("max_tokens must be nonnegative")
        if prefix is not None:
            self._check_block(prefix)
        tokens = self._tokens(prompt_tokens)
        prompt_length = (prefix.seq_len if prefix else 0) + len(tokens)
        self._check_length(prompt_length + max_tokens)
        if max_tokens == 0:
            self.last_decode_stats = {"generated_tokens": 0, "stop_reason": "max_tokens"}
            return []
        caches = self._cache(prefix)
        logits = self._forward(tokens, caches, logits=True)
        eos = getattr(self.tokenizer, "eos_token_ids", None)
        if eos is None:
            eos = self.config.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        eos = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
        out, stop = [], "max_tokens"
        for index in range(max_tokens):
            nxt = int(self.mx.argmax(logits).item())
            if nxt in eos:
                stop = "eos"
                break
            out.append(nxt)
            if index + 1 < max_tokens:
                logits = self._forward((nxt,), caches, logits=True)
        self.synchronize()
        self.last_decode_stats = {"generated_tokens": len(out), "stop_reason": stop,
            "prefill_tokens": len(tokens), "prefix_tokens": prompt_length - len(tokens)}
        return out

    def hidden_state(self, tokens, adapter="base"):
        self._role(adapter)
        tokens = self._tokens(tokens)
        self._check_length(len(tokens))
        h = self._forward(tokens, self._cache())
        return np.array(h[0, -1, :].astype(self.mx.float32))

    def _shift_keys(self, keys, layer, delta):
        if delta == 0:
            return keys
        # Singleton sequences apply CONSTANT phase; rope(keys, offset=delta)
        # would incorrectly add the within-block token index a second time.
        shape = keys.shape
        singles = keys.astype(self.mx.float32).reshape(-1, 1, shape[-1])
        shifted = self.model.layers[layer].self_attn.rope(singles, offset=delta)
        return shifted.reshape(shape).astype(self.mx.bfloat16)

    @staticmethod
    def _bridge_positions(lengths, ratio):
        total = sum(lengths)
        if ratio <= 0:
            return []
        if ratio >= 1:
            return list(range(total))
        boundaries = np.cumsum(lengths)[:-1].tolist()
        if not boundaries:
            return []
        ranked = sorted(range(total), key=lambda p: (
            min(min(abs(p - b), abs(p - (b - 1))) for b in boundaries), p))
        return sorted(ranked[:math.ceil(total * ratio)])

    def concat(self, blocks, bridge_ratio=0.0):
        if not blocks:
            raise ValueError("concat requires at least one block")
        if not 0 <= bridge_ratio <= 1:
            raise ValueError("bridge_ratio must lie between zero and one")
        for block in blocks:
            self._check_block(block)
        interventions = set()
        for block in blocks:
            if block.metadata.get("cache_intervention"):
                interventions.update(block.metadata.get(
                    "cache_interventions", [block.metadata["cache_intervention"]]))
        total = sum(b.seq_len for b in blocks)
        self._check_length(total)
        if bridge_ratio and interventions:
            raise ValueError("Recomputation would undo an audit cache intervention")
        tokens = tuple(t for b in blocks for t in b.token_ids)
        lengths = tuple(n for b in blocks for n in b.block_lengths)
        if bridge_ratio == 1:
            block = self.prefill(list(tokens))
            return replace(block, block_lengths=lengths, metadata={**block.metadata,
                "bridge_ratio_requested": 1.0, "bridge_tokens_recomputed": total,
                "bridge_ratio_actual": 1.0, "prefill": "joint_causal"})
        work = [self.dequantize(b) for b in blocks]
        keys, values = [], []
        for layer in range(len(self.model.layers)):
            offset, shifted = 0, []
            for b in work:
                shifted.append(self._shift_keys(b.keys[layer], layer, offset))
                offset += b.seq_len
            keys.append(self.mx.concatenate(shifted, axis=2))
            values.append(self.mx.concatenate([b.values[layer] for b in work], axis=2))
        self.mx.eval(keys, values)
        merged = MLXKVBlock(tuple(keys), tuple(values), tokens, lengths,
            "+".join(b.chunk_id for b in blocks), metadata={"backend_identity": self._identity,
            "bridge_ratio_requested": bridge_ratio, "bridge_tokens_recomputed": 0,
            "bridge_ratio_actual": 0.0, "prefill": "independent_causal_concat_rope"})
        if interventions:
            merged = replace(merged, metadata={**merged.metadata,
                "cache_intervention": next(iter(interventions)) if len(interventions) == 1 else "mixed",
                "cache_interventions": sorted(interventions)})
        selected = self._bridge_positions(lengths, bridge_ratio)
        if selected:
            starts, ends = [selected[0]], []
            for left, right in zip(selected, selected[1:]):
                if right != left + 1:
                    ends.append(left + 1)
                    starts.append(right)
            ends.append(selected[-1] + 1)
            for start, end in zip(starts, ends):
                cache = self._cache(merged, end=start)
                self._forward(tokens[start:end], cache)
                new_keys, new_values = zip(*(c.state for c in cache))
                keys = tuple(self.mx.concatenate((nk, old[:, :, end:, :]), axis=2)
                             for nk, old in zip(new_keys, merged.keys))
                values = tuple(self.mx.concatenate((nv, old[:, :, end:, :]), axis=2)
                               for nv, old in zip(new_values, merged.values))
                self.mx.eval(keys, values)
                merged = replace(merged, keys=keys, values=values)
            merged = replace(merged, metadata={**merged.metadata,
                "bridge_tokens_recomputed": len(selected),
                "bridge_ratio_actual": len(selected) / total,
                "bridge_method": "causal_boundary_spans"})
        return merged

    def quantize(self, block, bits=8, group_size=64):
        """Real packed affine storage; attention consumes dequantized bf16."""
        self._check_block(block)
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        source = self.dequantize(block)
        if group_size not in (32, 64, 128) or source.keys[0].shape[-1] % group_size:
            raise ValueError("group_size must be 32/64/128 and divide head dimension")
        keys = tuple(self.mx.quantize(k, bits=bits, group_size=group_size) for k in source.keys)
        values = tuple(self.mx.quantize(v, bits=bits, group_size=group_size) for v in source.values)
        result = replace(source, keys=keys, values=values, quant=f"int{bits}",
            quant_bits=bits, group_size=group_size, metadata={**source.metadata,
            "quantization": "mlx_affine_packed", "quantization_bits": bits,
            "quantization_group_size": group_size, "unquantized_kv_bytes": source.nbytes,
            "attention_precision": "bf16_after_dequantization"})
        self.mx.eval(result.arrays())
        return result

    def dequantize(self, block):
        self._check_block(block)
        if not block.quant_bits:
            return block
        def deq(x):
            return self.mx.dequantize(*x, group_size=block.group_size,
                                      bits=block.quant_bits).astype(self.mx.bfloat16)
        keys, values = tuple(deq(k) for k in block.keys), tuple(deq(v) for v in block.values)
        self.mx.eval(keys, values)
        return replace(block, keys=keys, values=values, quant="bf16", quant_bits=None)

    def perturb(self, block, mode, seed=0):
        """Zero or moment-match random K/V per layer and KV head for E3."""
        self._check_block(block)
        if mode not in ("zero", "random", "moment_matched_random"):
            raise ValueError("mode must be zero or random")
        source, mx = self.dequantize(block), self.mx
        rng = mx.random.key(seed)
        def changed(x):
            nonlocal rng
            if mode == "zero":
                return mx.zeros_like(x)
            rng, key = mx.random.split(rng)
            xf, axes = x.astype(mx.float32), (2, 3)
            noise = mx.random.normal(x.shape, key=key)
            noise = (noise - mx.mean(noise, axis=axes, keepdims=True)) / mx.maximum(
                mx.std(noise, axis=axes, keepdims=True), 1e-8)
            return (noise * mx.std(xf, axis=axes, keepdims=True) +
                    mx.mean(xf, axis=axes, keepdims=True)).astype(mx.bfloat16)
        result = replace(source, keys=tuple(changed(k) for k in source.keys),
            values=tuple(changed(v) for v in source.values),
            metadata={**source.metadata, "cache_intervention": mode,
                      "intervention_seed": seed,
                      "moment_matching_axes": "per_layer_per_kv_head_over_tokens_and_dimension"})
        mx.eval(result.arrays())
        return result

    def synchronize(self):
        self.mx.synchronize()

    def clear_cache(self):
        """Release unused allocator buffers after callers drop old block references."""
        self.synchronize()
        self.mx.clear_cache()

    def memory_stats(self):
        return {"active_bytes": self.mx.get_active_memory(),
                "peak_bytes": self.mx.get_peak_memory(),
                "cache_bytes": self.mx.get_cache_memory(),
                "memory_limit_bytes": self.memory_limit_bytes}

    def metadata(self):
        versions = {}
        for package in ("mlx", "mlx-lm", "transformers", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        return {"backend": self.name, "model_path": self.model_path,
                "model_revision": self.revision, "model_type": self.config["model_type"],
                "backend_identity": self._identity, "weight_dtype": "bfloat16",
                "kv_baseline_dtype": "bfloat16", "role_adapters": None,
                "decoding": "greedy", "thinking": False,
                "max_context": self.max_context, "prefill_batch_size": self.prefill_batch_size,
                "memory_limit_bytes": self.memory_limit_bytes,
                "rope_reindexing": "constant_phase_shift_of_post_rope_keys",
                "bridge_method": "causal_boundary_spans",
                "quantization": "packed_affine_storage_dequantized_for_attention",
                "versions": versions}
