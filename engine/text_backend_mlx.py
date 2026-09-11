"""Bounded native-template MLX-LM inference for local text-only pilots.

This backend intentionally exposes no relay/append API. It keeps checkpoint
weight precision unchanged and lets each model build its native cache, including
the recurrent state used by Qwen3.5. It never changes the wired-memory limit.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

GIB = 1024**3
SUPPORTED = {"qwen3", "qwen3_moe", "qwen3_5_moe"}


def validate_memory_budget(*, memory_limit_gib, min_available_gib, start_reserve_gib,
                           checkpoint_bytes, device_info, available_bytes):
    """Check the declared allocation bound and conservative startup headroom."""
    if not math.isfinite(memory_limit_gib) or not 0 < memory_limit_gib <= 104:
        raise ValueError("Text-only MLX memory limit must be in (0, 104] GiB")
    if any(not math.isfinite(value) or value < 0 for value in (min_available_gib, start_reserve_gib)):
        raise ValueError("Available-memory floor and startup reserve must be finite and nonnegative")
    recommended = int(device_info.get("max_recommended_working_set_size", 0))
    limit = int(memory_limit_gib * GIB)
    if recommended <= 0 or limit > recommended:
        raise ValueError("Declared MLX allocation limit exceeds the device recommended working set")
    estimate = checkpoint_bytes + int(start_reserve_gib * GIB)
    if estimate > limit:
        raise ValueError("Checkpoint bytes plus startup reserve exceed the declared MLX allocation limit")
    if available_bytes < estimate + min_available_gib * GIB:
        raise ValueError("Insufficient available RAM for checkpoint, startup reserve, and the required free-memory floor")
    return {"memory_limit_bytes": limit, "device_recommended_working_set_bytes": recommended,
            "checkpoint_bytes": checkpoint_bytes, "startup_reserve_bytes": int(start_reserve_gib * GIB),
            "estimated_startup_bytes": estimate, "available_bytes_before_load": available_bytes,
            "minimum_available_bytes": int(min_available_gib * GIB),
            "limitation": "Preload estimate, not a bound on peak runtime RSS; an outer process guard remains required."}


class MLXTextBackend:
    name = "mlx-lm-native-template-text-only"

    def __init__(self, model_path=None, *, revision=None, max_context=8192,
                 memory_limit_gib=32, min_available_gib=12, start_reserve_gib=4,
                 prefill_batch_size=256, model=None, tokenizer=None, config=None):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.cache import make_prompt_cache
        import psutil

        if not isinstance(max_context, int) or not 0 < max_context <= 8192:
            raise ValueError("The complete text context cap must be in (0, 8192]")
        if not isinstance(prefill_batch_size, int) or not 0 < prefill_batch_size <= 256:
            raise ValueError("Prefill batches must be in (0, 256]")
        self.mx, self._make_cache = mx, make_prompt_cache
        self.revision = revision
        self.model_path = str(Path(model_path).resolve()) if model_path is not None else "injected-cpu-fixture"
        self.prefill_batch_size = prefill_batch_size
        injected = model is not None
        if not injected:
            local = Path(self.model_path)
            if not local.is_dir() or not (local / "config.json").is_file():
                raise FileNotFoundError("A complete local model directory is required; implicit downloads are disabled")
            if not revision:
                raise ValueError("A pinned model revision is required")
            config = json.loads((local / "config.json").read_text())
            shards = sorted(local.glob("model*.safetensors"))
            if not shards or any(path.is_symlink() for path in shards):
                raise ValueError("Local model must contain ordinary safetensors shards")
            checkpoint_bytes = sum(path.stat().st_size for path in shards)
            self._checkpoint_files = [{"name": path.name, "bytes": path.stat().st_size} for path in shards]
        else:
            if config is None or tokenizer is None:
                raise ValueError("Injected test models require explicit tokenizer and config")
            checkpoint_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
            self._checkpoint_files = []
        if config.get("model_type") not in SUPPORTED or config.get("model_file"):
            raise ValueError("Only allowlisted built-in Qwen architectures are supported; custom model code is forbidden")
        if config.get("text_config", {}).get("model_file"):
            raise ValueError("Custom model code is forbidden")
        self.config = config
        text_config = config.get("text_config", config)
        self.max_context = min(max_context, int(text_config.get("max_position_embeddings", max_context)))
        self.vocab_size = int(text_config["vocab_size"])
        # Injected tiny CPU models do not allocate Metal model weights. Pure
        # guard tests cover the production budget checks without loading them.
        self.startup_memory = ({"injected_cpu_fixture": True, "memory_limit_bytes": int(memory_limit_gib * GIB)}
            if injected else validate_memory_budget(memory_limit_gib=memory_limit_gib,
                min_available_gib=min_available_gib, start_reserve_gib=start_reserve_gib,
                checkpoint_bytes=checkpoint_bytes, device_info=mx.device_info(),
                available_bytes=psutil.virtual_memory().available))
        self.memory_limit_bytes = self.startup_memory["memory_limit_bytes"]
        mx.set_memory_limit(self.memory_limit_bytes)
        mx.set_cache_limit(512 * 1024**2)
        started = time.perf_counter()
        if not injected:
            from mlx_lm import load
            model, tokenizer, loaded_config = load(self.model_path,
                tokenizer_config={"trust_remote_code": False, "local_files_only": True},
                return_config=True)
            self.config = loaded_config
        self.model, self.tokenizer = model, tokenizer
        self.model.eval()
        # No dtype casts: packed integers, scales, auxiliary weights and model
        # recurrent-state initialization retain their checkpoint precision.
        mx.eval(self.model.parameters())
        mx.synchronize()
        self.model_load_ms = (time.perf_counter() - started) * 1000
        self.parameter_bytes = sum(value.nbytes for _, value in tree_flatten(self.model.parameters()))
        self.parameter_dtypes = sorted({str(value.dtype) for _, value in tree_flatten(self.model.parameters())})
        self.native_cache_types = [type(cache).__name__ for cache in self._make_cache(self.model)]
        self.last_decode_stats = {}

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def chat_tokens(self, messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Provide nonempty native chat messages")
        if any(not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
               or not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("Only plain-text system/user/assistant messages are supported")
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False))

    def synchronize(self):
        self.mx.synchronize()

    def clear_cache(self):
        self.synchronize()
        self.mx.clear_cache()

    def memory_stats(self):
        return {"active_bytes": self.mx.get_active_memory(), "peak_bytes": self.mx.get_peak_memory(),
                "allocator_cache_bytes": self.mx.get_cache_memory(), "memory_limit_bytes": self.memory_limit_bytes}

    def _forward(self, tokens, cache):
        logits = None
        for start in range(0, len(tokens), self.prefill_batch_size):
            batch = self.mx.array([tokens[start:start + self.prefill_batch_size]], dtype=self.mx.int32)
            output = self.model(batch, cache=cache)
            logits = output[:, -1, :]
            self.mx.eval(logits, [entry.state for entry in cache])
        return logits

    def generate(self, messages, max_tokens, temperature=0.0, top_p=1.0, top_k=0,
                 seed=None, presence_penalty=0.0):
        """Generate once from fresh native state; never retain controller output."""
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("Generation token budget must be a positive integer")
        if (not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or
                not isinstance(top_k, int) or not 0 <= top_k < self.vocab_size):
            raise ValueError("Invalid temperature/top-p/top-k sampling configuration")
        if not math.isfinite(presence_penalty) or not -2 <= presence_penalty <= 2:
            raise ValueError("Presence penalty must lie in [-2, 2]")
        if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 2**32):
            raise ValueError("Sampling seed must be an unsigned 32-bit integer")
        if temperature > 0 and seed is None:
            raise ValueError("Non-greedy sampling requires an explicit reproducible seed")
        started = time.perf_counter()
        prompt = self.chat_tokens(messages)
        if not prompt or any(not isinstance(token, int) or not 0 <= token < self.vocab_size for token in prompt):
            raise ValueError("Native chat template produced empty or invalid token IDs")
        reservation = len(prompt) + max_tokens
        if reservation > self.max_context:
            raise ValueError(f"Complete prompt/output reservation {reservation} exceeds context cap {self.max_context}")
        template_ms = (time.perf_counter() - started) * 1000
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        if temperature > 0:
            self.mx.random.seed(seed)
        eos = set(self.tokenizer.eos_token_ids)
        caches = self._make_cache(self.model)
        self.synchronize()
        model_started = time.perf_counter()
        logits = self._forward(prompt, caches)
        self.synchronize()
        prefill_ms = (time.perf_counter() - model_started) * 1000
        generation_started = time.perf_counter()
        output, sampled, stop = [], [], "max_tokens"
        for index in range(max_tokens):
            if presence_penalty and output:
                # Unique indices enforce presence rather than frequency. The
                # evidence/prompt is excluded to avoid suppressing answer names.
                indices = self.mx.array(sorted(set(output)), dtype=self.mx.int32)
                logits = logits.at[:, indices].add(-presence_penalty)
            logprobs = logits - self.mx.logsumexp(logits, axis=-1, keepdims=True)
            token = int(sampler(logprobs).item())
            sampled.append(token)
            if token in eos:
                stop = "eos"
                break
            output.append(token)
            if index + 1 < max_tokens:
                logits = self._forward([token], caches)
        self.synchronize()
        generation_ms = (time.perf_counter() - generation_started) * 1000
        raw = self.decode_tokens(output)
        self.last_decode_stats = {"stop_reason": stop, "prefill_tokens": len(prompt),
            "generated_tokens": len(output), "sampled_tokens_including_eos": len(sampled),
            "sampled_token_ids": sampled, "context_reservation_tokens": reservation,
            "max_output_tokens": max_tokens, "native_cache_types": [type(cache).__name__ for cache in caches],
            "template_ms": template_ms, "prefill_ms": prefill_ms, "generation_ms": generation_ms,
            "model_inference_ms": prefill_ms + generation_ms,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed,
                         "presence_penalty": presence_penalty, "presence_scope": "unique generated tokens in this call; prompt excluded"},
            "thinking": False, "memory": self.memory_stats()}
        return {"raw_output": raw, "output_token_ids": output, "prompt_token_ids": prompt,
                "stats": self.last_decode_stats}

    def metadata(self):
        return {"backend": self.name, "model_path": self.model_path, "model_revision": self.revision,
            "model_type": self.config["model_type"], "model_config": self.config,
            "config_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
            "quantization": self.config.get("quantization", self.config.get("quantization_config")),
            "parameter_bytes": self.parameter_bytes, "parameter_dtypes": self.parameter_dtypes,
            "weight_dtype_policy": "unchanged checkpoint precision; no cast or dequantization",
            "native_cache_types": self.native_cache_types, "chat_template": "checkpoint native; enable_thinking=False",
            "max_context": self.max_context, "prefill_batch_size": self.prefill_batch_size,
            "model_load_ms": self.model_load_ms, "startup_memory_check": self.startup_memory,
            "checkpoint_files": self._checkpoint_files, "wired_limit_modified": False,
            "versions": {package: importlib.metadata.version(package) for package in ("mlx", "mlx-lm", "transformers")}}
