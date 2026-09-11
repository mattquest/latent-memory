#!/usr/bin/env python3
"""Bounded direct-library sanity check before the custom relay backend."""
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main():
    mx.set_memory_limit(32 * 1024**3)
    mx.set_cache_limit(1024**3)
    started = time.perf_counter()
    model, tokenizer = load("models/Qwen3-8B", tokenizer_config={"trust_remote_code": False})
    mx.eval(model.parameters())
    print(json.dumps({"event": "loaded", "seconds": time.perf_counter() - started,
                      "peak_gib": mx.get_peak_memory() / 1024**3}), flush=True)
    cases = [
        ("The private archive's current access code is CEDAR17. What is its current access code? Reply with only the code.", "CEDAR17"),
        ("Archive A points to archive B. Archive B has the color violet. What color does the archive reached from A have? Reply with only the color.", "violet"),
    ]
    for question, expected in cases:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )
        cache = make_prompt_cache(model)
        started = time.perf_counter()
        logits = model(mx.array([tokens]), cache=cache)
        output = []
        for _ in range(24):
            token = int(mx.argmax(logits[0, -1]).item())
            if token in tokenizer.eos_token_ids:
                break
            output.append(token)
            logits = model(mx.array([[token]]), cache=cache)
        mx.synchronize()
        answer = tokenizer.decode(output)
        print(json.dumps({"question": question, "answer": answer, "expected": expected,
                          "correct": answer.strip().lower() == expected.lower(),
                          "tokens": len(output), "input_tokens": len(tokens),
                          "seconds": time.perf_counter() - started,
                          "peak_gib": mx.get_peak_memory() / 1024**3}), flush=True)
        del cache, logits
        mx.clear_cache()


if __name__ == "__main__":
    main()
