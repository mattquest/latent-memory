# Large local controller development screen

All results use the same development questions. Differences combine model and precision changes.

| Model | Arm | EM | F1 | Cold p50 / p95 (s) | New support IDs |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-14B BF16 | basic_rag | 0.0/12 | 0.000 | 0.85 / 0.97 | 0 |
| Qwen3-14B BF16 | iterative_text | 0.0/12 | 0.000 | 4.90 / 7.91 | 6 |
| Qwen3.5-122B-A10B 5-bit | basic_rag | 1.0/12 | 0.125 | 1.68 / 5.65 | 0 |
| Qwen3.5-122B-A10B 5-bit | iterative_text | 1.0/12 | 0.190 | 8.04 / 13.36 | 8 |
