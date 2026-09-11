# Matched-trace context pressure

Status: **complete**.

All source questions are retained. The secondary overflow subset is defined by cumulative source-token exposure, not correctness.

| Arm | Subset | n | EM | F1 | Replay p50 / p95 (s) | Mean source planning / search (s) |
| --- | --- | --- | --- | --- | --- | --- |
| retained_kv_bf16 | all | 96 | 1.0% | 3.2% | 1.41 / 1.82 | 2.02 / 0.01 |
| retained_kv_bf16 | trace_overflow | 9 | 0.0% | 0.0% | 1.67 / 1.86 | 3.11 / 0.01 |
| retained_kv_int8 | all | 96 | 2.1% | 3.7% | 1.44 / 1.85 | 2.02 / 0.01 |
| retained_kv_int8 | trace_overflow | 9 | 0.0% | 0.0% | 1.76 / 1.93 | 3.11 / 0.01 |
| retained_text | all | 96 | 3.1% | 4.9% | 0.53 / 1.61 | 2.02 / 0.01 |
| retained_text | trace_overflow | 9 | 0.0% | 0.0% | 0.67 / 0.79 | 3.11 / 0.01 |
| rolling_summary | all | 96 | 4.2% | 5.8% | 0.81 / 13.99 | 2.02 / 0.01 |
| rolling_summary | trace_overflow | 9 | 0.0% | 0.0% | 11.10 / 20.79 | 3.11 / 0.01 |

## Original references

These timings include original search and planning; replay timings above exclude those separately listed costs.

| Original arm | Subset | n | EM | Original end-to-end p50 / p95 (s) |
| --- | --- | --- | --- | --- |
| basic_rag | all | 96 | 1.0% | 0.39 / 1.36 |
| basic_rag | trace_overflow | 9 | 0.0% | 0.48 / 0.54 |
| iterative_text | all | 96 | 3.1% | 2.56 / 4.14 |
| iterative_text | trace_overflow | 9 | 0.0% | 3.94 / 5.69 |

## Limits

- Shared-trace retention/interference ablation; controller decisions were made in the original larger-context run.
- All questions are included; overflow is defined by source-token exposure, not answer correctness.
- Evidence budget counts source tokens, KV slots and summary wrappers. Fixed prompts/questions and output reservations are additional, explicitly logged.
- Int8 reduces stored bytes, not attention positions. No trained latent compressor or latent query controller is tested.
- Most questions may have sufficient support within the smaller window. Overflow does not prove necessary information exceeds context.
- Support-ID coverage is annotation provenance, not proof truncated text or generated notes preserve the facts.
- Basic RAG and larger-context adaptive references retain their original retrieval settings and measured end-to-end costs.
