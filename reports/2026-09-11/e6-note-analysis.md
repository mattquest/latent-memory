# Saved E6 note diagnostic

Posthoc and descriptive; no causal intervention or generation change.

The completed source contains 1024 conditions; this note analysis covers 256 text conditions on 128 questions.

| Stratum | Policy | Conditions | ID retained / omitted | Current number retained | UNKNOWN | Correct | Final note tokens min–max | Notes at cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contradiction_resolution | all | 32 | 32 / 0 | Not applicable | 0 | 32 | 55–55 | 0 |
| contradiction_resolution | both_policies | 64 | 64 / 0 | Not applicable | 0 | 64 | 55–55 | 0 |
| contradiction_resolution | current | 32 | 32 / 0 | Not applicable | 0 | 32 | 55–55 | 0 |
| knowledge_update | all | 96 | 0 / 96 | 96/96 | 96 | 0 | 16–16 | 0 |
| knowledge_update | both_policies | 192 | 0 / 192 | 192/192 | 192 | 0 | 16–16 | 0 |
| knowledge_update | current | 96 | 0 / 96 | 96/96 | 96 | 0 | 16–16 | 0 |

Configured relay cap: 192 tokens. Conflict-value retention and per-arm accuracy context are in the JSON.

[Analysis and source hashes](e6-note-analysis.json) · [Every text condition and note](e6-note-examples.jsonl)

- Record IDs and values are exact case-insensitive lexical matches with alphanumeric boundaries; presence does not establish semantic association.
- The final note is the last generated relay_text event, skipping later empty-hop trace records.
- Relay stop reasons were not saved. A note shorter than its configured cap did not reach that cap; no more specific stop reason is inferred.
- All/current conditions for the same question are repeated policy conditions, not independent examples.
- Omitted IDs and UNKNOWN outputs co-occur; this saved-output analysis does not prove that omission alone caused failure.
- No notes were repaired, no prompts were tuned, and no model calls were made in this analysis.
- These results describe this particular evidence-note baseline; they do not establish a fundamental latent-memory advantage for updates.
