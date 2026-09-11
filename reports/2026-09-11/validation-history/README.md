# Validation and excluded run history

These receipts are excluded from the main benchmark metrics. Archive completeness does not imply all checks passed.

Interrupted original test conditions retain their test-run role; they are not relabeled as development data.

| Source | Role | Recorded status | Files |
| --- | --- | --- | --- |
| native-validation | numerical_original_failed_threshold | failed | 3 |
| native-validation-relative | numerical_revised_relative_threshold | complete | 3 |
| official-preflight | official_checkpoint_smoke_check | complete | 3 |
| pilot-private | private_evidence_development_pilot | complete | 7 |
| development | operator_stopped_earlier_development | stopped | 7 |
| development-v2 | revised_development_matrix | complete | 24 |
| judge-pilot | same_model_rubric_judge_pilot | complete | 6 |
| adaptive-preflight | adaptive_real_weight_mechanics_preflight | complete | 5 |
| adaptive-dev | adaptive_retrieval_development_run | complete | 8 |
| context-pressure-dev | context_pressure_development_run | complete | 11 |
| original-interrupted-multihop | unused_partial_test_run_after_budget_amendment | failed_or_interrupted | 2 |
| adaptive-data-validation.json | data_reproduction_audit | document_only | 1 |
| adaptive-initial-retrieval-audit-final.json | pre_inference_retrieval_annotation_diagnostic | document_only | 1 |
| adaptive-initial-retrieval-audit.json | pre_inference_retrieval_annotation_diagnostic | document_only | 1 |
| backend-report-notes.md | historical_methods_and_numerical_notes | document_only | 1 |
| bm25_hashseed_audit.json | hashseed_reproducibility_audit | document_only | 1 |
| bm25_hashseed_ledgers.json.gz | hashseed_reproducibility_raw_ledger | document_only | 1 |
| overnight/amendment.json | operator_budget_amendment_receipt | document_only | 1 |
| reranker_prompt_audit.json | reranker_tokenizer_prompt_audit | document_only | 1 |
| reranker_prompt_audit_final.json | reranker_tokenizer_prompt_audit | document_only | 1 |

See [manifest.json](manifest.json) for source identities, original statuses, and both source/artifact SHA-256 hashes.

Each artifact is gzip of the exact original file bytes. Decompress once to recover that source file; an original .gz ledger therefore has an outer .gz.gz wrapper.

The original numerical run failed its absolute-error threshold. The relative-threshold run is a later development validation, not a prespecified benchmark criterion.
