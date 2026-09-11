# Fresh data for the controller scaling experiment

The new local development and test questions are frozen before controller development or test inference. The development set has 12 questions, four each with two, three, and four structural hops. The test set has 96 questions, 32 per hop count. Both use the same 21,100-document global corpus as the previous adaptive experiment, copied byte-for-byte. Model choice and controller behavior therefore cannot change corpus membership.

This is a test of model capacity and reasoning budget for productive retrieval over the supplied MuSiQue paragraphs. It does not itself establish a long-chain benchmark or a need to fit more necessary evidence than the context allows. The complete annotated support sets contain at most 794 tokens in development and 889 tokens in test under the documented Qwen3 document wrapper. Finding and combining those supports can still require intermediate searches. Support sizes are recorded separately and were not used to select questions or construct prompts.

## Selection and prior exposure

Preparation uses the official MuSiQue v1.0 answerable development member: 2,417 source questions. Within each structural hop group, it sorts eligible questions by SHA-256 of `20260921:source_id`, takes four development questions, then 32 test questions. Selected question strings must be distinct after Unicode NFKC normalization, case folding, and whitespace normalization. Punctuation and articles remain intact. Source order, answers, generated outputs, accuracy, retrieval coverage, and token sizes do not enter the ranking.

Exclusions include all 128 original MuSiQue questions, six previous adaptive development questions, and 96 previous adaptive test questions. The script also reads every explicit pilot and cohort identity in the preserved [development overlap audit](../reports/2026-09-11/development-overlap-audit.json). It uses only ID/question pairs from that audit; scores, predictions, and success flags do not enter selection. Each prior normalized file is checked against both its pinned SHA-256 and the prior audit's data receipt.

The combined ledger has 1,399 source-labelled identity records, representing 817 distinct previously seen IDs across all datasets. Matching IDs or normalized questions excludes 231 MuSiQue source rows: the 230 prior MuSiQue IDs plus one additional question with an already-seen normalized question string. The remaining candidate counts are 1,151 two-hop, 685 three-hop, and 350 four-hop questions. New development and test IDs and normalized question strings are disjoint from one another and from the full prior ledger. The original pilot overlaps are therefore excluded along with their parent cohorts; no questions are removed based on model performance.

Question disjointness does not imply disjoint component facts, articles, templates, or freedom from public pretraining exposure. These are local subsets of a public development split, not the publisher's hidden test set. Exclusion coverage is limited to preserved input and identity ledgers; it cannot establish absence of unrecorded human inspection.

## Frozen artifacts

The local package is `data/controller-scaling/`. Its `manifest.json` records all output hashes, source receipts, exclusion rules, ordered membership, and checks. Its SHA-256 is `1846439dc9ece71b06aba5b231a3f8c468b228204ad165d9d9093fd8607dea1b`.

| Artifact | Records | SHA-256 |
| --- | ---: | --- |
| `dev.jsonl` | 12 | `17338e99e8d958d8e7950678f834a2b815c90dc6d7a30ddeee788a29f7cc92cc` |
| `test.jsonl` | 96 | `8d8f3e6188079022fbf429821174a9536eba650330e58f86754e23cdddbb5119` |
| `corpus.jsonl` | 21,100 | `4e8ad63e12ab37e7fabec0158241d890a96171e006d226a7f22b6d00d594ce4a` |
| `excluded-identities.jsonl` | 1,399 | `ba2449f260f9e0b441a21bbd1e7561a12082a5d9d7cb3956ab1145b13c8e158d` |
| `excluded-source-questions.jsonl` | 231 | `b7f9b766d28d188c06ac53e2bd5bf161f40e633c5789b7b128a9b24bcf49a467` |
| `support-sizes.scoring-only.jsonl` | 108 | `c20a118ba2daab4cf707a342b1c5a9abba36c408dc0ca18c9efb9cda8ec947a3` |

The corpus is 12,935,527 bytes and contains exactly `id`, `title`, and `text` per record. Rebuilding its content-hash IDs and paragraph content from the official archive gives the same records in the same order. The complete original bytes are then copied into the new package. No answers, support flags, decomposed questions, source question IDs, or paragraph-source indices enter retrievable records.

Normalized question files retain answers, aliases, structural hop counts, and supporting corpus IDs for scoring. They contain empty local evidence lists and reference the separate corpus. A controller may read only the question and retrieved corpus content. `corpus-provenance.jsonl` stores original paragraph occurrence mappings separately. The support-size file records complete and 300-token-capped support document fragments using `\n<document>\n{title}\n{text}\n</document>\n`; it excludes system, question, chat, controller, and output overhead. The tokenizer JSON SHA-256 is `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`. Median complete support size is 402 tokens for development and 394.5 for test. These are annotation diagnostics, not observed model results.

The package also preserves deterministic gzip copies of the exact official development member, all three prior MuSiQue normalized inputs, and the previous overlap audit, plus the source files needed for preparation. Every compressed source receipt records both packed and original byte counts and SHA-256 hashes. The full original ZIP remains in `data/raw/musique.zip`; its pinned hash is `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`. The selected member hash is `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`.

Data attribution and modifications are recorded in `DATA_LICENSE.md`: Trivedi et al. (2022), [MuSiQue](https://github.com/StonyBrookNLP/musique), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The preserved official download-script revision is `922ac98f19a201998dbdae6d7f2887a5258dbdeb`.

## Reproduction and validation

From the repository root, using the existing local source download and tokenizer:

```sh
.venv/bin/python scripts/prepare_controller_scaling_data.py \
  --output-dir data/controller-scaling \
  --seed 20260921 --dev-per-hop 4 --test-per-hop 32 \
  --tokenizer-json models/Qwen3-8B/tokenizer.json
```

The script refuses to overwrite an existing output directory. To check reproduction, choose a fresh output directory; the path is not embedded in output data or receipts. This operation uses CPU tokenization and does not load model weights, call a model, or alter prior inputs.

Eleven focused CPU tests pass. A second independent preparation into `runs/controller-scaling-data-reproduction/` produced all 17 package files byte-for-byte identically. Independent validation checked all 16 output receipts, compressed and original source hashes, schema loading, normalized question/ID separation, corpus field isolation, complete support mappings, and absence of support-size fields from question input records. The local validation receipt is `runs/controller-scaling-data-validation.json`.

The final experiment matrix must be declared after development and before test inference. Development observations may guide controller choices. The test membership remains frozen, and test outcomes must not guide changes to the controller or selection.
