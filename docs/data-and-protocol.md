# Data and experiment protocol

Prepared September 11, 2026. This document records the implemented data preparation and the interpretation limits for the overnight experiments. It is not an independent preregistration or a claim that every planned experiment has run. Actual execution manifests and per-example result files determine what was measured.

## Reproduction and files

Run from the repository with its local environment:

```sh
.venv/bin/python scripts/prepare_data.py --datasets synthetic --count 128 --seed 20260911
.venv/bin/python scripts/prepare_data.py --datasets musique --count 128 --seed 20260911
# HotpotQA Parquet fallback requires a Python environment with pyarrow installed.
python scripts/prepare_data.py --datasets hotpotqa --count 128 --seed 20260911
.venv/bin/python scripts/prepare_data.py --datasets beam --beam-conversations 1 2 3
```

The downloader uses HTTPS, timeouts, per-file limits, an 800 MiB default transfer budget per invocation, and atomic file replacement. It does not run downloaded code, deserialize pickle files, call paid APIs, or inspect personal files. It reads the one development member directly from the MuSiQue ZIP without extracting other files. Interrupted downloads are removed; completed source files are reused. Raw downloads and generated datasets stay under `data/` and should not be committed. `data/manifest-*.json` records requested seed, source URLs, bytes, SHA-256 digests, successful outputs, and errors. A run that cannot prepare a requested source exits nonzero instead of silently replacing it with synthetic data.

| File | Contents | Selection |
| --- | --- | --- |
| `data/synthetic_private_fact.jsonl` | 128 private registry lookups with six-digit random answers | Test seed 20260911 |
| `data/synthetic_multihop.jsonl` | 128 linked-depot questions with distractors; chain lengths 2, 3, 5, 10, 20 | Test seed 20260911 |
| `data/synthetic_updates.jsonl` | 96 explicit replacement probes and 32 unresolved-conflict probes | Test seed 20260911 |
| `data/synthetic_*_dev.jsonl` | 32 development examples per synthetic family | Separate seed 20260912 |
| `data/hotpotqa_dev_sample.jsonl` | 128 questions with all supplied distractor paragraphs | SHA-256 rank of seed and source question ID |
| `data/musique_dev_sample.jsonl` | 128 answerable development questions with all supplied paragraphs | SHA-256 rank of seed and source question ID |
| `data/beam_1m_{1,2,3}_corpus.jsonl` | Three complete public conversation histories, one message per line | Source conversation IDs 1, 2, 3 |
| `data/beam_1m_{1,2,3}_probes.jsonl` | All 20 official probes per selected conversation | 60 queries, all ten categories |

A runner can use fewer examples for a bounded run; its output must name the actual count, IDs, dataset checksum, and selection rule. The table describes prepared data, not completed inference.

## Provenance and licenses

**HotpotQA.** Yang et al., EMNLP 2018. The [official dataset homepage](https://hotpotqa.github.io/) licenses the dataset under CC BY-SA 4.0. The original CMU JSON download timed out during preparation. A first fallback to the Hugging Face rows service was rate-limited; it was stopped. The successful preparation uses the 26 MiB validation/distractor Parquet file from the publisher's [HotpotQA Hugging Face repository](https://huggingface.co/datasets/hotpotqa/hotpot_qa), pinned to revision `1908d6afbbead072334abe2965f91bd2709910ab`. Its source SHA-256 is recorded. Every sampled row retains its source ID. The existing `/opt/anaconda3/bin/python3` runtime supplied PyArrow for this preparation; no environment packages were changed. All provided contexts and supporting sentence annotations survive normalization. This is the distractor setting, not full-Wikipedia retrieval.

**MuSiQue.** Trivedi et al., TACL 2022. The [authors' repository](https://github.com/StonyBrookNLP/musique) distributes the data under CC BY 4.0. Its download script at revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb` points to the [official v1.0 archive](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view). Preparation reads the answerable development JSONL member, keeps answer aliases, source paragraph IDs, and supporting/decomposition annotations, and samples without changing supplied contexts. The archive checksum pins the actual bytes. This run does not cover unanswerable MuSiQue-Full examples or the complete development set.

**BEAM.** Tavakoli et al., [Beyond a Million Tokens](https://arxiv.org/abs/2510.27246). Data come directly from the [authors' repository](https://github.com/mohammadtavakoli78/BEAM/tree/b2da22eac88bb0874c64665f13457eb99835774a), revision `b2da22eac88bb0874c64665f13457eb99835774a`, paths `chats/1M/{1,2,3}/chat.json` and corresponding `probing_questions/probing_questions.json`. The [dataset card](https://huggingface.co/datasets/Mohammadta/BEAM) specifies CC BY-SA 4.0 for data; the repository's MIT code license does not replace that data license. Full histories are flattened in chronological order, retaining roles, source IDs, and available time anchors. Conversation plans, probing rubrics, answers, and source-evidence annotations are not part of the retrieval corpus. This selection is three named conversations, not a random sample of all 35 or a BEAM-10M evaluation.

**Synthetic.** Generated locally by `eval/datasets.py`, generator version 1. Entities, registry numbers, depot links, and revisions are fictional. No laptop documents, account information, contacts, or user conversations are used. Published benchmark contamination remains possible for public QA; synthetic nonce facts provide a separate information-transfer check.

## Runner interface and leakage boundaries

`eval.datasets.load_examples(path, limit=None, split=None)` returns normalized dictionaries. Each QA row contains `id`, `dataset`, `split`, `question`, `answers`, `evidence`, `category`, and `metadata`. An evidence item has `id`, `title`, and `text`, plus ordinary provenance such as timestamps or explicit version status when applicable. Rows may contain `supporting_context_ids` and `hop_context_ids`.

`answers`, `supporting_context_ids`, `hop_context_ids`, question decompositions, BEAM rubrics, and source-evidence annotations are **evaluation-only gold**. A normal retriever must receive only the question and corpus text/ordinary provenance. It must not select, rank, truncate around, or generate queries from gold. Do not concatenate serialized rows into prompts.

BEAM probes intentionally have empty `evidence` arrays and a relative `corpus_path`, resolved against the probes file's parent directory. This prevents repeating megabytes of history for each question. Retrieval should search the whole corresponding corpus and record selected message IDs, chunks, retrieval latency, and truncation. Gold `source_chat_ids` are retained within `metadata.source_annotation` for later diagnosis only.

Synthetic replacement examples mark the earlier record `active: false` and identify its successor. This metadata implements known correct supersession for an explicit diagnostic. It does not evaluate the quality of a learned contradiction detector. Unresolved conflicts keep both records active and expect `CONFLICT`; automatically discarding one would erase the very condition being tested.

## What each experiment can establish

| Experiment | Valid overnight claim | What it cannot establish alone |
| --- | --- | --- |
| E1, equal hops | Relative answer quality and measured costs for matched model, questions, evidence, and hop budget | Universal parity, a leaderboard win, or a learned multi-agent advantage |
| E2, hop sweep | Quality/cost as additional evidence becomes available under a fully recorded schedule | Adaptive retrieval value when gold determines the schedule; improvement caused by latent communication rather than added evidence |
| E3, cache interventions | Paired-vs-mismatched cache dependence on new private facts | Reproduction of the published paper's full tasks, seeds, statistics, or channel implementations |
| E4, recompute sweep | Effect of the implemented recomputation strategy and actual recomputed-token count | Reproduction of CacheBlend from a percentage label or boundary-token approximation |
| E5, precision sweep | Answer degradation and measured serialized cache bytes at tested precision | Quantized compute speedups if caches are dequantized before attention; 10M storage feasibility from a tiny sample |
| E6, updates/conflicts | Whether each reader respects explicit revisions and preserves unresolved contradictions | BEAM category scores when the data are synthetic, or automatic write-time supersession accuracy |

Any use of `hop_context_ids` must label the run **oracle evidence schedule / controlled diagnostic**. Gold ordering can make the answer appear only at later hops by construction. A rising curve under this schedule verifies access to newly supplied information; it is not evidence that a planner or latent verifier discovered better retrieval steps. A run that repeatedly reads an unchanged evidence set should record that fact; equal hop numbers need not imply equal useful reasoning.

For public QA, selecting only known supporting documents is likewise an **oracle evidence diagnostic**. Searching all supplied distractor paragraphs is a bounded supplied-context evaluation. For BEAM, searching a complete selected history is a genuine corpus-retrieval experiment, but a 60-query conversation subset still is not the full published benchmark.

## Costs, scoring, and uncertainty

Use the same checkpoint revision, tokenizer, prompt template, temperature, output cap, and selected evidence across compared arms. Record warm-up, weight loading, offline cache construction, online retrieval, recomputation, decoding, and total time separately. Include synchronous device completion in timed regions. Distinguish cold end-to-end latency from warm cache latency; precomputation is a cost even when amortized. Randomize or alternate arm order when possible to reduce warm-up and thermal bias.

Report actual generated token counts, final tokens returned to the host, internal decoded tokens, input/prefill tokens, selected context tokens, cache bytes, example count, and p50/p95 latency. Character lengths are not token counts. Passing an in-process tensor reference can be cheap while the production, position correction, copying, recomputation, and consumption of that tensor remain expensive.

Use normalized exact match and token F1 for short public-QA answers, maximizing over released aliases. Preserve raw answers for review. A permissive substring match alone can reward answers that list contradictory possibilities. BEAM questions include summaries, abstention, ordering, and compliance; its released rubric belongs in a separately recorded judge or structured human review. Exact match/F1 against BEAM reference prose is a diagnostic proxy, not an official BEAM score. No paid/cloud judge has been used by data preparation.

Report paired differences and intervals when sample size permits; present small runs as exploratory. A non-significant difference does not establish equivalence. The guiding document's two-point threshold cannot be established reliably from a handful of examples. Retain failed and superseded runs with reason labels, and avoid tuning on the final evaluation seed or selecting only successful examples after seeing answers.

For E3, mismatch without fixed points, avoid donor-answer collisions, and keep model, receiver prompt, cache shape, dtype, and receiver decoding fixed. A different-length donor can confound content with position/geometry. Zeroed and random controls supplement the natural donor intervention. Moment-matched randomization must state exactly which dimensions and moments are preserved. Confirm gold content is absent from receiver text and any still-live cache buffers. Add no-cache and fully informed text controls where practical.

## Check against the guiding references

The core cited papers exist, but their published scope is narrower than several design assumptions.

- **Cheng, Das, and Ramnath, [causal audit](https://arxiv.org/abs/2608.04893v2), August 2026.** The paper supports testing correct, deranged, zeroed, and moment-matched caches, with stronger pairing effects when sender-private facts are necessary. It does not predict general text-versus-latent parity or turn a two-point small-sample deficit into proof of broken mechanics. The paper describes released artifacts, but no working audit-repository link was found in the inspected arXiv text; our harness is an independent implementation, not a claimed execution of their released code.
- **Zou et al., [LatentMAS](https://arxiv.org/abs/2511.20639), [official code](https://github.com/Gen-Verse/LatentMAS).** The method combines latent thought generation, alignment, and cache relay. Reusing document-prefill caches without its latent rollout is not a complete reproduction. Their repository recommends its standard Hugging Face path for reproducing published results and warns that the vLLM path can differ numerically.
- **Lu et al., [TurboRAG](https://arxiv.org/abs/2410.07590), [EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.334/).** It supports precomputing document KV and carefully handling positional embeddings and attention masks. Its accuracy procedure also involves fine-tuning. Position-label changes alone do not establish that RoPE-rotated keys remain correct.
- **Yao et al., [CacheBlend](https://arxiv.org/abs/2405.16444), [EuroSys paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf).** Selective recomputation is driven by token-level KV deviations across layers. The paper reports small recomputation fractions in its tested workloads. An arbitrary boundary-only 15% policy is a different method and its actual compute may exceed the fraction of cache entries replaced.
- **Yang et al., [CacheClip](https://arxiv.org/abs/2510.10129v2).** Auxiliary-model token selection, shared prefixes, and grouping matter. The 20% setting is an experimental configuration, not a universal guarantee for Qwen3 on this laptop.
- The cited [latent-communication survey](https://arxiv.org/abs/2606.05711), [Latent Memory compressor](https://arxiv.org/abs/2606.10572), [QKVShare](https://arxiv.org/abs/2605.03884), [LCGuard](https://arxiv.org/abs/2605.22786), and [KV-cache integrity paper](https://arxiv.org/abs/2606.28958) resolve to the stated topics. Their existence does not mean their implementations, training, security claims, or benchmark results have been reproduced here.

The guiding document's historical Hindsight value, 64.1% on BEAM-10M, appears in the [vendor's April 2, 2026 report](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota). Mem0's [published evaluation repository](https://github.com/mem0ai/memory-benchmarks) separates platform/OSS configurations and graded scores/pass rates. These are external reference reports, not local baselines reproduced tonight. They should not share a ranking table with a locally scored BEAM subset or short-answer proxy without clearly separating protocol, model, scale, judge, and cost definitions.

The model memory table also needs one arithmetic correction: the official [Qwen3-8B configuration](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) has 36 layers and [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) has 40; both have eight KV heads and head dimension 128. At BF16, their raw KV payloads are 147,456 and 163,840 bytes/token respectively, a 1.11x ratio rather than 1.7x. Allocator overhead and quantization metadata are additional.
