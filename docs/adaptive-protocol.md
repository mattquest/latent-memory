# Adaptive RAG comparison, September 11, 2026

This extension follows the user's request to test difficult questions that basic
RAG misses, and compare retrieval enhancements on both accuracy and latency.
It is specified before evaluating the extension's development or test questions.
The original fixed-evidence E1–E6 matrix remains a separate experiment.

## Question and corpus selection

Use the official MuSiQue answerable development archive already downloaded for
the original run. Construct one global corpus from all paragraphs in that split,
deduplicated by exact title and text. Index only document IDs, titles, and text;
question decompositions, answer labels, and support annotations are evaluation
metadata and cannot enter retrieval or model prompts.

Exclude the 128 question IDs used in the original matrix. Select six development
questions (two each with two, three, and four hops) and 96 separate test questions
(32 per hop count), deterministically by seed/hash within each structural group.
Do not select test questions based on any model's answer. The data manifest
records exact selected IDs, corpus construction, source revision, licenses,
checksums, and support-to-corpus mappings. These are new local splits of a public
development dataset, not the publisher's hidden test set.

## Conditions

Every arm uses the same official BF16 Qwen3-8B checkpoint and global BM25 corpus.
Basic RAG and the four iterative arms begin with the same six highest-ranked
paragraphs. Query expansion and reranking can change which six paragraphs enter
their final answer prompt. Each single-round baseline receives up to six unique paragraphs, enough to
contain all four supports of a four-hop question. Each subsequent iterative
retrieval can add up to three new paragraphs. With five total retrieval rounds,
an iterative arm can see at most 18 unique paragraphs. Per-document and total
context caps remain 300 and 6,144 tokens respectively.

| Arm | Retrieval and state handling |
| --- | --- |
| Basic RAG | One question-based retrieval, followed by an answer |
| Query expansion | One short model-generated rewrite from the question, fused original/rewrite rankings, then an answer |
| Reranked RAG | BM25 top 20, official Qwen3-Reranker-0.6B relevance scoring, retain six, then the same Qwen3-8B answer; requires successful local model preflight |
| Iterative text | Short SEARCH/ANSWER decisions using all accumulated original evidence; full text is prefilled again for each decision |
| Incremental KV, BF16 | The same iterative decision protocol, with exact causal append of new evidence to the retained prefix |
| Independent-document relay, BF16 | The same decision protocol, with separately prefilled document caches and 20% boundary recomputation |
| Independent-document relay, int8 | The relay condition with packed cache quantization between retrieval rounds |

Iterative arms can make at most five retrieval rounds. They share the same short
action format, action-token cap, final-answer prompt, and answer-token cap. An
ANSWER decision stops retrieval and triggers the common final-answer decode.
SEARCH decisions contain actual model-generated queries based on current
evidence. Neither gold chains nor benchmark answers guide those queries.
Duplicate evidence is not added again. Exhaustion and malformed actions are
recorded explicitly under a fixed fallback/stop policy.

Full-text and incremental KV paths construct the same sequence of evidence
tokens for a given retrieval trace. Temporary decision prompts and generated
actions do not enter the retained evidence prefix. Different numeric execution
paths can still cause BF16 rounding differences; a development mechanics check
must inspect those separately from downstream test accuracy.

The incremental KV arm is a strong ordinary prefix-caching baseline. It prevents
attributing a generic caching improvement to independent-document latent relay.
All iterative arms decode short queries: this extension is adaptive retrieval
with cache reuse, not the proposed trained zero-text latent planner.

The small reranker is the auxiliary model named in the original guiding plan.
Its separate checkpoint, prompts, token counts, model-load cost, and online
reranking latency must be recorded. Both models together remain under the same
32 GiB process/allocation limits. If local compatibility or validation fails,
record that failure and do not silently substitute a different system.

## Costs, scoring, and success criteria

Primary latency is synchronized end-to-end query time, including question
tokenization, retrieval, expansion/planning, on-demand document-cache creation,
assembly/recomputation, and final generation. Model loading and building the
shared corpus index are reported separately. Keep caches bounded per question;
do not assume the whole global corpus has been prefilled. Component timings,
decoded tokens, actual rounds, evidence IDs, raw actions, and outputs are saved.

Use normalized exact match and token F1 against official answers and aliases.
Report all 96 test questions, structural hop-count groups, and a clearly labeled
conditional slice where basic RAG's exact match is zero. For that slice report
recovery counts, denominator, and accuracy/latency of every arm. Also report
regressions on questions basic RAG answers correctly. The conditional slice is a
diagnostic, not an unbiased estimate of overall superiority.

Pair comparisons by question, show uncertainty for overall accuracy differences,
and plot measured accuracy against end-to-end latency. Compare the relay against
the strongest text and incremental-cache baselines, not only basic RAG. More
retrieval and more evidence are explicit additional compute. An improvement in
accuracy alone does not establish a latency improvement; lower latency with an
accuracy loss does not establish dominance.

The development set can expose runtime/prompt-format defects before freezing
the final source and test manifest. Record every development revision. Do not
change generation settings after inspecting final-test outcomes. Runs remain
serial under the same laptop resource guard as the original matrix.

## Separate context-pressure diagnostic

Replay the iterative-text test retrieval traces under a 1,536-token working
evidence budget. Use all test traces, with no selection based on answers. Compare
relevance-retained original text, native independent-document BF16/int8 caches
over the same retained document IDs, and a rolling text summary plus recent
evidence. Summary tokens count against the same evidence budget. Gold support
labels never guide retention. Each model prompt must fit its stated working
evidence budget, including any summary and newly ingested evidence.

Save retention decisions, token budgets, summary text, outputs, and all cache,
selection, compaction, and final-answer costs. The source retrieval controller's
cost is separate; any sum with replay costs is labeled explicitly. Report all
questions and a separately labeled subset whose cumulative retrieved evidence
exceeds the working budget, without treating that subset as a random population.

This is a matched-trace retention/interference ablation. Its search queries came
from the original, less constrained controller, so it does not measure the
end-to-end behavior of an adaptive agent under pressure. MuSiQue's small support
sets may still fit within 1,536 tokens even when cumulative retrieved evidence
does not. The test therefore cannot establish that necessary information exceeds
the context limit, or validate a trained latent compressor that is not present.

## Relation to existing work

Interleaving retrieval with newly derived information addresses the multi-step
retrieval problem studied by [IRCoT (Trivedi et al., ACL 2023)](https://aclanthology.org/2023.acl-long.557/).
The local SEARCH/ANSWER controller is a bounded implementation of that general
class of approach; it does not reproduce IRCoT's full prompts, backbone, or
published results. Likewise, it does not implement [FLARE's confidence-triggered
sentence regeneration](https://arxiv.org/abs/2305.06983).
The [MuSiQue authors' repository](https://github.com/StonyBrookNLP/musique)
documents the dataset and CC BY 4.0 license. Named commercial memory systems or
published leaderboard scores are not measured local baselines in this extension.

## Prepared corpus receipt

The global corpus contains 21,100 distinct title/text pairs and 2,518,129 Qwen
tokens. Deduplication removed 27,215 repeated paragraph occurrences; 1,090 titles
still have multiple distinct text variants, which remain separate documents.
All 6,404 supporting-paragraph occurrences in the source split map to corpus
content IDs. This is retrieval over the supplied public development paragraphs,
not the whole Wikipedia. Disjoint question IDs do not make their component
facts or source articles disjoint.

Preparation used seed `20260913` and excluded both IDs and exact question text
from the original 128-question run. The corpus SHA-256 is
`4e8ad63e12ab37e7fabec0158241d890a96171e006d226a7f22b6d00d594ce4a`.
[`scripts/prepare_adaptive_musique.py`](../scripts/prepare_adaptive_musique.py)
and its manifest record the byte-identical reproduction check. Inputs are
`data/adaptive_musique_{corpus,dev,test}.jsonl`; the separate provenance ledger
does not enter the retriever.
