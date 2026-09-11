# Controller size and reasoning budget experiment

This follow-up tests whether model capacity and a bounded reasoning controller
enable useful multi-step retrieval. It follows the September 11 initial study;
its new questions and predictions must remain separate from that study.

## Motivation and comparison

The relevant claim is cheaper discovery of diverse evidence through successive
retrievals. It does not require the final supporting evidence to exceed context.
The earlier controller usually repeated a query and stopped, preventing a useful
test of long retrieval trajectories. This experiment asks whether a stronger
controller removes that limitation. It does not train a latent query head or
compressor, and independent-document KV remains an approximate representation.

The four configurations form a size-by-controller comparison:

| Configuration | Weights | Maximum reasoning tokens per decision | Maximum action tokens |
| --- | --- | ---: | ---: |
| Fresh short-controller control | Qwen3-8B BF16 | 0 | 32 |
| Smaller model with reasoning | Qwen3-8B BF16 | 256 initially | 32 |
| Larger model, short controller | Qwen3-14B BF16 | 0 | 32 |
| Larger model with reasoning | Qwen3-14B BF16 | 256 initially | 32 |

The official 8B revision is `b968826d9c46dd6066d109eabc6255188de91218`.
The official 14B revision is `40c069824f4251a91eefaf281ebe4c544efd3e18`.
Their tokenizer JSON bytes match. The 14B download includes verified official
SHA-256 weight hashes, small-file Git blob hashes, license, and model card.

The pinned model card explicitly warns against greedy decoding in thinking
mode because it can degrade performance and cause repetition. All four new
configurations therefore use a common controller sampler: temperature 0.6,
top-p 0.95, top-k 20. Seeds are deterministically derived from the fixed run
seed, question ID, controller event, and round, with the same recipe across
models and arms. Final answers remain nonthinking, greedy, and capped at 48
tokens. The new short-controller control is a fresh sampled-controller
measurement, not a relabeling of the earlier greedy experiment.

Thinking begins in the model's native open think block. Its output is decoded
until the closing delimiter, EOS, or the reasoning cap. A reached cap causes
explicit, recorded closure before the separate action phase. Reasoning and
action continue through the same transient causal cache. Reasoning text and
actions are never silently inserted into the reusable evidence snapshot.
Generated, forced, reasoning, and action tokens are separately counted.
No cross-model KV transfer is attempted.

## Retrieval conditions

The first screening pass runs iterative text only. A subsequent cache comparison,
if justified by that screen, adds basic RAG, ordinary incremental BF16 prefix
reuse, and independent-document BF16 relay with 20% boundary recompute.
The shared BM25 index contains 21,100 public paragraphs, approximately 2.52
million tokenizer tokens. All four arms start with the same six paragraphs.
Iterative arms may add three new paragraphs per search, for at most five
retrieval rounds and eighteen documents. Paragraphs are capped at 300 tokens,
evidence at 6,144, and complete model input/output reservation at 8,192.
The index and prompts contain no answers, decomposition labels, or support IDs.

For each model, basic RAG has no controller and thus intentionally repeats
across the two controller-budget jobs. Those outputs are a consistency check,
not independent evidence. Ordinary prefix reuse is the key efficiency baseline:
avoiding redundant prefill is not uniquely a latent-relay benefit.

## Development and frozen evaluation

The [data protocol](controller-scaling-data.md) selects twelve development
questions and ninety-six disjoint test questions by structural hash sampling.
Every prior question ID and normalized question match is excluded. Both splits
are balanced across two-, three-, and four-hop questions. Public pretraining
exposure and shared articles are not ruled out by question disjointness.

Before development, run small synthetic mechanics checks with both models:
phase transitions, bounded decoding, seeded sampling, causal-prefix comparison,
snapshot immutability, token accounting, and full-reservation rejection.
No benchmark questions enter these checks.

At the user's request, the initial development pass is limited to **48 conditions**:
twelve questions times the four model/controller configurations, with iterative
text as the sole arm and a 256-token reasoning cap. It records failures and cap
hits as outcomes. This isolates controller viability before paying for more
cache comparisons. Compare exact matched questions for new support discovery,
query repetition, answer correctness and full latency. If the
controller is still prevented from issuing useful actions by truncation or a
mechanics/prompt failure, a revised development pass may adjust the common
reasoning budget or controller protocol. Preserve every earlier pass and its
executed source. Do not select or exclude questions by their outcomes.

Proceed to cache comparisons or a larger test only if the screen shows useful
controller improvement worth investigating. Development is a screen, not a
population superiority claim. Any selected follow-up must be documented before
test inference, and the full original four-configuration screening results must
remain visible, including poor configurations. If the screen is flat, report
that outcome without automatically spending the full evaluation budget.

Freeze the final source, sampler, prompt, token and round budgets, condition
matrix, model revisions, and test input hashes after development and before
any test inference. Run all retained configurations on the same test questions;
do not evaluate only the development winner. Any runtime-driven reduction in
test size must be declared before test inference and use a fixed structural
prefix, never observed accuracy. The available test has 96 questions. A full four-job/four-arm
comparison would have 1,536 conditions, but is deferred behind the development
screen and is not currently queued or promised as completed.

## Outcomes and limits

Primary measures are exact match, token F1, cold end-to-end median/p95 query
latency, and paired changes between configurations and arms. Report initial
and newly retrieved supporting IDs, support coverage by round, repeated/invalid
actions, actual round counts, output caps, and recovered basic-RAG failures.
Support labels enter these diagnostics only after inference. Document presence
does not prove that answer-bearing text survived truncation or was used.

Report all online controller reasoning, actions, retrieval, tokenization,
cache construction/recomputation and final decoding. Model loading and offline
index preparation are separate. Record peak MLX allocation and sampled process
memory. Token counts and measured runtime are cost measures; no unmeasured API
price, dollar saving, or energy consumption is inferred. Conditional per-round
groups are descriptive, not randomized accuracy-versus-round-budget curves.

The evidence caches in this experiment are built cold per question. This does
not measure ingestion-time persistent-cache amortization, trained latent
planning, a larger effective attention context, or production-wide economics.
An increase in useful retrieval and accuracy would justify testing longer
trajectories; model size alone does not establish a latent communication gain.

## Execution constraints

Model runs are serial, at normal process priority. The 8B process/MLX allocation
ceiling is 32 GiB; the 14B ceiling is 40 GiB. Guards sample every five seconds
and stop below 12 GiB available RAM, below 40 GiB free disk, on severe reported
throttling, or at each stage's runtime limit. The user explicitly authorized
battery operation; the guard still stops at 20% charge while discharging. These
are sampled termination thresholds, not guarantees about unsampled peaks.
No administrator or system memory settings are changed.

Sources: [official Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B/tree/40c069824f4251a91eefaf281ebe4c544efd3e18),
[Qwen3 technical report](https://arxiv.org/abs/2505.09388).
