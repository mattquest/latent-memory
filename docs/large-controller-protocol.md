# Large local controller development screen

This screen asks whether a substantially different, larger local model can
retrieve useful additional evidence and answer questions that basic RAG misses.
It compares fresh Qwen3-14B BF16 and Qwen3.5-122B-A10B 5-bit runs using the same
two text-based retrieval arms. It is a follow-up development experiment, not
a test of latent-memory accuracy or a causal estimate of parameter scaling.

## Model choice and provenance

| Role | Checkpoint | Pinned revision | Representation |
| --- | --- | --- | --- |
| Fresh smaller reference | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | Official BF16, dense |
| Larger practical local candidate | `mlx-community/Qwen3.5-122B-A10B-5bit` | `958b33bf6418f8c462a5cbec59366ef763cc3069` | Community affine 5-bit, group size 64, hybrid MoE |

Qwen describes the larger language model as 122 billion total parameters with
10 billion activated per token. Its layers mix Gated DeltaNet, attention and
expert routing. The activated count concerns computation, not the resident
weight footprint. This screen uses text only. The upstream base-model revision
observed during selection was `dc4d348443bc740c68e2d77492492c11606384d5`;
the executable artifact is the separately pinned community conversion above.
The community card does not establish that its conversion used that exact
upstream revision. [Official model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B),
[MLX conversion card](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-5bit).

The selected checkpoint contains 84,857,317,565 bytes of safetensors weights
(79.03 GiB). Metadata and tokenizer files add a small amount. These are file
sizes, not measured peak memory. The 4-bit and 6-bit alternatives contain
64.81 and 93.24 GiB of safetensors respectively. Five bits preserves more weight
precision than four while leaving materially more working memory than six.
These sizes were obtained from publisher file metadata on September 11, 2026.
[5-bit metadata](https://huggingface.co/api/models/mlx-community/Qwen3.5-122B-A10B-5bit?blobs=true).

The larger-total-parameter alternative considered was
`mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ`, revision
`de2cd795c531d2e6bdc386baf1b153d887623c05`. Its safetensors occupy 95.79 GiB;
the 4-bit DWQ version occupies 123.16 GiB. The official model has 235 billion
total and 22 billion activated parameters. Three-bit weights would leave
little room beneath this Mac's 107.52 GiB recommended GPU working set for
caches and temporary allocations. Its older checkpoint and more aggressive
quantization also prevent assuming that its larger parameter count gives
better answers. It is not part of this experiment.
[Official 235B card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507),
[3-bit conversion](https://huggingface.co/mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ).

Each downloaded file must match the pinned publisher's LFS SHA-256 or Git blob
hash, with local SHA-256 and byte counts preserved in the download manifest.
Hash verification establishes artifact identity; it does not establish that
quantization preserves BF16 accuracy. Do not execute model-supplied custom
Python. The installed MLX-LM text-only Qwen3.5 implementation supports the
architecture. Its hybrid recurrent and attention cache cannot be substituted
into the old Qwen3 KV-relay implementation without separate engineering and
validation. [MLX-LM implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py).

## Fixed cohort and comparisons

Run all twelve questions in `data/controller-scaling/dev.jsonl`, with four
questions each requiring two, three and four annotated hops. Each model runs
`basic_rag` and `iterative_text`: **12 questions × 2 models × 2 arms = 48
conditions**, or twelve observations per model/arm. No question is dropped,
replaced or promoted into evaluation based on its earlier or current result.
Failures and resource stops remain visible; a partial run is not a complete
comparison. The separate 96-question test split remains unused by this screen.

The development-file SHA-256 is
`17338e99e8d958d8e7950678f834a2b815c90dc6d7a30ddeee788a29f7cc92cc`.
The shared 21,100-paragraph corpus SHA-256 is
`4e8ad63e12ab37e7fabec0158241d890a96171e006d226a7f22b6d00d594ce4a`.
This is the same development cohort already inspected in the smaller
controller screen. It cannot be described as fresh or held out. Its public
source data may also have appeared in model pretraining.
[Data protocol](controller-scaling-data.md).

Basic RAG retrieves once and synthesizes an answer. Iterative text uses the
same initial retrieval, then may issue additional searches before a separate
final synthesis call. Both use the same corpus, retrieval implementation,
document limits and generation policy within the new screen. Native chat
templates disable thinking. Controller and final synthesis receive separate
phase-specific instructions; final synthesis does not receive search-command
instructions, generated thoughts or discarded controller answer payloads.
The official Qwen3.5 template uses `enable_thinking=False`, rather than a
`/nothink` text command. [Official usage guidance](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).

The [frozen matrix](../configs/large-controller-screen.json) sets the run seed
to 20260923. Controllers use temperature 0.7, top-p 0.8, top-k 20 and a 1.5
presence penalty. The penalty applies once to each distinct token generated
in the current call, excluding the prompt. Each decision seed is the unsigned
integer from the first eight hexadecimal SHA-256 digits of the sorted-key
JSON encoding of `[seed, question_id, "decision", round]`. Final answers use
greedy decoding with no presence penalty. Actions are capped at 32 tokens and
final answers at 48; terminal EOS is counted separately from retained output.

Both arms initially retrieve six documents. Iterative retrieval may add three
new documents per round, up to five total retrieval rounds and eighteen
documents. Each document is capped at 300 tokens, concatenated evidence at
6,144, the question at 256, and the complete native prompt plus reserved output
at 8,192. Each call builds a fresh native cache; controller calls and final
synthesis do not reuse a previous call's cache. This makes the text path a
baseline whose repeated prefill cost is explicitly measured. Each model process
loads its weights once and receives no benchmark warmup call. Here, cold query
latency means a fresh prompt and native cache, not a fresh model process.
Compilation or weight paging on an early call can affect its measured latency;
that cost stays inside the online result. Synthetic preflights run in separate
processes, and their timings are reported separately.

Run the 14B job before the 122B job, serially. Preserve source question order.
For each question, shuffle the two arms with Python's `random.Random`, seeded
from the first eight hexadecimal SHA-256 digits of
`"20260923:{question_id}:arm-order"`. This gives both models the same arm order
on each question. The manifest preserves the exact resulting condition order.
The per-condition runtime allowance is 300 seconds and each model job has a
2,400-second allowance; checks occur between calls/conditions and an outer
process guard supplies a 2,700-second wall-clock termination bound. Synthetic
preflights have a separate 600-second process bound. Preserve synthetic
preflight outputs and every executed source snapshot before the development
run. Never silently replace a completed condition following a prompt or parser
adjustment.

## Measures and interpretation

Report exact match, token F1, cold end-to-end query median/p95, controller and
answer tokens, full prompt/prefill tokens, retrieval time, model-load time and
peak memory. Model loading and offline index preparation stay separate from
query latency. Count invalid and repeated searches, actual rounds, output-cap
hits and literal command leakage into final answers. Preserve every prompt,
output token ID, raw answer, retrieval trace, seed and condition identity.
Report runtime and token work as measured costs; do not invent dollar or
energy estimates. Different tokenizers make equal token counts an imperfect
cross-model measure of work and may retain different text under equal caps.

Use question-paired comparisons for each model's iterative versus basic RAG
result and each arm's larger versus smaller model result. Show both recovered
basic-RAG failures and cases made worse, with the common denominator of twelve.
Report newly found annotated supporting document IDs and coverage by step,
keeping those labels evaluation-only. A support-ID match does not prove the
needed relation appears in the source paragraph, survives truncation, or is
used in the answer. Keep the previously documented annotation limitations and
all affected cases visible. Early-stopped per-step groups describe trajectories;
they are not randomized accuracy-versus-round-budget comparisons.
[Qualitative annotation audit](controller-scaling-qualitative-audit.md).

This comparison changes model generation, architecture, tokenizer, pretraining,
post-training and precision together. It evaluates these two usable local
systems under a shared protocol, not size alone. Both are rerun under the new
prompts; results cannot be pooled with the earlier 8B/14B controller screen or
its post hoc final-synthesis diagnostic. No latent cache arm, independent KV
relay, trained compression, persistent-cache amortization or production cost
claim is tested here. An encouraging development result would justify freezing
a broader comparison before using the untouched test split.

## Local execution limits

Run one model at a time on the 128 GiB Mac, at normal priority, with no
administrator or system-memory changes. Require approximately 96 GiB available
RAM before loading the 122B checkpoint. This is a planning threshold; the
preflight and observed peak determine whether the configuration fits.

The process/MLX allocation ceilings are 88 GiB for 122B and 40 GiB for 14B.
Guards sample every five seconds and terminate below 12 GiB available RAM,
below 40 GiB free disk, on severe reported throttling or at the frozen runtime
limit. Previously authorized battery operation retains the 20% discharging
floor. Stop and record a resource failure if a guard trips; do not raise the
ceiling merely to finish a condition. These sampled limits do not guarantee
that a transient allocation stays below the threshold.

Begin with bounded synthetic mechanics checks and a short context. Preserve
resource samples, process exits, model loading time, installed library versions,
checkpoint receipts and executed source. The final report must validate the
complete 48-condition matrix against these inputs and archive raw data and
source with both packed and unpacked hashes.
