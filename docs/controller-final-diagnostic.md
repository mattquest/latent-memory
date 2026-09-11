# Saved-evidence final-answer diagnostic

This is a posthoc development diagnostic, declared after inspecting the limited
48-condition controller-scaling pilot and before running this diagnostic. It
preserves every pilot result. It does not start the held-out test or rerun search.

## Trigger and scope

Several controller traces contain useful new supporting documents and plausible
answers, followed by a separate final call that returns UNKNOWN or another SEARCH
command. The final call does not receive the controller's transient reasoning or
answer payload. The common system message describes both controller actions and
final answers. This motivates testing phase confusion rather than assuming that
poor final scores show no useful retrieval.

Use all twelve development questions from each of the four pilot jobs, including
failures. Reconstruct the exact ordered, individually truncated document tokens
from the pinned corpus and verify the saved evidence hash and token counts.
Each of the 48 saved evidence states gets two fresh final generations:

1. Original system and final prompt, greedy, thinking disabled, 48 output tokens.
   Its output token IDs must exactly reproduce the original saved final answer.
2. Final-only system message with search-command instructions removed; the same
   question, document tokens, final suffix, decoder and output budget.

That is 96 short final generations, with no new retrieval or controller calls.
Run one pinned model at a time under the same 32/40 GiB guards as the pilot.
Inputs, scripts, prompt text, reconstructed token hashes, outputs, timing, token
work and memory are recorded. Answers and support annotations enter scoring only.
If original replay does not reproduce, preserve that failure and withhold paired
headline conclusions; do not silently count a changed baseline as equivalent.

## Interpretation

The intervention tests the final system prompt on fixed retrieved evidence.
It does not retain controller thought state, improve search, increase reasoning
budgets, or estimate a complete revised pipeline. Report latency as final-stage
latency only; do not substitute it for a measured end-to-end rerun. Different
saved evidence states can overlap and are not independent questions.

The diagnostic is exploratory and uses development questions already inspected.
Even a gain would require a separately frozen end-to-end confirmation. The larger
96-question test and cache-relay comparisons remain deferred. For a future native
cache comparison, use a common neutral prologue and phase-specific suffixes;
changing the cached system prefix at the final step would require rebuilding it
and charging that work.
