"""Experiment runners E1, E2, E4, E5, E6 (scaffolded).

These need the MLXBackend and real benchmarks; they are specified
here so the morning work is "fill in the data source", not "design
the experiment". Each runner's contract is documented; each raises
NotImplementedError until the backend lands.

E3 (the audit) is fully implemented in audit.py and runs here.
"""

from __future__ import annotations


def _need_mlx():
    raise NotImplementedError(
        "Requires MLXBackend on the M5 Max (engine/backends_mlx.py). "
        "The runner logic below is specified; wire the backend first."
    )


def run_e1(backend=None, dataset=None, n_hops: int = 3):
    """E1 -- text vs latent at equal hops.

    Expect PARITY (the Cheng audit predicts it). Parity at lower
    latency is a win. A latent loss > 2 points means the relay
    mechanics are broken -- check position re-indexing and bridge
    recompute first.

    Contract: returns dict with keys
      text_accuracy, latent_accuracy, text_p50_ms, latent_p50_ms,
      tokens_to_host_text, tokens_to_host_latent
    """
    _need_mlx()


def run_e2(backend=None, dataset=None, hop_budgets=(1, 3, 5, 10, 20)):
    """E2 -- hop-budget curve. THE thesis experiment.

    Accuracy vs hops for both arms. Latent must keep climbing past
    where text becomes too slow/expensive. KILL if the latent curve
    is flat beyond 5 hops.

    Contract: returns {hops: {text_acc, latent_acc, text_ms, latent_ms}}
    """
    _need_mlx()


def run_e4(backend=None, dataset=None, ratios=(0.0, 0.10, 0.20, 1.0)):
    """E4 -- bridge recompute ratio sweep. Find the knee.

    0% = pure concat (no cross-chunk attention); 100% = full recompute
    (equivalent to normal prefill). Expect the knee near 0.15-0.20
    per CacheBlend. If 0% ~= 100%, cross-chunk attention doesn't
    matter for your workload -- simplify the design.

    Contract: returns {ratio: accuracy}
    """
    _need_mlx()


def run_e5(backend=None, dataset=None):
    """E5 -- KV precision: bf16 vs int8 vs 4-bit relays.

    Determines the cold-store footprint at 10M tokens and whether
    the final synthesizer hop needs bf16. Uses relay.quant plus the
    int4 path (to be implemented in relay/quant.py).

    Contract: returns {precision: (accuracy, bytes_per_token)}
    """
    _need_mlx()


def run_e6(backend=None, beam_path: str | None = None):
    """E6 -- update handling on BEAM knowledge-update + contradiction
    categories, in isolation.

    This is where a latent store is most likely to blur facts. If it
    does, keep the update/contradiction logic in TEXT (the remember
    path already does the check with a reader hop; E6 measures
    whether the latent read path respects supersession).

    Needs: BEAM benchmark checkout at beam_path.
    Contract: returns {category: accuracy} for both arms.
    """
    _need_mlx()
