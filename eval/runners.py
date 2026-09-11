"""Python entry points to real-model fixed-evidence diagnostics.

Every run requires native MLX, a normalized JSONL dataset and an artifact
output directory. These are not the unfinished adaptive trained-agent pipeline.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from .real_experiments import ExperimentConfig, run_suite


def _run(experiment, backend, dataset, output_dir, **overrides):
    if backend is None:
        raise ValueError("Pass a native MLXBackend; mock scores are not supported")
    if dataset is None or output_dir is None:
        raise ValueError("dataset JSONL path and output_dir are required")
    config = ExperimentConfig(dataset=str(Path(dataset)), output_dir=str(Path(output_dir)),
                              experiments=(experiment,))
    return run_suite(backend, replace(config, **overrides))


def run_e1(backend=None, dataset=None, n_hops=3, *, output_dir=None, **options):
    """Text notes, latent relay, and direct text at equal evidence hops."""
    return _run("E1", backend, dataset, output_dir, equal_hops=n_hops, **options)


def run_e2(backend=None, dataset=None, hop_budgets=(1, 3, 5, 10, 20), *, output_dir=None, **options):
    """Fixed evidence-budget curve, including the direct-text baseline."""
    return _run("E2", backend, dataset, output_dir, hops=tuple(hop_budgets), **options)


def run_e3(backend=None, dataset=None, *, output_dir=None, **options):
    """Answer-level correct, mismatched, zero, random, and no-context controls."""
    return _run("E3", backend, dataset, output_dir, **options)


def run_e4(backend=None, dataset=None, ratios=(0, 0.1, 0.2, 1), *, output_dir=None, **options):
    """Real causal boundary recomputation; 100% is full joint prefill."""
    return _run("E4", backend, dataset, output_dir, bridge_ratios=tuple(ratios), **options)


def run_e5(backend=None, dataset=None, *, output_dir=None, **options):
    """bf16, packed int8 and packed int4 storage, with bf16 attention."""
    return _run("E5", backend, dataset, output_dir, **options)


def run_e6(backend=None, beam_path=None, *, dataset=None, output_dir=None, **options):
    """Explicit current-version metadata filtering versus all versions."""
    return _run("E6", backend, dataset or beam_path, output_dir, **options)
