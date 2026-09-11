#!/usr/bin/env python3
"""Run a pinned experiment matrix sequentially using one resident model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.real_experiments import ExperimentConfig, _atomic_json, run_suite


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=18000)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from engine.backends_mlx import MLXBackend
    backend = MLXBackend(matrix["model"], revision=matrix["revision"])
    started, completed = time.monotonic(), []
    for job in matrix["jobs"]:
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining < 1:
            raise TimeoutError("Matrix wall-time budget exhausted")
        config = ExperimentConfig(**{**matrix.get("defaults", {}), **job["config"],
                                     "output_dir": str(args.output_dir / job["name"]),
                                     "max_runtime_seconds": remaining})
        print(json.dumps({"event": "job_start", "job": job["name"],
                          "remaining_seconds": remaining}), flush=True)
        summary = run_suite(backend, config)
        completed.append({"job": job["name"], "completed_runs": summary["completed_unique_runs"],
                          "planned_runs": summary["planned_runs"],
                          "stop_reason": summary["stop_reason"]})
        _atomic_json(args.output_dir / "matrix-status.json", {
            "matrix": str(args.matrix), "jobs": completed,
            "elapsed_seconds": time.monotonic() - started,
            "memory": backend.memory_stats(),
        })
        if summary["stop_reason"] != "complete":
            raise RuntimeError(f"Incomplete job: {job['name']}: {summary['stop_reason']}")
        backend.clear_cache()
    print(json.dumps({"event": "matrix_complete", "jobs": len(completed),
                      "seconds": time.monotonic() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
