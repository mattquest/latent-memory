#!/usr/bin/env python3
"""Run one local experiment with bounded resources and durable monitoring.

Only the child process group is terminated. No machine settings are changed.
The command should itself checkpoint at example boundaries.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

GIB = 1024**3


def snapshot(pid: int, directory: Path, *, check_power: bool = True) -> dict:
    processes = []
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        pass
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    result = {
        "time": time.time(), "pid": pid, "process_count": len(processes),
        "rss_gib": rss / GIB,
        "available_gib": psutil.virtual_memory().available / GIB,
        "disk_free_gib": shutil.disk_usage(directory).free / GIB,
    }
    if check_power and sys.platform == "darwin":
        battery = psutil.sensors_battery()
        result["on_ac_power"] = battery.power_plugged if battery else None
        try:
            thermal = subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5,
            )
            result["thermal"] = thermal.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            result["thermal"] = "unavailable"
    return result


def stop_reason(state: dict, args: argparse.Namespace, elapsed: float) -> str | None:
    if elapsed >= args.max_seconds:
        return "wall_time_limit"
    if state["rss_gib"] > args.max_rss_gib:
        return "process_memory_limit"
    if state["available_gib"] < args.min_available_gib:
        return "low_system_memory"
    if state["disk_free_gib"] < args.min_disk_gib:
        return "low_disk_space"
    if not args.allow_battery and state.get("on_ac_power") is False:
        return "battery_power"
    # macOS normally protects thermals itself; stop if it reports severe limits.
    for line in state.get("thermal", "").splitlines():
        if "CPU_Speed_Limit" in line and "=" in line:
            try:
                if int(line.split("=", 1)[1].strip()) < 50:
                    return "severe_thermal_throttling"
            except ValueError:
                pass
    return None


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=6 * 3600)
    parser.add_argument("--max-rss-gib", type=float, default=32)
    parser.add_argument("--min-available-gib", type=float, default=12)
    parser.add_argument("--min-disk-gib", type=float, default=40)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    if min(args.max_seconds, args.max_rss_gib, args.poll_seconds) <= 0:
        parser.error("time, RSS, and polling limits must be positive")
    if min(args.min_available_gib, args.min_disk_gib) < 0:
        parser.error("minimum resource thresholds cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    lock_dir = root / "runs"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / ".experiment.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another guarded experiment is already running")
        return run(command, args)


def run(command: list[str], args: argparse.Namespace) -> int:
    initial = snapshot(os.getpid(), args.output)
    reason = stop_reason(initial, args, 0)
    if reason:
        print(f"Refusing to start: {reason}", file=sys.stderr, flush=True)
        (args.output / "guard-status.json").write_text(json.dumps({
            "status": "refused", "reason": reason, "resources": initial,
        }, indent=2))
        return 2
    start = time.monotonic()
    process = None
    awake = None
    peak = 0.0
    reason = None
    try:
        with (args.output / "process.log").open("a", buffering=1) as output, \
                (args.output / "resources.jsonl").open("a", buffering=1) as resources:
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
            process = subprocess.Popen(
                ["nice", "-n", "10", *command], stdout=output,
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            print(f"Experiment PID {process.pid}; log: {args.output / 'process.log'}", flush=True)
            if sys.platform == "darwin" and shutil.which("caffeinate"):
                awake = subprocess.Popen(["caffeinate", "-i", "-w", str(process.pid)])
            while process.poll() is None:
                state = snapshot(process.pid, args.output)
                state["elapsed_seconds"] = time.monotonic() - start
                peak = max(peak, state["rss_gib"])
                resources.write(json.dumps(state) + "\n")
                reason = stop_reason(state, args, state["elapsed_seconds"])
                if reason:
                    print(f"Stopping experiment: {reason}", flush=True)
                    terminate_group(process)
                    break
                time.sleep(args.poll_seconds)
    except (KeyboardInterrupt, SystemExit):
        reason = "interrupted"
        if process is not None:
            terminate_group(process)
    finally:
        if process is not None and process.poll() is None:
            terminate_group(process)
        if awake is not None and awake.poll() is None:
            awake.terminate()
            awake.wait(timeout=5)
        status = {
            "status": "stopped" if reason else "complete" if process and process.returncode == 0 else "failed",
            "reason": reason, "returncode": process.returncode if process else None,
            "elapsed_seconds": time.monotonic() - start, "peak_rss_gib": peak,
            "command": command,
            "limits": {k: v for k, v in vars(args).items() if k not in {"command", "output"}},
        }
        (args.output / "guard-status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
    return 2 if reason else process.returncode if process else 1


if __name__ == "__main__":
    raise SystemExit(main())
