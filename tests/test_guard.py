from argparse import Namespace
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from scripts import run_guarded
from scripts.run_guarded import live_group_members, stop_reason, terminate_group


def test_resource_guard_only_stops_at_real_limits():
    args = Namespace(max_seconds=3600, max_rss_gib=32,
                     min_available_gib=12, min_disk_gib=40, allow_battery=False)
    state = dict(rss_gib=16, available_gib=48, disk_free_gib=600,
                 on_ac_power=True, thermal="No thermal warning level has been recorded")
    assert stop_reason(state, args, 10) is None
    assert stop_reason(state, args, 3600) == "wall_time_limit"
    for change, reason in [
        ({"rss_gib": 33}, "process_memory_limit"),
        ({"available_gib": 11}, "low_system_memory"),
        ({"disk_free_gib": 39}, "low_disk_space"),
        ({"on_ac_power": False}, "battery_power"),
        ({"thermal": "CPU_Speed_Limit = 40"}, "severe_thermal_throttling"),
    ]:
        assert stop_reason({**state, **change}, args, 10) == reason


def descendant_command(pidfile, ignore_term=False):
    child = (
        "import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN) if sys.argv[2]=='ignore' else None; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(3)"
    )
    leader = (
        "import subprocess,sys,time; from pathlib import Path; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); "
        "deadline=time.monotonic()+2\n"
        "while not Path(sys.argv[2]).exists() and time.monotonic()<deadline: time.sleep(.01)"
    )
    return [sys.executable, "-c", leader, child, str(pidfile), "ignore" if ignore_term else "normal"]


def cleanup_test_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.parametrize("ignore_term", [False, True])
def test_cleanup_survives_exited_leader_and_kills_stubborn_descendant(tmp_path, ignore_term):
    pidfile = tmp_path / "descendant.pid"
    leader = subprocess.Popen(descendant_command(pidfile, ignore_term), start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        leader.wait(timeout=2)
        child_pid = int(pidfile.read_text())
        assert leader.returncode == 0
        assert child_pid in {p.pid for p in live_group_members(leader.pid)}
        terminate_group(leader, grace_seconds=0.1, kill_seconds=1)
        assert not live_group_members(leader.pid)
        # An already vanished process group is normal, not an exception.
        terminate_group(leader, grace_seconds=0.1, kill_seconds=1)
    finally:
        cleanup_test_group(leader.pid)
        leader.wait(timeout=2)


def test_guard_cleans_descendants_before_reporting_completion(tmp_path, monkeypatch):
    # Call run() directly so this inert unit test does not contend for the
    # repository's model-experiment lock or query real power/thermal state.
    monkeypatch.setattr(run_guarded, "snapshot", lambda *args, **kwargs: {
        "rss_gib": 0.01, "available_gib": 64, "disk_free_gib": 100,
        "on_ac_power": True,
    })
    monkeypatch.setattr(run_guarded.shutil, "which", lambda _: None)
    pidfile = tmp_path / "descendant.pid"
    args = Namespace(output=tmp_path, max_seconds=2, max_rss_gib=1,
                     min_available_gib=0, min_disk_gib=0, poll_seconds=0.02,
                     allow_battery=True)
    result = run_guarded.run(descendant_command(pidfile), args)
    status = json.loads((tmp_path / "guard-status.json").read_text())
    assert result == 2
    assert status["status"] == "stopped"
    assert status["reason"] == "leader_exited_with_live_descendants"
    child_pid = int(pidfile.read_text())
    try:
        import psutil
        assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
