from argparse import Namespace

from scripts.run_guarded import stop_reason


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
