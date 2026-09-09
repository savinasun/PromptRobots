"""What the loop actually spends time on: request size, prompt-cache stability, motion profile."""
import json
import shlex
from pathlib import Path

import numpy as np
import pytest

from astra_yam.cli import _config_from_args, build_parser
from astra_yam.config import AstraConfig, PipelineConfig
from astra_yam.observation import build_observation_item, count_images, images_kept_before, prune_image_history


def _conversation(n_turns: int, frames=None):
    # ~8 KB per frame, in the range a 640x480 JPEG at quality 85 actually comes out at
    frames = frames if frames is not None else {c: b"\xff\xd8" + bytes(8000) for c in ("top", "left", "right")}
    eef = {f"{a}_{d}": 0.1 for a in ("left", "right")
           for d in ("x", "y", "z", "yaw", "pitch", "roll", "gripper")}
    items = [{"role": "system", "content": "s"}, {"role": "user", "content": "Goal: g"}]
    for turn in range(1, n_turns + 1):
        items.append(build_observation_item("g", np.zeros(14), eef, 3000, turn, frames))
        items.append({"type": "function_call", "call_id": f"c{turn}", "name": "move_to", "arguments": "{}"})
        items.append({"type": "function_call_output", "call_id": f"c{turn}", "output": "{}"})
    return items


def test_boundary_moves_once_every_n_turns_and_keeps_n_to_2n():
    N = 4
    dropped = [images_kept_before(n, N) for n in range(1, 21)]
    assert dropped[:7] == [0] * 7                     # nothing dropped until 2N observations exist
    assert dropped[7:12] == [4, 4, 4, 4, 8]
    assert all(d % N == 0 for d in dropped)           # always a multiple of N: a whole block at a time
    for n, d in enumerate(dropped, start=1):
        assert N <= n - d <= 2 * N or n <= N          # images in view stay within one block of N
    assert sum(1 for a, b in zip(dropped, dropped[1:]) if a != b) == len(dropped) // N - 1
    assert images_kept_before(50, 0) == 0             # 0 = keep everything


def _items_json(items):
    return [json.dumps(it, sort_keys=True) for it in items]


def test_pruned_request_prefix_is_stable_between_turns():
    """The provider's prompt cache matches on a prefix, so the pruned tail must not shift every turn."""
    N = 3
    rewrote_the_prefix = []
    for turns in range(2, 16):
        before = _items_json(prune_image_history(_conversation(turns - 1), N))
        after = _items_json(prune_image_history(_conversation(turns), N))
        rewrote_the_prefix.append(after[: len(before)] != before)   # True: last turn's items were rewritten
    assert sum(rewrote_the_prefix) == len(rewrote_the_prefix) // N  # once per block of N turns, not every turn


def test_bounded_history_stops_the_request_from_growing():
    items = _conversation(60)
    full, pruned = json.dumps(items), json.dumps(prune_image_history(items, 4))
    assert count_images(items) == 180 and count_images(prune_image_history(items, 4)) <= 8 * 3
    assert len(pruned) < len(full) / 5                # the whole run vs the last handful of observations
    assert PipelineConfig().astra.image_history == 4  # and that is the default


def test_prompt_cache_key_is_sent_and_dropped_if_rejected(monkeypatch):
    import openai

    from astra_yam.astra_client import OpenAIAstraClient

    class FakeResponses:
        def __init__(self):
            self.kwargs = []

        def create(self, **kw):
            self.kwargs.append(kw)
            if len(self.kwargs) == 1:
                raise ValueError("Unknown parameter: 'prompt_cache_key'")
            return type("R", (), {"output": [], "usage": None, "id": "r", "model": "m"})()

    fake = FakeResponses()
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: type("C", (), {"responses": fake})())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAIAstraClient(AstraConfig(), tools=[])
    client.create([{"role": "user", "content": "hi"}])
    assert fake.kwargs[0]["prompt_cache_key"] == "astra-yam"
    assert "prompt_cache_key" not in fake.kwargs[1]
    assert AstraConfig(prompt_cache_key=None).prompt_cache_key is None      # opt out entirely


def _cfg(cmdline):
    return _config_from_args(build_parser().parse_args(shlex.split(cmdline)))


def test_fast_profile_speeds_up_motion_without_overriding_explicit_flags():
    base, fast = _cfg("run --goal g").motion, _cfg("run --goal g --fast").motion
    assert (base.linear_speed_mps, base.yaw_speed_rps, base.settle_seconds) == (0.01, 0.15, 0.3)
    assert fast.linear_speed_mps == 0.05 and fast.yaw_speed_rps == 0.6 and fast.settle_seconds == 0.1
    assert fast.gripper_speed_per_s == 1.5
    assert _cfg("run --goal g --fast --speed 0.02").motion.linear_speed_mps == 0.02          # flag wins
    assert _cfg("run --goal g --fast --set motion.settle_seconds=0.4").motion.settle_seconds == 0.4
    # the gateway's safety numbers are untouched by the profile
    for field in ("tracking_abort_rad", "max_ik_joint_jump_rad", "arm_clearance_m", "joint_limit_margin_rad"):
        assert getattr(fast, field) == getattr(base, field)


def test_faster_speed_means_proportionally_fewer_waypoints():
    """The wall clock of a motion is waypoints / cadence_hz, so speed maps straight onto time."""
    from astra_yam.config import ARMS
    from astra_yam.embodiment import ARM_SLICES
    from astra_yam.gateway import SafetyGateway
    from astra_yam.kinematics import ArmKinematics
    from astra_yam.sim import SimYamRobot

    steps = {}
    for speed in (0.01, 0.05):
        cfg = PipelineConfig()
        cfg.motion.linear_speed_mps = speed
        kin = ArmKinematics(limit_margin=cfg.motion.joint_limit_margin_rad)
        q = np.zeros(14)
        q[0:6] = q[7:13] = cfg.robot.home_joints_left
        q[6] = q[13] = 1.0
        robot = SimYamRobot(initial_q=q)
        gw = SafetyGateway(cfg, kin, robot, {a: kin.fk(q[ARM_SLICES[a]])[1] for a in ARMS}, realtime=False)
        plan = gw.plan({"left_x": 0.40, "left_z": 0.10})
        steps[speed] = plan.steps
        assert not plan.paced                        # joint pacing does not eat the gain at 5 cm/s
    assert steps[0.01] / steps[0.05] == pytest.approx(5, abs=0.5)


def test_transcript_reports_where_the_time_went(tmp_path):
    from test_session_sim import _make

    cfg, runner, world, astra = _make(tmp_path)
    outcome = runner.run("time me")
    text = (Path(outcome.log_dir) / "transcript.txt").read_text()
    time_table = text.split("--- TIME ")[1]
    assert "Astra (API + model)" in time_table and "motion (gateway)" in time_table
    assert "observation capture" in time_table and "TOTAL" in time_table
    assert "motion budget: 1 cm/s" in time_table and "--fast" in time_table
    assert "images in this request" in text
