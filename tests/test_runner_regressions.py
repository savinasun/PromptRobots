import pytest

from astra_yam.cli import _config_from_args, build_parser


def test_cli_honors_waypoint_budget_and_policy_notes():
    args = build_parser().parse_args(["run", "--sim", "--goal", "test", "--max-waypoints", "17",
                                      "--policy-notes", "example.md"])
    cfg = _config_from_args(args)
    assert cfg.limits.max_waypoints == 17 and cfg.policy_notes_path == "example.md"


@pytest.mark.parametrize("call", [{"name": "nonexistent", "arguments": {}},
                                 {"name": "move_to", "arguments": "{broken"}])
def test_invalid_calls_respect_consecutive_rejection_budget(tmp_path, call):
    from test_session_sim import _make
    _, runner, _, _ = _make(tmp_path, script=[call] * 5, **{"limits.max_consecutive_rejections": 2})
    outcome = runner.run("test")
    assert outcome.status == "too_many_rejections" and outcome.llm_calls == outcome.rejections == 2


def test_successful_move_does_not_erase_rejection_count(tmp_path):
    from test_session_sim import _make
    script = [{"name": "move_to", "arguments": {"targets": {"left_x": 99}, "note": "bad"}},
              {"name": "move_to", "arguments": {"targets": {"left_gripper": .8}, "note": "ok"}},
              {"name": "done", "arguments": {"summary": "done", "hindsight": ""}}]
    _, runner, _, _ = _make(tmp_path, script=script)
    outcome = runner.run("test")
    assert outcome.status == "done" and outcome.rejections == 1


def test_estop_never_homes_on_end(tmp_path, monkeypatch):
    from test_session_sim import _make
    _, runner, _, _ = _make(tmp_path, **{"home_on_end": True, "robot.home_at_start": False})
    class Hooks:
        def on_astra_response(self, response):
            runner.request_estop()
    runner.hooks = Hooks()
    def unexpected_motion(*args, **kwargs):
        pytest.fail("homing after estop")
    monkeypatch.setattr("astra_yam.session.move_joint_space", unexpected_motion)
    assert runner.run("stop").status == "estop"
