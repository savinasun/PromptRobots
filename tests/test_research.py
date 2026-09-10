import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from astra_yam.astra_client import ScriptedAstraClient
from astra_yam.cli import _config_from_args, build_parser
from astra_yam.config import REPO_ROOT, load_config
from astra_yam.evaluation import Case, load_suite, promotion_gate, reset_scene, run_case, verify
from astra_yam.research import run_research, training_context, validate_proposal
from astra_yam.sim import SimWorld

SUITE = str(REPO_ROOT / "configs/airpods_research.yaml")


def config():
    return load_config(overrides={"robot.backend": "sim", "cameras.backend": "sim", "astra.backend": "scripted"})


def record(case, score=0.0, success=False, calls=5):
    return {"case": asdict(case), "verification": {"success": success, "score": score},
            "outcome": {"status": "done", "llm_calls": calls, "rejections": 0, "usage": {}, "hindsight": ""},
            "assisted": False, "trace": []}


def test_done_does_not_mean_success(tmp_path):
    result = run_case(config(), load_suite(SUITE)[0], tmp_path,
                      client_factory=lambda cfg, tools: ScriptedAstraClient(
                          [{"name": "done", "arguments": {"summary": "I opened it", "hindsight": ""}}], tools))
    assert result["outcome"]["status"] == "done"
    assert not result["verification"]["success"] and result["verification"]["score"] == 0
    assert (Path(result["outcome"]["log_dir"]) / "evaluation.json").exists()


def test_airpods_fixture_completes_through_the_runner(tmp_path):
    cfg = config()
    cfg.astra.script_path = str(REPO_ROOT / "tasks/research/airpods_nominal.json")
    result = run_case(cfg, load_suite(SUITE)[0], tmp_path)
    assert result["verification"]["success"]
    assert result["verification"]["metrics"]["lid_released"]
    assert result["outcome"]["rejections"] == 0


def test_verifier_requires_lid_release_and_body_support():
    world = SimWorld(scene="airpods")
    case = load_suite(SUITE)[0]
    obj = world.objects["airpods case"]
    obj.lid_angle, obj.held_by, obj.hanging, obj.lid_holder = 1.2, "left", True, "right"
    assert not verify(world, case)["success"]
    obj.lid_holder = None
    assert verify(world, case)["success"] and verify(world, case)["score"] == pytest.approx(1)
    obj.held_by = None
    assert not verify(world, case)["success"]


def test_schematic_camera_shows_lid_articulation():
    from astra_yam.sim import SimCameraSource
    world = SimWorld(scene="airpods")
    obj = world.objects["airpods case"]
    obj.hanging = True
    camera = SimCameraSource(world)
    closed = np.zeros((300, 300, 3), dtype=np.uint8)
    opened = closed.copy()
    camera._draw_object(closed, obj, (150, 150), 2500)
    obj.lid_angle = 1.5
    camera._draw_object(opened, obj, (150, 150), 2500)
    assert np.count_nonzero(closed != opened) > 100


def test_resets_are_reproducible_and_preserve_support():
    world = SimWorld(scene="airpods")
    case = load_suite(SUITE)[1]
    reset_scene(world, case)
    expected = {n: o.pos.copy() for n, o in world.objects.items()}
    world.objects["airpods case"].pos += 1
    reset_scene(world, case)
    assert all(np.array_equal(o.pos, expected[n]) for n, o in world.objects.items())
    obj, dish = world.objects["airpods case"], world.objects["wooden dish"]
    assert obj.pos[2] - obj.height / 2 == pytest.approx(dish.pos[2] + dish.height / 2)


def test_gate_rejects_regression_even_if_average_improves():
    cases = load_suite(SUITE)
    old = [record(cases[0], .4), record(cases[1], .4)]
    new = [record(cases[0], 1, True), record(cases[1], .3)]
    assert not promotion_gate(old, new)[0]
    new[1] = record(cases[1], .4)
    assert promotion_gate(old, new)[0]
    new[0]["assisted"] = True
    assert not promotion_gate(old, new)[0]


def test_gate_requires_improvement_and_checks_rejections():
    case = load_suite(SUITE)[0]
    old = [record(case, 1, True, 10)]
    assert not promotion_gate(old, copy.deepcopy(old))[0]
    new = [record(case, 1, True, 9)]
    assert promotion_gate(old, new)[0]
    new[0]["outcome"]["rejections"] = 1
    assert not promotion_gate(old, new)[0]
    assert not promotion_gate(old, [record(load_suite(SUITE)[1])])[0]


def test_optimizer_never_gets_validation_trace():
    cases = load_suite(SUITE)
    records = [record(c) for c in cases]
    records[1]["trace"] = ["validation-only-secret"]
    assert "validation-only-secret" not in json.dumps(training_context(records, "", [], ""))


def test_multitask_suite_uses_shared_candidate_and_hides_validation(tmp_path):
    suite = str(REPO_ROOT / "configs/multitask_research.yaml")
    cases = load_suite(suite)
    assert len({c.verifier for c in cases}) == 3
    from astra_yam.research import ScriptedOptimizer

    class Optimizer(ScriptedOptimizer):
        def propose(self, context, iteration):
            assert len(context["training_trials"]) == 3
            assert all(t["case_id"].endswith("train") for t in context["training_trials"])
            return super().propose(context, iteration)

    notes_seen = []
    def evaluate(cfg, case, root, notes):
        advice = notes.read_text()
        notes_seen.append(advice)
        return record(case, score=1 if advice else 0, success=bool(advice))

    report = run_research(config(), suite, str(tmp_path / "multi"), iterations=1, mock_optimizer=True,
                          optimizer_factory=lambda cfg, root: Optimizer(), evaluate=evaluate)
    assert report["iterations"][0]["accepted"]
    assert notes_seen[:6] == [""] * 6
    assert len(set(notes_seen[6:])) == 1
    assert "airpod" not in notes_seen[6].lower()


def test_candidate_cannot_set_controller_config():
    with pytest.raises(ValueError):
        validate_proposal({"motion.linear_speed_mps": 5})
    with pytest.raises(ValueError):
        validate_proposal({key: "x" * 7000 for key in
                           ("hypothesis", "evidence", "policy_notes", "expected_improvement")})


def test_two_iterations_keep_improvement_then_reject_regression(tmp_path):
    class Optimizer:
        def __init__(self):
            self.calls = 0
        def propose(self, context, iteration):
            self.calls += 1
            return {"hypothesis": "test", "evidence": "test", "expected_improvement": "test",
                    "policy_notes": f"candidate {iteration}"}
        def close(self):
            pass
    optimizer = Optimizer()
    def evaluate(cfg, case, root, notes):
        score = {"": 0, "candidate 1": 1, "candidate 2": .5}[notes.read_text()]
        return record(case, score, score == 1)
    output = tmp_path / "experiment"
    report = run_research(config(), SUITE, str(output), iterations=2, mock_optimizer=True,
                          optimizer_factory=lambda cfg, root: optimizer, evaluate=evaluate)
    assert optimizer.calls == 2
    assert [i["accepted"] for i in report["iterations"]] == [True, False]
    assert (output / "best_policy.md").read_text() == "candidate 1"
    assert report["best"]["success_rate"] == 1 and report["baseline"]["success_rate"] == 0
    assert (output / "candidate_002.diff").exists()
    with pytest.raises(FileExistsError):
        run_research(config(), SUITE, str(output), mock_optimizer=True)


def test_contract_mutation_stops_experiment(tmp_path):
    cfg = config()
    def evaluate(cfg, case, root, notes):
        cfg.motion.linear_speed_mps *= 2
        return record(case)
    output = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="contract changed"):
        run_research(cfg, SUITE, str(output), iterations=0, evaluate=evaluate)
    assert json.loads((output / "report.json").read_text())["status"] == "failed"


def test_strategy_mutation_invalidates_comparison(tmp_path):
    def evaluate(cfg, case, root, notes):
        notes.write_text("changed advice during evaluation")
        return record(case)
    with pytest.raises(RuntimeError, match="policy changed"):
        run_research(config(), SUITE, str(tmp_path / "changed"), iterations=0, evaluate=evaluate)


def test_script_fixture_is_part_of_frozen_contract(tmp_path):
    cfg = config()
    script = tmp_path / "script.json"
    script.write_text("[]")
    cfg.astra.script_path = str(script)
    def evaluate(cfg, case, root, notes):
        script.write_text('[{"name":"done"}]')
        return record(case)
    with pytest.raises(RuntimeError, match="contract changed"):
        run_research(cfg, SUITE, str(tmp_path / "changed"), iterations=0, evaluate=evaluate)


def test_research_stop_file_prevents_next_trial(tmp_path):
    feedback = tmp_path / "feedback.txt"
    feedback.write_text("/stop\n")
    def never_run(*args):
        pytest.fail("trial started after stop")
    report = run_research(config(), SUITE, str(tmp_path / "stop"), iterations=0,
                          feedback_file=str(feedback), evaluate=never_run)
    assert report["status"] == "operator_stop"


def test_hardware_rejected_before_any_trial_or_artifacts(tmp_path):
    cfg = config()
    cfg.robot.backend = "zmq"
    with pytest.raises(ValueError, match="simulation"):
        run_research(cfg, SUITE, str(tmp_path / "no"))
    assert not (tmp_path / "no").exists()


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
