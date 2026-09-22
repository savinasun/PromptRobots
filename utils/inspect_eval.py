"""Run Astra/YAM tasks using Inspect Robots' evaluator, graders and scorers."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from inspect_robots import eval
from inspect_robots.errors import EmbodimentFault
from inspect_robots.logging.json_log import JsonLogSink
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length, operator_scorer
from inspect_robots.task import Epochs, Task

from utils.astra_client import ScriptedAstraClient, find_api_key
from utils.config import DIM_NAMES, REPO_ROOT, load_config, to_dict
from utils.inspect_adapter import (
    AstraInspectPolicy, GatewayTraceSink, PromptYamEmbodiment, Rot6dYamEmbodiment,
)


def load_scenes(path: str) -> list[Scene]:
    """Load PromptRobots goal files, retaining per-goal setup comments and IDs."""
    source = Path(path).resolve()
    source_text = source.read_text()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    scenes = []
    comments = []
    for line_no, raw in enumerate(source_text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line.removeprefix("#").strip())
            continue
        notes = "\n".join(comments)
        ids = re.findall(r"^([A-Z]{2}\d+)\b", notes, flags=re.MULTILINE)
        scenes.append(Scene(
            id=ids[-1] if ids else f"{source.stem}-{len(scenes) + 1:03d}", instruction=line,
            metadata={"source": str(source), "source_sha256": source_hash,
                      "source_line": line_no, "setup_notes": notes,
                      "setup_guide": str(source.parent / "README.md")},
        ))
        comments = []
    if not scenes:
        raise ValueError(f"no goals found in {source}")
    return scenes


def make_task(scenes, cfg, *, name="astra-yam", epochs=1, graded=True):
    """Use the built-in operator success scorer and episode-length scorer.

For ungraded plumbing runs only episode length is reported. A model's `done`
claim never produces a success score. One episode step is a tool call.
    """
    scorers = [operator_scorer(), episode_length()] if graded else [episode_length()]
    return Task(name=name, scenes=scenes, scorer=scorers,
                max_steps=cfg.limits.max_waypoints, epochs=Epochs(count=epochs, reducer="mean"))


def make_policy(name, cfg, params):
    """Select PromptRobots Astra, the existing agent plugin, or another Policy factory."""
    if name == "prompt_astra":
        if params:
            raise ValueError("-P options apply to interchangeable policies; prompt_astra uses the station YAML")
        return AstraInspectPolicy(cfg)
    if name == "agent":
        from inspect_robots_agent.policy import agent_policy
        # The agent policy names models as <provider>/<model>; "openrouter" models already carry their own
        # vendor prefix (qwen/...), so the OpenRouter provider prefix is added in front of it.
        provider = "openai" if cfg.astra.backend == "openai" else "openrouter"
        settings = {"model": f"{provider}/{cfg.astra.model}", "max_llm_calls": cfg.limits.max_llm_calls,
                    **params}
        if settings["model"].startswith("openai/"):
            settings.setdefault("wire", "responses")
            settings.setdefault("effort", "low")
        if settings["model"].startswith(("openai/", "openrouter/")):
            env = dict(os.environ)
            if not env.get(cfg.astra.api_key_env):
                key, _ = find_api_key(cfg.astra.api_key_env)
                if key:
                    env[cfg.astra.api_key_env] = key
            settings.setdefault("env", env)
        return agent_policy(**settings)
    if ":" in name:
        module, factory = name.split(":", 1)
        return getattr(importlib.import_module(module), factory)(**params)
    from inspect_robots.registry import resolve
    return resolve("policy", name, **params)


def policy_params(values):
    """Parse framework-style -P key=value options without evaluating Python."""
    params = {}
    for item in values:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError("policy arguments must be -P key=value")
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def prepare_hardware_scene(scene):
    """Require a fresh operator-staged scene and arm pose before every repeat."""
    print(f"\nScene {scene.id}: {scene.instruction}")
    if scene.metadata.get("setup_notes"):
        print(scene.metadata["setup_notes"])
    if scene.metadata.get("setup_guide"):
        print(f"Setup and pass criteria: {scene.metadata['setup_guide']}")
    try:
        answer = input("Stage the scene and starting arm pose; keep the e-stop available. "
                       "Type RUN to start this trial: ")
    except EOFError as exc:
        raise EmbodimentFault("hardware scene setup needs an attended terminal") from exc
    if answer.strip() != "RUN":
        raise EmbodimentFault("operator did not start the hardware trial")


def provenance(cfg, config_path):
    """Record the local source revisions and hashes that define this evaluation."""
    import inspect_robots

    result = {"config_path": str(Path(config_path).resolve()), "files": {}, "repositories": {}}
    from utils.arm_models import get_arm_model

    filenames = [config_path, cfg.robot.mjcf_path, cfg.system_prompt_path,
                 cfg.tilt_note_path, cfg.prompts_path, cfg.viz.urdf_path or get_arm_model(cfg.embodiment_name).urdf_path,
                 Path(__file__), Path(__file__).with_name("inspect_adapter.py")]
    if Path(cfg.cameras.station_config_path).is_file():
        filenames.append(cfg.cameras.station_config_path)
    for filename in filenames:
        path = Path(filename).resolve()
        result["files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, root in (("PromptRobots", REPO_ROOT),
                       ("inspect-robots", Path(inspect_robots.__file__).resolve().parents[2])):
        try:
            commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                             text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                                 text=True, stderr=subprocess.DEVNULL).strip())
            result["repositories"][name] = {"commit": commit, "dirty": dirty}
        except (OSError, subprocess.CalledProcessError):
            result["repositories"][name] = {"commit": None}
    return result


def smoke_client(cfg, tools):
    """Exercise strict tools and real gateway planning without API or hardware."""
    script = [{"name": "move_to", "arguments": {
        "targets": {name: (opening if name == "left_gripper" else None) for name in DIM_NAMES}}}
        for opening in (0.8, 1.0)]
    script.append({"name": "done", "arguments": {}})
    return ScriptedAstraClient(script, tools=tools, actions_only=True)


def parser():
    """Expose a station-specific launcher around the framework Python API."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "smoke", "run"))
    p.add_argument("--config", default=str(REPO_ROOT / "configs/skild_yam_8.yaml"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--hardware", action="store_true", help="use the in-house ZMQ YAM and RealSense cameras")
    mode.add_argument("--sim", action="store_true", help="use PromptRobots' kinematic simulator (default)")
    goals = p.add_mutually_exclusive_group()
    goals.add_argument("--goal")
    goals.add_argument("--goals-file")
    p.add_argument("--scene-id", help="run just this ID from a goals file, e.g. SP01")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-calls", type=int)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--max-waypoints", type=int)
    p.add_argument("--model", help="model ID for the configured astra backend (default: the backend's default)")
    p.add_argument("--policy", default="prompt_astra",
                   help="prompt_astra, agent, a registered policy, or module:factory")
    p.add_argument("--action-format", choices=("euler", "rot6d"),
                   help="default: rot6d for agent, PromptRobots Euler for other policies")
    p.add_argument("-P", dest="policy_params", action="append", default=[], metavar="KEY=VALUE",
                   help="arguments for the selected agent/policy factory, e.g. model=provider/model")
    p.add_argument("--grader", choices=("operator", "none"), default="operator")
    p.add_argument("--scene", default="blocks", help="PromptRobots simulator preset")
    p.add_argument("--pin-tilt", action="store_true", help="pin pitch/roll to zero, as required by spatial tasks")
    p.add_argument("--tool-floor-z", type=float, help="measured tool floor in metres relative to the arm bases")
    p.add_argument("--fast-sim", action="store_true")
    p.add_argument("--log-dir", default=str(REPO_ROOT / "runs/inspect"))
    p.add_argument("--no-store-frames", action="store_true")
    p.add_argument("--rerun", action="store_true", help="save a framework Rerun recording (requires rerun-sdk)")
    return p


def main(argv=None):
    """Validate configuration, then delegate all episodes and scoring to eval()."""
    p = parser()
    args = p.parse_args(argv)
    params = policy_params(args.policy_params)
    if args.command == "smoke" and (args.policy != "prompt_astra" or params):
        p.error("smoke uses a scripted Astra protocol fixture; use run for other policies")
    if args.hardware and (args.command == "smoke" or args.fast_sim):
        p.error("smoke and --fast-sim require simulation")
    overrides = {"robot.home_at_start": False, "home_on_end": False,
                 "astra.model": args.model, "sim.scene": args.scene,
                 "limits.max_llm_calls": args.max_calls, "limits.max_trial_seconds": args.max_seconds,
                 "limits.max_waypoints": args.max_waypoints, "motion.tool_floor_z_m": args.tool_floor_z}
    if not args.hardware:
        overrides.update({"robot.backend": "sim", "cameras.backend": "sim"})
    if args.pin_tilt:
        overrides.update({"bounds.pitch": [0.0, 0.0], "bounds.roll": [0.0, 0.0]})
    if args.command == "smoke":
        overrides.update({"astra.backend": "scripted", "limits.max_llm_calls": 3})
    cfg = load_config(args.config, overrides)
    action_format = args.action_format or ("rot6d" if args.policy == "agent" else "euler")
    if args.policy == "prompt_astra" and action_format != "euler":
        p.error("prompt_astra requires --action-format euler")
    body_class = Rot6dYamEmbodiment if action_format == "rot6d" else PromptYamEmbodiment
    for label, value in (("epochs", args.epochs), ("max-calls", cfg.limits.max_llm_calls),
                         ("max-seconds", cfg.limits.max_trial_seconds),
                         ("max-waypoints", cfg.limits.max_waypoints)):
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            p.error(f"{label} must be positive and finite")
    if args.hardware and (cfg.robot.backend != "zmq" or cfg.cameras.backend != "realsense"):
        p.error("--hardware requires robot.backend=zmq and cameras.backend=realsense")
    sources = provenance(cfg, args.config)
    if args.command == "preflight":
        # Configuration/import checks only: no sockets, camera resets, API requests or motion.
        policy = make_policy(args.policy, cfg, params)
        body = body_class(cfg, prepare_scene=prepare_hardware_scene if args.hardware else None)
        from inspect_robots.compat import assert_compatible
        bind = getattr(policy, "bind", None)
        if callable(bind):
            bind(body.info)
        assert_compatible(policy, body, make_task([Scene("preflight", "check configuration")], cfg))
        key, _ = find_api_key(cfg.astra.api_key_env)
        report = {"mode": "hardware" if args.hardware else "sim", "policy": args.policy,
                  "model": params.get("model", cfg.astra.model),
                  "api_key_present": bool(key), "robot": f"{cfg.robot.host}:{cfg.robot.port}",
                  "cameras": cfg.cameras.names, "station_config_exists": Path(cfg.cameras.station_config_path).is_file(),
                  "motion_performed": False, "connectivity_tested": False, "sources": sources}
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "smoke":
        scenes = [Scene("smoke", "Open and close the left gripper, then finish.")]
        graded = False
    else:
        if not args.goal and not args.goals_file:
            p.error("run requires --goal or --goals-file")
        scenes = load_scenes(args.goals_file) if args.goals_file else [Scene("goal-001", args.goal)]
        graded = args.grader == "operator"
    if args.scene_id:
        scenes = [scene for scene in scenes if scene.id == args.scene_id]
        if not scenes:
            p.error(f"no scene with ID {args.scene_id!r}")
    if (args.hardware or graded) and not sys.stdin.isatty():
        p.error("hardware setup and operator grading require an attended terminal; use --grader none for ungraded simulation")
    if args.hardware and cfg.motion.tool_floor_z_m is None:
        p.error("station bounds extend below the table: set --tool-floor-z to the measured table height before hardware evaluation")
    if args.policy == "prompt_astra" and cfg.astra.is_remote and not find_api_key(cfg.astra.api_key_env)[0]:
        p.error(f"{cfg.astra.api_key_env} is missing; use PromptRobots' existing key configuration")
    # Provenance and resolved config are attached to each Scene so the core persists them.
    scenes = [Scene(scene.id, scene.instruction, metadata={**scene.metadata, "provenance": sources,
                    "promptrobots_config": to_dict(cfg), "graded": graded}) for scene in scenes]
    task_name = "yam-smoke" if args.command == "smoke" else (
        f"yam-{Path(args.goals_file).stem}" if args.goals_file else "yam-eval")
    task = make_task(scenes, cfg, name=task_name,
                     epochs=args.epochs, graded=graded)
    kwargs = {"client_factory": smoke_client} if args.command == "smoke" else {}
    policy = AstraInspectPolicy(cfg, **kwargs) if args.command == "smoke" else make_policy(args.policy, cfg, params)
    body = body_class(cfg, realtime=not (args.fast_sim or args.command == "smoke"),
                              prepare_scene=prepare_hardware_scene if args.hardware else None)
    json_sink = JsonLogSink(args.log_dir)
    sinks = [json_sink, GatewayTraceSink(args.log_dir)]
    if args.rerun:
        from inspect_robots.logging.rerun_sink import RerunSink
        sinks.append(RerunSink(recording_dir=args.log_dir))
    try:
        (log,) = eval(task, policy, body, log_dir=args.log_dir, sinks=sinks,
                      grader="operator" if graded else None, fail_on_error=True,
                      store_frames=not args.no_store_frames, store_actions=True)
    finally:
        try:
            close = getattr(policy, "close", None)
            if callable(close):
                close()
        finally:
            body.close()
    print(json.dumps({"status": log.status, "error": log.error, "log": str(json_sink.path),
                      "metrics": log.results.metrics, "trials": log.results.total_trials}, indent=2))
    return 0 if log.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
