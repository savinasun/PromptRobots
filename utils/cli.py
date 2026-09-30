"""Command line interface.

  python -m utils run --goal "pick up blue and place on top of green block"
  python -m utils run --goals-file goals.txt
  python -m utils check                                    # robot server / cameras / API key
  python -m utils sim-server                               # simulated rig on the gello ZMQ protocol
  python -m utils show-prompt                              # system prompt + tool schemas
  python -m utils run --sim --viser --goal "..."           # + 3D view / operator UI at http://<host>:8080
  python -m utils viz --scene kitchen                      # 3D manual bench (gizmo targets through the gateway)
  python -m utils workspace                                # URDF joint limits + reachable bounds as YAML
  python -m utils run --embodiment ur5e_arms --sim ...     # drive the bimanual UR5e (or --config configs/skild_ur5e.yaml)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils import __version__
from utils.arm_models import ARM_MODELS, DEFAULT_ARM_MODEL, get_arm_model
from utils.config import ARMS, PipelineConfig, dump_config, load_config
from utils.embodiment import (ARM_SLICES, arm_poses, build_policy_prompt, build_tools, eef_state_dict,
                              format_eef_state, start_rotations)
from utils.kinematics import ArmKinematics


def _parse_set(values: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in values or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got '{kv}'")
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="utils", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="YAML config (see configs/)")
        sp.add_argument("--embodiment", choices=sorted(ARM_MODELS),
                        help="which arms are on the rig: yam_arms (default) or ur5e_arms; sets embodiment_name and "
                             "every kinematic default the config leaves unset")
        output = sp.add_mutually_exclusive_group()
        output.add_argument("--actions-only", dest="actions_only", action="store_true", default=None,
                            help="strict actions, low reasoning effort, no language output (branch default)")
        output.add_argument("--language-output", dest="actions_only", action="store_false",
                            help="restore notes, hindsight, and configurable reasoning for comparisons")
        sp.add_argument("--policy-notes", help="optional strategy file; no task advice is loaded by default")
        planning = sp.add_mutually_exclusive_group()
        planning.add_argument("--planning", dest="planning", action="store_true", default=None,
                              help="System 2 text planner -> System 1 motion model (configured under astra)")
        planning.add_argument("--no-planning", dest="planning", action="store_false",
                              help="disable planning mode enabled in YAML")
        sp.add_argument("--planner-model", help="System 2 model ID; also enables planning mode")
        sp.add_argument("--planner-backend", choices=["openrouter", "openai", "scripted"])
        sp.add_argument("--planner-effort", choices=["low", "medium", "high", "xhigh", "max"])
        sp.add_argument("--dynamic-scene", action="store_true", help="recheck the scene before actions and observe between short moves")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="dotted override, e.g. motion.linear_speed_mps=0.02")
        sp.add_argument("--robot", choices=["zmq", "sim"], help="robot backend")
        sp.add_argument("--cameras", choices=["station", "realsense", "sim", "none"], help="camera backend")
        sp.add_argument("--host"), sp.add_argument("--port", type=int)
        sp.add_argument("--sim", action="store_true", help="shortcut for --robot sim --cameras sim")
        sp.add_argument("--log-dir", help="where trial folders (and check frames) are written")
        sp.add_argument("--scene", choices=["blocks", "kitchen", "airpods", "airpod_bowl", "chili", "empty"], help="simulator object preset")
        sp.add_argument("--viser", action="store_true", help="3D visualization + operator UI in the browser")
        sp.add_argument("--viser-port", type=int, help="viser port (default 8080)")
        sp.add_argument("--viser-host", help="viser bind host (default 0.0.0.0)")
        sp.add_argument("--no-viser-render", action="store_true",
                        help="sim: keep schematic camera images instead of browser renders")

    r = sub.add_parser("run", help="run one goal (or a file of goals) in closed loop")
    common(r)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--goal", help="task instruction for Astra")
    g.add_argument("--goals-file", help="text file, one goal per line (# comments allowed)")
    r.add_argument("--script", help="JSON list of tool calls for the scripted Astra backend")
    r.add_argument("--model"), r.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    shadow = r.add_mutually_exclusive_group()
    shadow.add_argument("--shadow-model", metavar="MODEL", action="append",
                        help="also send every request to this OpenRouter model and show its proposed action + note "
                             "next to the controlling model's (never executed); enables shadow mode; repeatable")
    shadow.add_argument("--no-shadow", action="store_true", help="disable a shadow model enabled in the YAML")
    r.add_argument("--max-calls", type=int,
                   help="hard cap on LLM calls per trial (planner + motion calls in planning mode)")
    r.add_argument("--max-waypoints", type=int)
    r.add_argument("--max-seconds", type=float), r.add_argument("--image-history", type=int)
    r.add_argument("--speed", type=float, help="linear speed in m/s (default 0.01)")
    r.add_argument("--fast", action="store_true",
                   help="faster motion profile: 5 cm/s linear (5x), 0.6 rad/s yaw (4x), gripper 1.5/s (3x), "
                        "0.1 s settle. Same gateway checks and the same 0.25 rad tracking abort; explicit "
                        "flags (--speed) and --set still win")
    r.add_argument("--release-tilt", type=float, metavar="DEG",
                   help="set symmetric pitch/roll bounds of +/- DEG degrees; 0 pins them at the start "
                        "orientation (default: the full representable range, +/-90 deg pitch, +/-180 deg roll)")
    r.add_argument("--no-home", action="store_true", help="do not move to the home pose before the trial")
    r.add_argument("--home-on-end", action="store_true")
    r.add_argument("--strict-gateway", action="store_true", help="first rejected packet ends the session")
    r.add_argument("--feedback-file", help="operator feedback lines are tailed from this file")
    r.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation before moving the robot")
    r.add_argument("--fast-sim", action="store_true", help="simulation only: skip real-time sleeps")

    c = sub.add_parser("check", help="connectivity check: robot server, cameras, API key")
    common(c)
    c.add_argument("--save-frames", action="store_true", help="save one frame per camera to runs/check_frames/")

    s = sub.add_parser("sim-server", help="serve a simulated bimanual rig on the gello ZMQ protocol")
    s.add_argument("--host", default="127.0.0.1"), s.add_argument("--port", type=int, default=6001)
    s.add_argument("--embodiment", choices=sorted(ARM_MODELS), default=DEFAULT_ARM_MODEL)

    v = sub.add_parser("viz", help="3D manual test bench: simulated arms + gizmo targets through the gateway")
    common(v)

    w = sub.add_parser("workspace", help="derive joint limits + reachable bounds from the URDF (no estimation)")
    common(w)
    w.add_argument("--samples", type=int, default=200_000, help="FK samples of the joint box (default 200000)")
    w.add_argument("--arm", default="left", choices=["left", "right"])

    sp = sub.add_parser("show-prompt", help="print the system prompt and the tool schemas")
    common(sp)
    fmt = sp.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="print the tools as JSON only")
    fmt.add_argument("--bundle-json", action="store_true", help="print the exact assembled system prompt and tools as JSON")
    sp.add_argument("--max-calls", type=int)
    return p


def _config_from_args(args) -> PipelineConfig:
    overrides = _parse_set(getattr(args, "set", None))
    if getattr(args, "embodiment", None):
        overrides["embodiment_name"] = args.embodiment
    if getattr(args, "sim", False):
        overrides.setdefault("robot.backend", "sim")
        overrides.setdefault("cameras.backend", "sim")
    if getattr(args, "robot", None):
        overrides["robot.backend"] = args.robot
    if getattr(args, "cameras", None):
        overrides["cameras.backend"] = args.cameras
    if getattr(args, "host", None):
        overrides["robot.host"] = args.host
    if getattr(args, "port", None):
        overrides["robot.port"] = args.port
    if getattr(args, "script", None):
        overrides["astra.script_path"] = args.script
    if getattr(args, "planning", None) is not None:
        overrides["planning.enabled"] = args.planning
    if getattr(args, "planner_model", None):
        if getattr(args, "planning", None) is False:
            raise SystemExit("--planner-model cannot be combined with --no-planning")
        overrides["planning.enabled"] = True
    for attr, key in (("planner_model", "planning.planner.model"),
                      ("planner_backend", "planning.planner.backend"),
                      ("planner_effort", "planning.planner.reasoning_effort")):
        if getattr(args, attr, None) is not None:
            overrides[key] = getattr(args, attr)
    if getattr(args, "actions_only", None) is False:
        # --language-output is an explicit request for notes, so it also opts out of the planning-mode
        # default that keeps System 1 silent (see PlanningConfig.motion_actions_only).
        overrides["planning.motion_actions_only"] = False
    for attr, key in [("actions_only", "astra.actions_only"),
                      ("model", "astra.model"), ("effort", "astra.reasoning_effort"), ("max_calls", "limits.max_llm_calls"),
                      ("max_waypoints", "limits.max_waypoints"), ("max_seconds", "limits.max_trial_seconds"),
                      ("policy_notes", "policy_notes_path"),
                      ("image_history", "astra.image_history"), ("speed", "motion.linear_speed_mps"),
                      ("log_dir", "log_dir"), ("feedback_file", "operator_feedback_file")]:
        v = getattr(args, attr, None)
        if v is not None:
            overrides[key] = v
    if getattr(args, "shadow_model", None):
        overrides["shadow.enabled"] = True
        overrides["shadow.model"] = args.shadow_model[0]
        overrides["shadow.extra_models"] = list(args.shadow_model[1:])
    if getattr(args, "no_shadow", False):
        overrides["shadow.enabled"] = False
    if getattr(args, "fast", False):
        # setdefault: anything already named by --set or by an explicit flag (--speed) keeps its value
        for key, value in (("motion.linear_speed_mps", 0.05), ("motion.yaw_speed_rps", 0.6),
                           ("motion.gripper_speed_per_s", 1.5), ("motion.settle_seconds", 0.1)):
            overrides.setdefault(key, value)
    if getattr(args, "release_tilt", None) is not None:
        lim = round(abs(float(args.release_tilt)) * 3.141592653589793 / 180.0, 4)
        overrides["bounds.pitch"] = [-lim, lim]
        overrides["bounds.roll"] = [-lim, lim]
    if getattr(args, "scene", None):
        overrides["sim.scene"] = args.scene
    if getattr(args, "viser", False):
        overrides["viz.enabled"] = True
    if getattr(args, "viser_port", None):
        overrides["viz.port"] = args.viser_port
    if getattr(args, "viser_host", None):
        overrides["viz.host"] = args.viser_host
    if getattr(args, "no_viser_render", False):
        overrides["viz.render_observations"] = False
    if getattr(args, "no_home", False):
        overrides["robot.home_at_start"] = False
    if getattr(args, "home_on_end", False):
        overrides["home_on_end"] = True
    if getattr(args, "strict_gateway", False):
        overrides["limits.strict_gateway"] = True
    if getattr(args, "dynamic_scene", False):
        overrides["reactive.enabled"] = True
    return load_config(getattr(args, "config", None), overrides)


def _make_robot_and_cameras(cfg: PipelineConfig, kin: ArmKinematics):
    from utils.cameras import make_camera_source

    sim_world = None
    if cfg.robot.backend == "sim":
        from utils.sim import SimRobot, SimWorld

        sim_world = SimWorld(kin, table_z=cfg.sim.table_z, scene=cfg.sim.scene)
        q0 = np.zeros(14)
        q0[0:6], q0[7:13] = cfg.robot.home_joints_left, cfg.robot.home_joints_right
        q0[6] = q0[13] = cfg.robot.home_gripper
        robot = SimRobot(initial_q=q0, world=sim_world)
        sim_world.update(q0)
    elif cfg.robot.backend == "zmq":
        from utils.robot_interface import ZmqRobot

        robot = ZmqRobot(cfg.robot.host, cfg.robot.port, cfg.robot.zmq_timeout_ms, cfg.robot.gello_software_path)
    else:
        raise SystemExit(f"unknown robot backend {cfg.robot.backend}")
    if cfg.cameras.backend == "sim" and sim_world is None:
        raise SystemExit("--cameras sim requires --robot sim")
    try:
        cameras = make_camera_source(cfg.cameras, cfg.robot.gello_software_path, sim_world=sim_world)
    except BaseException:
        robot.close()
        raise
    return robot, cameras, sim_world


def _confirm_factory(yes: bool):
    if yes:
        return None
    if not sys.stdin.isatty():
        def _deny(msg: str) -> bool:
            print(f"[confirm] {msg} -> refusing: non-interactive session and --yes not given", flush=True)
            return False
        return _deny

    def _ask(msg: str) -> bool:
        ans = input(f"{msg} [y/N] ").strip().lower()
        return ans in ("y", "yes")
    return _ask


def _read_goals(path: str) -> List[str]:
    goals = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            goals.append(line)
    if not goals:
        raise SystemExit(f"no goals found in {path}")
    return goals


def cmd_run(args) -> int:
    from utils.astra_client import make_astra_client
    from utils.session import OperatorInput, TrialRunner

    cfg = _config_from_args(args)
    realtime = not (args.fast_sim and cfg.robot.backend == "sim")
    if cfg.robot.backend == "zmq" and cfg.astra.is_remote and not args.yes and not sys.stdin.isatty():
        raise SystemExit("real robot + real model in a non-interactive session requires --yes")
    print(dump_config(cfg))
    print(f"[embodiment] {cfg.embodiment_name}: {get_arm_model(cfg.embodiment_name).description}", flush=True)
    kin = ArmKinematics.from_config(cfg)
    tools = build_tools(cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled,
                        actions_only=cfg.astra.actions_only, max_waypoints=cfg.motion.max_waypoints_per_call)
    astra = make_astra_client(cfg.astra, tools)
    planner = None
    if cfg.planning.enabled:
        from utils.planning import planner_config
        planner = make_astra_client(planner_config(cfg), [])
        print(f"[planning] System 2 {planner.model} -> System 1 {astra.model} -> IK/gateway -> robot", flush=True)
    if cfg.shadow.enabled:
        from utils.shadow import ShadowClient, shadow_configs

        shadow_cfgs = shadow_configs(cfg)
        astra = ShadowClient(astra, [make_astra_client(c, tools) for c in shadow_cfgs],
                             wait_timeout_s=cfg.shadow.wait_timeout_s,
                             independent_history=cfg.shadow.independent_history)
        print(f"[shadow] {', '.join(c.model for c in shadow_cfgs)} answer every request alongside "
              f"{cfg.astra.model}; their actions are shown, never executed", flush=True)
    robot, cameras, sim_world = _make_robot_and_cameras(cfg, kin)
    operator = OperatorInput(use_stdin=sys.stdin.isatty(), file_path=cfg.operator_feedback_file)
    viz = None
    if cfg.viz.enabled:
        from utils.viser_ui import MirroredRobot, ViserCameraSource, ViserVisualizer

        viz = ViserVisualizer(cfg, kin, world=sim_world, mode="sim" if cfg.robot.backend == "sim" else "robot")
        if cfg.robot.backend == "sim":
            robot.on_command = viz.update_robot
            if cfg.viz.render_observations:
                cameras = ViserCameraSource(viz, fallback=cameras, names=list(cfg.cameras.names.values()))
        else:
            robot = MirroredRobot(robot, viz)
        viz.update_robot(robot.get_joint_positions(), force=True)
        operator.add_source(viz.poll_operator)
        print(f"\n[viser] open {viz.url()} for the 3D view and operator controls\n", flush=True)
        if cfg.robot.backend == "sim" and cfg.viz.render_observations and cfg.viz.wait_for_client_s > 0:
            import time as _time

            t0 = _time.perf_counter()
            while not viz.has_client() and _time.perf_counter() - t0 < cfg.viz.wait_for_client_s:
                _time.sleep(0.5)
            print(f"[viser] browser client connected: {viz.has_client()} "
                  f"({'3D renders' if viz.has_client() else 'schematic images'} will be sent to Astra)", flush=True)
    runner = TrialRunner(cfg, robot, cameras, kin, astra, operator=operator, realtime=realtime,
                         confirm=_confirm_factory(args.yes) if cfg.robot.backend == "zmq" else None,
                         sim_world=sim_world, hooks=viz, planner=planner)
    if viz is not None:
        viz.on_estop = runner.request_estop
        viz.on_reobserve = runner.request_reobserve
    goals = [args.goal] if args.goal else _read_goals(args.goals_file)
    rc = 0
    try:
        for i, goal in enumerate(goals):
            if viz is not None and cfg.viz.wait_for_start and not args.yes:
                print(f"[viser] press 'Start trial' in the browser (or Enter here) to begin: {goal}", flush=True)
                viz.wait_for_start()
            elif i > 0 and sys.stdin.isatty() and not args.yes:
                ans = input("\nReset the scene, then press Enter to start the next goal (q to quit): ").strip().lower()
                if ans == "q":
                    break
            if i > 0 and sim_world is not None:
                sim_world.reset_objects()      # each goal starts from the preset scene
                if viz is not None:
                    viz.update_objects()
            outcome = runner.run(goal)
            print(json.dumps(outcome.as_dict(), indent=2))
            if outcome.status in ("error", "aborted", "estop"):
                rc = 1
                break
        if viz is not None and sys.stdin.isatty() and not args.yes:
            input("[viser] trial(s) finished - press Enter to close the visualizer ")
    finally:
        operator.close()
        cameras.close()
        robot.close()
        if viz is not None:
            viz.close()
        for client in (astra, planner):
            close = getattr(client, "close", None)
            if callable(close):
                close()
    return rc


def cmd_viz(args) -> int:
    """Simulated arms in the browser with gizmo targets that run through the real gateway (no Astra)."""
    from utils.viser_ui import ViserVisualizer, run_manual_bench

    args.sim = True
    cfg = _config_from_args(args)
    cfg.viz.enabled = True
    kin = ArmKinematics.from_config(cfg)
    robot, cameras, sim_world = _make_robot_and_cameras(cfg, kin)
    viz = ViserVisualizer(cfg, kin, world=sim_world, mode="sim")
    robot.on_command = viz.update_robot
    try:
        run_manual_bench(cfg, kin, sim_world, robot, viz)
    finally:
        viz.close()
    return 0


def cmd_check(args) -> int:
    """Independent checks: API key, robot server, cameras. Nothing moves."""
    cfg = _config_from_args(args)
    ok = True
    print(f"config:\n{dump_config(cfg)}")
    model = get_arm_model(cfg.embodiment_name)
    print(f"[embodiment] {model.name}: {model.description}")
    kin = ArmKinematics.from_config(cfg, limit_margin=0.0)

    # 1) API key
    try:
        from utils.astra_client import find_api_key

        envs = [cfg.astra.api_key_env] if cfg.astra.is_remote else []
        if cfg.planning.enabled and cfg.planning.planner.is_remote:
            envs.append(cfg.planning.planner.api_key_env)
        if cfg.shadow.enabled:
            from utils.shadow import shadow_configs

            for sc in shadow_configs(cfg):
                if sc.is_remote and sc.api_key_env not in envs:
                    envs.append(sc.api_key_env)
        for env_name in dict.fromkeys(envs):
            key, source = find_api_key(env_name)
            if not key:
                raise RuntimeError(f"{env_name} not found (environment, .env, .secrets/)")
            print(f"[ok] {env_name} found in {source}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[!!] {e}")

    # 2) robot
    robot = None
    sim_world = None
    try:
        if cfg.robot.backend == "sim":
            from utils.sim import SimRobot, SimWorld

            sim_world = SimWorld(kin, table_z=cfg.sim.table_z)
            q0 = np.zeros(14)
            q0[0:6], q0[7:13] = cfg.robot.home_joints_left, cfg.robot.home_joints_right
            q0[6] = q0[13] = cfg.robot.home_gripper
            robot = SimRobot(initial_q=q0, world=sim_world)
            sim_world.update(q0)
        else:
            from utils.robot_interface import ZmqRobot

            print(f"[..] connecting to robot server tcp://{cfg.robot.host}:{cfg.robot.port} ...", flush=True)
            robot = ZmqRobot(cfg.robot.host, cfg.robot.port, cfg.robot.zmq_timeout_ms, cfg.robot.gello_software_path)
        q = robot.get_joint_positions()
        start_rot = start_rotations(q, kin)
        eef = eef_state_dict(arm_poses(q, kin, start_rot))
        print(f"[ok] robot ({cfg.robot.backend}) joint_pos = {np.round(q, 4).tolist()}")
        print(f"     eef (yaw/pitch/roll relative to *this* pose): {format_eef_state(eef)}")
        for arm in ARMS:
            for dim in ("x", "y", "z"):
                lo, hi = cfg.bounds.for_dim(dim)
                v = eef[f"{arm}_{dim}"]
                if not lo <= v <= hi:
                    print(f"[..] {arm}_{dim}={v:.3f} is outside the configured bounds [{lo}, {hi}] "
                          f"(fine before homing; the home pose is inside)")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[!!] robot check failed: {type(e).__name__}: {e}")
        if cfg.robot.backend == "zmq":
            print(f"     start the robot server first:  {model.robot_server_hint}")
    finally:
        if robot is not None:
            robot.close()

    # 3) cameras
    cameras = None
    try:
        from utils.cameras import make_camera_source

        if cfg.cameras.backend == "sim" and sim_world is None:
            raise RuntimeError("camera backend 'sim' needs robot backend 'sim'")
        print(f"[..] opening cameras ({cfg.cameras.backend}) ...", flush=True)
        cameras = make_camera_source(cfg.cameras, cfg.robot.gello_software_path, sim_world=sim_world)
        frames = cameras.read_jpeg_frames()
        print(f"[ok] cameras ({cfg.cameras.backend}): {[(k, len(v)) for k, v in frames.items()]} bytes")
        if args.save_frames and frames:
            out = Path(cfg.log_dir) / "check_frames"
            out.mkdir(parents=True, exist_ok=True)
            for k, v in frames.items():
                (out / f"{k}.jpg").write_bytes(v)
            print(f"     saved to {out}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[!!] camera check failed: {type(e).__name__}: {e}")
    finally:
        if cameras is not None:
            cameras.close()

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def cmd_sim_server(args) -> int:
    from utils.sim import SimRobot, SimWorld, serve_sim_zmq

    model = get_arm_model(args.embodiment)
    kin = ArmKinematics(model=model)
    q0 = model.home_q14()
    world = SimWorld(kin)
    robot = SimRobot(initial_q=q0, world=world)
    world.update(q0)
    try:
        serve_sim_zmq(robot, args.host, args.port)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_workspace(args) -> int:
    from utils.workspace import report

    cfg = _config_from_args(args)
    print(report(cfg.viz.urdf_path, samples=args.samples, arm=args.arm,
                 reference_joints=cfg.robot.home_joints_left if args.arm == "left" else cfg.robot.home_joints_right,
                 model=get_arm_model(cfg.embodiment_name)))
    return 0


def cmd_show_prompt(args) -> int:
    cfg = _config_from_args(args)
    tools = build_tools(cfg.bounds, cfg.prompts_path, reactive=cfg.reactive.enabled,
                        actions_only=cfg.astra.actions_only, max_waypoints=cfg.motion.max_waypoints_per_call)
    if args.json:
        print(json.dumps(tools, indent=2))
        return 0
    system_prompt = build_policy_prompt(cfg)
    bundle = {"system_prompt": system_prompt, "tools": tools}
    if cfg.planning.enabled:
        from utils.planning import build_planner_prompt
        bundle["planner_system_prompt"] = build_planner_prompt(cfg)
        bundle["planner_tools"] = []
    if args.bundle_json:
        print(json.dumps(bundle, indent=2))
        return 0
    if cfg.planning.enabled:
        print("=== SYSTEM 2 PLANNER PROMPT (text only, no tools) ===")
        print(bundle["planner_system_prompt"])
    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== TOOLS ===")
    print(json.dumps(tools, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    rc = {"run": cmd_run, "check": cmd_check, "sim-server": cmd_sim_server, "show-prompt": cmd_show_prompt,
          "viz": cmd_viz, "workspace": cmd_workspace}[args.cmd](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
