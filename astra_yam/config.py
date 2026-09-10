"""Configuration dataclasses + YAML loading for the Astra <-> YAM pipeline."""
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_YAM_XML = str(ASSETS_DIR / "yam.xml")
DEFAULT_GELLO_SOFTWARE = os.environ.get(
    "GELLO_SOFTWARE_PATH", "/home/skild/bimanual_manipulation/skild-gello/gello_software"
)
DEFAULT_STATION_CONFIG = "/home/skild/bimanual_manipulation/metadata/station_config.json"
DEFAULT_SYSTEM_PROMPT = str(REPO_ROOT / "configs" / "SYSTEM_PROMPT.md")
DEFAULT_TILT_NOTE = str(REPO_ROOT / "configs" / "TILT_NOTE.md")
DEFAULT_PROMPTS = str(REPO_ROOT / "configs" / "PROMPTS.yaml")

# ---------------------------------------------------------------------------
# Embodiment constants (bimanual YAM, gello layout)
# ---------------------------------------------------------------------------
ARMS = ("left", "right")
ARM_DIMS = ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
DIM_NAMES = tuple(f"{arm}_{dim}" for arm in ARMS for dim in ARM_DIMS)

ARM_JOINTS = 6          # revolute joints per arm
DOFS_PER_ARM = 7        # 6 joints + 1 normalized gripper (0 = closed, 1 = open)
NUM_DOFS = 14           # [left_arm(6) | left_gripper | right_arm(6) | right_gripper]
LEFT_SLICE = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_SLICE = slice(7, 13)
RIGHT_GRIPPER = 13

# Same as gello.robots.robot_constants.JOINTS_{LOWER,UPPER}_LIMIT_YAM (arm joints only).
YAM_JOINT_LOWER = (-2.617, 0.0, 0.0, -1.57, -1.57, -2.09)
YAM_JOINT_UPPER = (3.13, 3.65, 3.13, 1.57, 1.57, 2.09)

# Exact revolute limits of skild_yam_v2.urdf (astra_yam.workspace.urdf_joint_limits), i.e. gello's rounded
# constants above widened back to what the driver really accepts. Both IK and the robot driver clamp here.
URDF_JOINT_LOWER = (-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944)
URDF_JOINT_UPPER = (3.13, 3.65, 3.13, 1.5708, 1.5708, 2.0944)

# Start pose used by the reference BluPe trials (grasp point ~ (0.297, 0.000, 0.186), tool 58 deg down).
REFERENCE_HOME_JOINTS = (-0.02, 0.7967, 0.6167, -0.3756, -0.0204, -0.0078)


@dataclass
class Bounds:
    """Per-arm Cartesian bounds in the arm's own base frame (meters / radians).

    Equal lower/upper bounds pin an axis: targets must equal that value.

    Defaults are the robot's own limits, not hand-tuned numbers: the axis-aligned envelope of the grasp point
    over the full URDF joint box (`python -m astra_yam workspace`, max reach 0.866 m, rounded outward to the
    millimetre) and the full range the reported extrinsic xyz Euler angles can represent (the convention caps
    pitch at +/-pi/2; yaw and roll span +/-pi). This is a bounding box, not the reachable set: the gateway still
    rejects unreachable targets via IK, joint limits, configuration flips, arm clearance and tracking error.
    Note that z reaches 0.516 m below the base plane, so nothing here stops a command into the table - set
    `motion.tool_floor_z_m` for that.
    """

    x: Tuple[float, float] = (-0.777, 0.777)
    y: Tuple[float, float] = (-0.771, 0.772)
    z: Tuple[float, float] = (-0.516, 0.864)
    yaw: Tuple[float, float] = (-3.1416, 3.1416)
    pitch: Tuple[float, float] = (-1.5708, 1.5708)
    roll: Tuple[float, float] = (-3.1416, 3.1416)
    gripper: Tuple[float, float] = (0.0, 1.0)

    def for_dim(self, dim: str) -> Tuple[float, float]:
        lo, hi = getattr(self, dim)
        return float(lo), float(hi)

    def is_pinned(self, dim: str) -> bool:
        lo, hi = self.for_dim(dim)
        return lo == hi


def reference_bounds() -> Bounds:
    """The narrow, tilt-pinned bounds of the reference BluPe trials (`prompts/000*.json`).

    Kept so the reference prompt/tool text can be reproduced byte-for-byte; the defaults above are the full
    robot envelope instead.
    """
    return Bounds(
        x=(0.15, 0.48),
        y=(-0.25, 0.25),
        z=(0.03, 0.40),
        yaw=(-3.142, 3.142),
        pitch=(0.0, 0.0),
        roll=(0.0, 0.0),
    )


@dataclass
class MotionConfig:
    cadence_hz: float = 10.0            # Cartesian waypoint cadence reported to the model
    control_hz: float = 30.0            # rate at which joint commands are streamed to the robot
    linear_speed_mps: float = 0.01      # reference behaviour: ~52 waypoints for a 5.5 cm move
    yaw_speed_rps: float = 0.15
    gripper_speed_per_s: float = 0.5    # normalized gripper units per second
    max_joint_step_rad: float = 0.05    # per waypoint; larger joint steps are subdivided ("pacing")
    max_ik_joint_jump_rad: float = 0.35 # consecutive-waypoint jump => configuration flip => reject
    ik_pos_tol_m: float = 0.002
    ik_ori_tol_rad: float = 0.02
    tracking_abort_rad: float = 0.25    # measured-vs-commanded arm-joint error that aborts the session
    tracking_check_every: int = 5       # waypoints between tracking checks
    settle_seconds: float = 0.3         # wait after a motion before observing
    homing_seconds: float = 5.0         # duration of the joint-space move to the home pose
    joint_limit_margin_rad: float = 0.01
    tool_floor_z_m: Optional[float] = None   # z the tilt guard treats as the table (None = use bounds.z lower).
                                             # Set it when bounds.z is opened to the full reach envelope and you
                                             # still want the jaw tips kept above the table.
    arm_clearance_m: float = 0.02       # clearance required between arm links / gripper housings of the two arms;
                                        # fingers and gripper necks only need arm_tip_clearance_m (0 disables the check)
    arm_tip_clearance_m: float = 0.005
    detour_enabled: bool = True         # if the straight path would clip the other arm, try over/side detours
    detour_rise_m: float = 0.12         # height added above the higher of start/goal for the "over" detour
    detour_side_m: float = 0.08         # lateral offset away from the other arm for the "side" detour
    gripper_init_squeeze: float = 0.15  # no-home start with a partly closed gripper: command (measured - this) so a
                                        # held object stays squeezed instead of the jaws relaxing to the stall value


@dataclass
class RobotConfig:
    backend: str = "zmq"                # "zmq" = gello robot server (launch_nodes.py), "sim" = kinematic simulator
    host: str = "127.0.0.1"
    port: int = 6001
    zmq_timeout_ms: int = 2000
    gello_software_path: str = DEFAULT_GELLO_SOFTWARE
    yam_xml_path: str = DEFAULT_YAM_XML
    home_at_start: bool = True
    home_joints_left: List[float] = field(default_factory=lambda: list(REFERENCE_HOME_JOINTS))
    home_joints_right: List[float] = field(default_factory=lambda: list(REFERENCE_HOME_JOINTS))
    home_gripper: float = 1.0
    joint_lower: List[float] = field(default_factory=lambda: list(URDF_JOINT_LOWER))
    joint_upper: List[float] = field(default_factory=lambda: list(URDF_JOINT_UPPER))
    max_homing_joint_delta_rad: float = 3.14  # refuse homing if any joint must move further than this


@dataclass
class CameraConfig:
    backend: str = "realsense"          # "realsense" | "sim" | "none"
    station_config_path: str = DEFAULT_STATION_CONFIG
    # station camera name -> name shown to the model (reference trials use top_cam/left_cam/right_cam)
    names: Dict[str, str] = field(
        default_factory=lambda: {
            "top_camera": "top_cam",
            "left_wrist_camera": "left_cam",
            "right_wrist_camera": "right_cam",
        }
    )
    hz: int = 30
    jpeg_quality: int = 85
    reset_on_start: bool = True         # hardware-reset RealSense devices before opening (gello convention)
    detail: str = "high"                # OpenAI image detail level
    warmup_seconds: float = 1.5


@dataclass
class AstraConfig:
    backend: str = "openai"             # "openai" | "scripted"
    model: str = "gpt-6-astra"
    reasoning_effort: Optional[str] = None   # low | medium | high | xhigh | max (None = API default)
    reasoning_summary: Optional[str] = "auto"  # auto | concise | detailed | None; the readable reasoning trace
                                               # written to transcript.txt (dropped automatically if rejected)
    tool_choice: str = "required"
    parallel_tool_calls: bool = False
    request_timeout_s: float = 300.0
    max_retries: int = 4
    max_output_tokens: Optional[int] = None
    image_history: int = 4              # images are kept for the newest N..2N observations (0 = keep every image,
                                        # which makes every request carry the whole run: 4.9 MB and ~140k image
                                        # tokens by turn 60, re-uploaded and re-prefilled every call)
    prompt_cache_key: Optional[str] = "astra-yam"   # routes requests sharing our system prompt + tools to the
                                                    # same prompt cache (None = let the provider decide)
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    script_path: Optional[str] = None   # scripted backend: JSON list of {"name":..., "arguments":{...}}


@dataclass
class LimitsConfig:
    max_llm_calls: int = 100
    max_waypoints: int = 10000
    max_trial_seconds: float = 1800.0
    max_consecutive_rejections: int = 5
    strict_gateway: bool = False        # True: the first rejected packet ends the session (reference semantics)
    prompt_llm_calls: Optional[int] = None  # budget announced in the system prompt (None = max_llm_calls);
                                            # set it higher than max_llm_calls for short smoke tests, otherwise
                                            # Astra rightly gives up when told it has only 2 calls


@dataclass
class VizConfig:
    """viser 3D visualization + operator UI (python -m astra_yam run --viser)."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    urdf_path: str = str(REPO_ROOT / "configs" / "skild_yam_v2.urdf")   # bimanual kinematics URDF (left at origin, right at y=-0.61)
    mesh_dir: Optional[str] = None      # default: <gello_software>/third_party/robot_models/yam (yam.xml + assets/*.stl)
    render_observations: bool = True    # sim only: the agent's camera images are rendered by the connected browser
    render_width: int = 640
    render_height: int = 480
    render_timeout_s: float = 8.0
    update_hz: float = 30.0             # max rate of robot pose updates pushed to the browser
    wait_for_start: bool = True         # wait for the Start button (or Enter) before each goal when interactive
    wait_for_client_s: float = 0.0      # with render_observations: wait up to this long for a browser before starting
    show_labels: bool = False           # 3D text labels on objects/bases; they leak into agent renders unless removed
    label_settle_s: float = 0.6         # with labels on: wait this long after removing them before capturing renders


@dataclass
class SimConfig:
    scene: str = "blocks"               # object preset for the simulator: blocks | kitchen | empty
    table_z: float = 0.0                # table surface height in the arm base frame


@dataclass
class PipelineConfig:
    embodiment_name: str = "yam_arms"
    bounds: Bounds = field(default_factory=Bounds)
    motion: MotionConfig = field(default_factory=MotionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    astra: AstraConfig = field(default_factory=AstraConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    log_dir: str = str(REPO_ROOT / "runs")
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT
    tilt_note_path: str = DEFAULT_TILT_NOTE   # appended to the system prompt when tilt is actuated
    prompts_path: str = DEFAULT_PROMPTS       # tool descriptions, observation and session messages
    home_on_end: bool = False
    operator_feedback_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------
def _build(cls, data: Any):
    """Recursively build a dataclass from a (possibly partial) dict."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected mapping for {cls.__name__}, got {type(data).__name__}")
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = known[key].type
        target = _dataclass_for_field(cls, key)
        if target is not None:
            kwargs[key] = _build(target, value)
        elif isinstance(value, list) and _is_tuple_field(ftype):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _dataclass_for_field(cls, name: str):
    """Return the nested dataclass type of field `name`, or None for plain values."""
    factory = cls.__dataclass_fields__[name].default_factory  # type: ignore[attr-defined]
    if factory is MISSING:
        return None
    probe = factory()
    return type(probe) if is_dataclass(probe) else None


def _is_tuple_field(ftype: Any) -> bool:
    return "Tuple" in str(ftype) or "tuple" in str(ftype)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load a YAML config (optional) and apply dotted overrides such as {"astra.model": "gpt-6-astra"}."""
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return _build(PipelineConfig, data)


def dump_config(cfg: PipelineConfig) -> str:
    return yaml.safe_dump(to_dict(cfg), sort_keys=False)
