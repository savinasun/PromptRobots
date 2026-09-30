"""Configuration dataclasses + YAML loading for the Astra <-> robot pipeline (bimanual YAM or UR5e).

Embodiment-specific defaults (MJCF, joint box, home pose, prompt files, bounds) come from `utils.arm_models`;
`embodiment_name` selects the rig and `PipelineConfig.__post_init__` fills in whatever the YAML left unset."""
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
DEFAULT_SYSTEM_PROMPT = str(REPO_ROOT / "docs" / "SYSTEM_PROMPT.md")
DEFAULT_TILT_NOTE = str(REPO_ROOT / "docs" / "TILT_NOTE.md")
DEFAULT_PROMPTS = str(REPO_ROOT / "configs" / "PROMPTS.yaml")
DEFAULT_PLANNER_PROMPT = str(REPO_ROOT / "docs" / "PLANNER_PROMPT.md")
DEFAULT_MOTION_PROMPT = str(REPO_ROOT / "docs" / "MOTION_PROMPT.md")

# ---------------------------------------------------------------------------
# Embodiment constants shared by every rig (14-DoF gello layout); the YAM numbers below are that rig's defaults
# and the reference values several tests compare against. Other rigs live in utils/arm_models.py.
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

# Exact revolute limits of skild_yam_v2.urdf (utils.workspace.urdf_joint_limits), i.e. gello's rounded
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
    over the full URDF joint box (`python -m utils workspace`, max reach 0.866 m, rounded outward to the
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
    max_waypoints_per_call: int = 1     # >1 lets one move_to carry a sequence of Cartesian waypoints executed as
                                        # consecutive straight segments (whole path validated before any motion),
                                        # so one inference call buys a whole stroke instead of one leg of it.
                                        # 1 keeps the reference single-target schema exactly as it was.
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
    # None = the selected embodiment's own value (utils/arm_models.py); PipelineConfig.__post_init__ fills them in.
    mjcf_path: Optional[str] = None             # single-arm MuJoCo model for FK/IK (yam.xml / ur5e.xml)
    home_at_start: bool = True
    home_joints_left: Optional[List[float]] = None
    home_joints_right: Optional[List[float]] = None
    home_gripper: float = 1.0
    joint_lower: Optional[List[float]] = None   # exact URDF limits; both IK and the robot driver clamp here
    joint_upper: Optional[List[float]] = None
    max_homing_joint_delta_rad: float = 3.14  # refuse homing if any joint must move further than this

    @property
    def yam_xml_path(self) -> Optional[str]:
        """Old name of `mjcf_path`, kept for callers written against the YAM-only pipeline."""
        return self.mjcf_path


@dataclass
class CameraConfig:
    backend: str = "realsense"          # "station" | "realsense" | "sim" | "none"
                                        # "station": open whatever camera_ids declares (RealSense and/or
                                        # ZED) through gello's initialize_cameras; "realsense": open local
                                        # RealSense devices by serial directly
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


# Remote model backends and their defaults. "openai" speaks the OpenAI Responses API (gpt-6-astra);
# "openrouter" speaks OpenAI-compatible Chat Completions through OpenRouter (any tool-capable model there).
REMOTE_BACKENDS = ("openai", "openrouter")
BACKEND_DEFAULTS = {
    "openai": {"model": "gpt-6-astra", "api_key_env": "OPENAI_API_KEY", "base_url": None},
    "openrouter": {"model": "qwen/qwen3.8-max-0902", "api_key_env": "OPENROUTER_API_KEY",
                   "base_url": "https://openrouter.ai/api/v1"},
}


@dataclass
class AstraConfig:
    backend: str = "openrouter"         # "openrouter" | "openai" | "scripted"
    model: Optional[str] = None         # None = the backend's default (see BACKEND_DEFAULTS)
    actions_only: bool = True           # strict robot actions; low effort, one required tool; summaries allowed
    reasoning_effort: Optional[str] = None   # low | medium | high | xhigh | max (None = API default)
    reasoning_summary: Optional[str] = "auto"  # auto | concise | detailed | None; the readable reasoning trace
                                               # written to transcript.txt (dropped automatically if rejected;
                                               # openrouter returns the model's reasoning text directly instead)
    tool_choice: str = "required"
    parallel_tool_calls: bool = False
    request_timeout_s: float = 300.0
    max_retries: int = 4
    max_output_tokens: Optional[int] = None
    image_history: int = 4              # images are kept for the newest N..2N observations (0 = keep every image,
                                        # which makes every request carry the whole run: 4.9 MB and ~140k image
                                        # tokens by turn 60, re-uploaded and re-prefilled every call)
    prompt_cache_key: Optional[str] = "astra-yam"   # routes requests sharing our system prompt + tools to the
                                                    # same prompt cache (None = let the provider decide; openai only)
    api_key_env: Optional[str] = None   # None = the backend's default (OPENAI_API_KEY / OPENROUTER_API_KEY)
    base_url: Optional[str] = None      # None = the backend's default endpoint
    script_path: Optional[str] = None   # scripted backend: JSON list of {"name":..., "arguments":{...}}
    openrouter_providers: List[str] = field(default_factory=list)   # openrouter: restrict routing to these
                                                                    # providers (empty = OpenRouter's default routing)

    def __post_init__(self) -> None:
        defaults = BACKEND_DEFAULTS.get(self.backend, BACKEND_DEFAULTS["openai"])
        if self.model is None:
            self.model = "scripted-astra" if self.backend == "scripted" else defaults["model"]
        if self.api_key_env is None:
            self.api_key_env = defaults["api_key_env"]
        if self.base_url is None:
            self.base_url = defaults["base_url"]

    @property
    def is_remote(self) -> bool:
        """True when a real model (and an API key) is behind this backend."""
        return self.backend in REMOTE_BACKENDS


@dataclass
class ShadowConfig(AstraConfig):
    """A second model queried with the identical request every turn, for side-by-side comparison.

    Its proposed action and note are printed and logged next to the controlling model's; nothing it returns is
    executed or fed back into the conversation. Tools and language/action mode always follow `astra`.
    """

    enabled: bool = False
    extra_models: List[Any] = field(default_factory=list)   # more shadows, each queried in parallel: a model ID
                                                            # (same backend/settings) or a mapping of AstraConfig
                                                            # fields, e.g. {backend: openai, model: gpt-6-astra}
    independent_history: bool = True    # each shadow sees its own earlier proposals (never the controlling model's
                                        # notes) plus a "not executed" result naming the targets actually run;
                                        # False = shadows get the controlling model's exact request
    wait_timeout_s: float = 60.0        # how long a turn waits for the shadow answer after the controlling model
                                        # has replied (a late answer is dropped, the robot is never held longer)


@dataclass
class PlannerConfig(AstraConfig):
    actions_only: bool = False
    tool_choice: str = "none"
    reasoning_effort: Optional[str] = "high"


@dataclass
class PlanningConfig:
    """System 2 produces text; `astra` is the System 1 motion model."""

    enabled: bool = False
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    system_prompt_path: str = DEFAULT_PLANNER_PROMPT
    motion_prompt_path: str = DEFAULT_MOTION_PROMPT
    motion_images: bool = False         # System 2 owns perception; System 1 gets text + robot state by default
    motion_calls_per_plan: int = 1      # System 1 turns (each with a fresh observation) per System 2 plan. The
                                        # planner is the slowest step in the loop by far, so >1 amortizes it over
                                        # more arm motion. A rejection, operator feedback or a stale observation
                                        # still calls the planner back early.
    speculative_motion: bool = True     # dispatch the motion request the moment System 2's instruction block is
                                        # complete in the stream, hiding its time-to-first-token under the
                                        # planner's remaining tokens. Inert unless the planner emits the markers.
    motion_actions_only: bool = True    # System 1 emits actions only: no notes, summaries or hindsight. Scene
                                        # assessment and narration are System 2's job and restating them costs
                                        # output tokens on the critical path.


@dataclass
class LimitsConfig:
    max_llm_calls: int = 100
    max_waypoints: int = 10000
    max_trial_seconds: float = 1800.0
    max_consecutive_rejections: int = 5
    strict_gateway: bool = False        # True: the first rejected packet ends the session (reference semantics)


@dataclass
class VizConfig:
    """viser 3D visualization + operator UI (python -m utils run --viser)."""

    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    urdf_path: Optional[str] = None     # bimanual kinematics URDF; None = the embodiment's (skild_yam_v2.urdf / skild_ur5e.urdf)
    mesh_dir: Optional[str] = None      # visual meshes; None = the embodiment's default (YAM: <gello_software>/third_party/
                                        # robot_models/yam; UR5e: the folder holding the URDF's assets/ + robotiq_meshes/)
    render_observations: bool = True    # sim only: the agent's camera images are rendered by the connected browser
    render_width: int = 640
    render_height: int = 480
    render_timeout_s: float = 8.0
    render_retry_s: float = 30.0        # cooldown for a browser that fails to return a frame
    update_hz: float = 30.0             # max rate of robot pose updates pushed to the browser
    wait_for_start: bool = True         # wait for the Start button (or Enter) before each goal when interactive
    wait_for_client_s: float = 0.0      # with render_observations: wait up to this long for a browser before starting
    show_labels: bool = False           # 3D text labels on objects/bases; they leak into agent renders unless removed
    label_settle_s: float = 0.6         # with labels on: wait this long after removing them before capturing renders


@dataclass
class SimConfig:
    scene: str = "blocks"               # object preset for the simulator: blocks | kitchen | empty
    table_z: Optional[float] = None     # table surface height in the left arm's frame; None = where the rig's URDF
                                        # root sits (YAM bases stand on the table: 0.0; UR5e bases are 0.113 m above it)


@dataclass
class ReactiveConfig:
    enabled: bool = False
    max_motion_seconds: float = 3.0  # executed trajectory time between observations
    camera_name: str = "top_cam"    # fixed camera for pre-execution scene-change checks
    change_fraction: float = 0.01   # fraction of blurred pixels with a material color change
    pixel_difference: float = 25.0  # 8-bit color difference, after downsampling


@dataclass
class PipelineConfig:
    embodiment_name: str = "yam_arms"
    bounds: Bounds = field(default_factory=Bounds)
    motion: MotionConfig = field(default_factory=MotionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    astra: AstraConfig = field(default_factory=AstraConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    log_dir: str = str(REPO_ROOT / "runs")
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT
    tilt_note_path: str = DEFAULT_TILT_NOTE   # appended to the system prompt when tilt is actuated
    prompts_path: str = DEFAULT_PROMPTS       # tool descriptions, observation and session messages
    home_on_end: bool = False
    operator_feedback_file: Optional[str] = None
    policy_notes_path: Optional[str] = None  # versioned, task-specific advice appended to the fixed system prompt
    reactive: ReactiveConfig = field(default_factory=ReactiveConfig)

    def __post_init__(self) -> None:
        # One writer for the System 1 output contract, so every construction path (YAML, CLI, the Inspect
        # adapter, a hand-built config) agrees. Opt out with planning.motion_actions_only=false.
        if self.planning.enabled and self.planning.motion_actions_only:
            self.astra.actions_only = True
        self.resolve_embodiment()

    def resolve_embodiment(self) -> None:
        """Fill every embodiment-dependent field the YAML/CLI left unset from the selected arm model.

        `bounds` and the prompt paths default to the YAM values at the dataclass level (the reference trials);
        they are swapped for the model's own only while still untouched, so a YAML that spells them out wins.
        """
        from utils.arm_models import get_arm_model

        model = get_arm_model(self.embodiment_name)
        r = self.robot
        if r.mjcf_path is None:
            r.mjcf_path = model.mjcf_path
        if r.home_joints_left is None:
            r.home_joints_left = list(model.home_joints["left"])
        if r.home_joints_right is None:
            r.home_joints_right = list(model.home_joints["right"])
        if r.joint_lower is None:
            r.joint_lower = list(model.joint_lower)
        if r.joint_upper is None:
            r.joint_upper = list(model.joint_upper)
        if self.viz.urdf_path is None:
            self.viz.urdf_path = model.urdf_path
        if self.sim.table_z is None:
            self.sim.table_z = -float(model.base_pos["left"][2])
        yam = get_arm_model(None)
        if model is not yam:
            if self.bounds == Bounds():
                self.bounds = model.default_bounds
            if self.system_prompt_path == yam.system_prompt_path:
                self.system_prompt_path = model.system_prompt_path
            if self.tilt_note_path == yam.tilt_note_path:
                self.tilt_note_path = model.tilt_note_path
            if self.astra.prompt_cache_key == yam.prompt_cache_key:
                self.astra.prompt_cache_key = model.prompt_cache_key


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
    """Load a YAML config (optional) and apply dotted overrides such as {"astra.model": "qwen/qwen3.8-max-0902"}."""
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
    if (data.get("planning") or {}).get("enabled"):
        # With no explicit motion backend, planning mode uses Astra's Responses API.
        data.setdefault("astra", {}).setdefault("backend", "openai")
    cfg = _build(PipelineConfig, data)
    # `resolve_embodiment` can only guess whether a dataclass-default value was "left unset"; here we know which
    # keys the YAML/CLI actually gave, so those win even when they happen to equal the YAM defaults.
    from dataclasses import replace

    from utils.arm_models import get_arm_model

    model = get_arm_model(cfg.embodiment_name)
    if "bounds" in data and data["bounds"]:
        cfg.bounds = replace(model.default_bounds, **{k: tuple(v) for k, v in data["bounds"].items()})
    for key in ("system_prompt_path", "tilt_note_path"):
        if data.get(key) is not None:
            setattr(cfg, key, data[key])
    if (data.get("astra") or {}).get("prompt_cache_key") is not None:
        cfg.astra.prompt_cache_key = data["astra"]["prompt_cache_key"]
    return cfg


def dump_config(cfg: PipelineConfig) -> str:
    return yaml.safe_dump(to_dict(cfg), sort_keys=False)
