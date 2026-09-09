"""Embodiment description: tool schemas, system prompt, and eef-state computation for `yam_arms`.

The text itself lives in `configs/` (`PROMPTS.yaml`, `SYSTEM_PROMPT.md`, `TILT_NOTE.md`); this module only
assembles it. The tool texts reproduce the reference trial format (see `000*_example_input.json`) so that
transcripts from this pipeline are interchangeable with the reference ones.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from astra_yam.config import (
    ARMS,
    DEFAULT_TILT_NOTE,
    ARM_DIMS,
    DIM_NAMES,
    LEFT_GRIPPER,
    LEFT_SLICE,
    NUM_DOFS,
    RIGHT_GRIPPER,
    RIGHT_SLICE,
    Bounds,
)
from astra_yam.kinematics import ArmKinematics, relative_ypr, unwrap_angle
from astra_yam.prompts import prompt

ARM_SLICES = {"left": LEFT_SLICE, "right": RIGHT_SLICE}
ARM_GRIPPER_INDEX = {"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER}


def _fmt_bound(v: float) -> str:
    v = float(v)
    if v.is_integer():
        return str(int(v))
    return repr(round(v, 4))


def bounds_text(bounds: Bounds) -> str:
    """'left_x: [0.15, 0.48], left_y: [-0.25, 0.25], ...' exactly like the reference prompt."""
    parts = []
    for arm in ARMS:
        for dim in ARM_DIMS:
            lo, hi = bounds.for_dim(dim)
            parts.append(f"{arm}_{dim}: [{_fmt_bound(lo)}, {_fmt_bound(hi)}]")
    return ", ".join(parts)


def build_tools(bounds: Bounds, prompts_path: Optional[str] = None) -> List[dict]:
    """The three tool schemas, with every description taken from `configs/PROMPTS.yaml`."""
    def text(key: str, **fmt) -> str:
        return prompt(key, prompts_path, **fmt)

    hindsight = text("tools.hindsight")
    return [
        {
            "type": "function",
            "name": "move_to",
            "description": text("tools.move_to.description", bounds=bounds_text(bounds)),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": text("tools.move_to.targets", dim_names=", ".join(DIM_NAMES)),
                    },
                    "note": {"type": "string", "description": text("tools.move_to.note")},
                },
                "required": ["targets", "note"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "done",
            "description": text("tools.done.description"),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "hindsight": {"type": "string", "description": hindsight},
                },
                "required": ["summary", "hindsight"],
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "give_up",
            "description": text("tools.give_up.description"),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "hindsight": {"type": "string", "description": hindsight},
                },
                "required": ["reason", "hindsight"],
            },
            "strict": False,
        },
    ]


def tilt_note(path: Optional[str] = None) -> str:
    """The tilt/clearance paragraphs appended to the system prompt when pitch and roll are actuated.

    Text lives in `configs/TILT_NOTE.md` (see `PipelineConfig.tilt_note_path`) so it can be edited without
    touching the code, like the system prompt itself.
    """
    text = Path(path or DEFAULT_TILT_NOTE).read_text().strip("\n")
    return "\n\n" + text          # blank line: it reads as its own section, not a continuation of the last one


def build_system_prompt(path: str, max_llm_calls: int, embodiment_name: str = "yam_arms",
                        bounds: Optional[Bounds] = None, tilt_note_path: Optional[str] = None) -> str:
    """Load configs/SYSTEM_PROMPT.md and inject the configured LLM-call budget and embodiment name.

    When pitch/roll are not pinned in `bounds`, `configs/TILT_NOTE.md` is appended.
    """
    text = Path(path).read_text().rstrip("\n")
    text = re.sub(r"budget of \d+ LLM calls", f"budget of {int(max_llm_calls)} LLM calls", text)
    text = text.replace("named 'yam_arms'", f"named '{embodiment_name}'")
    if bounds is not None and not (bounds.is_pinned("pitch") and bounds.is_pinned("roll")):
        text += tilt_note(tilt_note_path)
    return text


# ---------------------------------------------------------------------------
# End-effector state
# ---------------------------------------------------------------------------
class ArmPose:
    __slots__ = ("pos", "rot", "gripper", "yaw", "pitch", "roll")

    def __init__(self, pos, rot, gripper, yaw, pitch, roll):
        self.pos, self.rot, self.gripper = pos, rot, gripper
        self.yaw, self.pitch, self.roll = yaw, pitch, roll


def split_arm(q14: np.ndarray, arm: str) -> np.ndarray:
    return np.asarray(q14, dtype=float)[ARM_SLICES[arm]]


def arm_poses(
    q14: np.ndarray,
    kin: ArmKinematics,
    start_rot: Dict[str, np.ndarray],
    yaw_ref: Optional[Dict[str, float]] = None,
) -> Dict[str, ArmPose]:
    """FK of both arms + orientation relative to the trial start. `yaw_ref` keeps yaw unwrapped."""
    q14 = np.asarray(q14, dtype=float)
    if q14.shape[0] != NUM_DOFS:
        raise ValueError(f"expected {NUM_DOFS} joint values, got {q14.shape[0]}")
    poses = {}
    for arm in ARMS:
        pos, rot = kin.fk(q14[ARM_SLICES[arm]])
        yaw, pitch, roll = relative_ypr(rot, start_rot[arm])
        if yaw_ref is not None and arm in yaw_ref:
            yaw = unwrap_angle(yaw, yaw_ref[arm])
        poses[arm] = ArmPose(pos, rot, float(q14[ARM_GRIPPER_INDEX[arm]]), float(yaw), float(pitch), float(roll))
    return poses


def eef_state_dict(poses: Dict[str, ArmPose]) -> Dict[str, float]:
    """Ordered {dimension name: value} following DIM_NAMES."""
    out: Dict[str, float] = {}
    for arm in ARMS:
        p = poses[arm]
        out[f"{arm}_x"] = float(p.pos[0])
        out[f"{arm}_y"] = float(p.pos[1])
        out[f"{arm}_z"] = float(p.pos[2])
        out[f"{arm}_yaw"] = p.yaw
        out[f"{arm}_pitch"] = p.pitch
        out[f"{arm}_roll"] = p.roll
        out[f"{arm}_gripper"] = p.gripper
    return out


def format_eef_state(eef: Dict[str, float]) -> str:
    return " ".join(f"{name}={eef[name]:.4f}" for name in DIM_NAMES)


def format_joint_pos(q14: np.ndarray) -> str:
    vals = [round(float(v), 4) for v in np.asarray(q14, dtype=float)]
    return "[" + ", ".join(repr(v) if v != 0 else "0.0" for v in vals) + "]"
