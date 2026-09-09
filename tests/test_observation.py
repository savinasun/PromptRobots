import json
from pathlib import Path

import numpy as np

from astra_yam.config import Bounds
from astra_yam.embodiment import build_system_prompt, build_tools, format_eef_state, format_joint_pos
from astra_yam.observation import (
    IMAGE_OMITTED_TEXT,
    build_observation_item,
    count_images,
    observation_text,
    prune_image_history,
    redact_images,
)

from conftest import ROOT, reference_transcript
EXAMPLE = reference_transcript("0000_example_input.json")
REF_OBS_TEXT = EXAMPLE["input"][2]["content"][0]["text"]


def _parse_reference():
    jp = json.loads(REF_OBS_TEXT.split("state[joint_pos]: ")[1].split("\n")[0])
    eef_line = REF_OBS_TEXT.split("state[eef_state]: ")[1].split("\n")[0]
    eef = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in eef_line.split(" ")}
    return np.array(jp), eef


def test_tools_match_reference_verbatim():
    assert build_tools(Bounds()) == EXAMPLE["tools"]


def test_system_prompt_matches_reference_verbatim():
    # The reference transcript predates the camera-use section, which is appended at the end of
    # SYSTEM_PROMPT.md; everything the reference trials saw must still be there byte-for-byte.
    prompt = build_system_prompt(str(ROOT / "configs" / "SYSTEM_PROMPT.md"), 100)
    assert prompt.startswith(EXAMPLE["input"][0]["content"])
    assert "budget of 25 LLM calls" in build_system_prompt(str(ROOT / "configs" / "SYSTEM_PROMPT.md"), 25)


def test_system_prompt_describes_camera_use():
    prompt = build_system_prompt(str(ROOT / "configs" / "SYSTEM_PROMPT.md"), 100)
    assert "Cameras:" in prompt
    assert "wrist camera per arm" in prompt and "steerable sensors" in prompt


def test_observation_text_matches_reference_format():
    q14, eef = _parse_reference()
    text = observation_text("pick up blue and place ontop of green block", q14, eef, 3000)
    assert text == REF_OBS_TEXT


def test_formatting_helpers():
    assert format_joint_pos(np.array([-0.02, 0.0, 1.0, 0.12345])) == "[-0.02, 0.0, 1.0, 0.1235]"
    assert format_eef_state({k: 0.0 for k in ("left_x",)} | {k: 0.0 for k in
                            ["left_y", "left_z", "left_yaw", "left_pitch", "left_roll", "left_gripper", "right_x",
                             "right_y", "right_z", "right_yaw", "right_pitch", "right_roll", "right_gripper"]}).startswith(
        "left_x=0.0000 left_y=0.0000")


def test_observation_item_layout_and_redaction():
    q14, eef = _parse_reference()
    frames = {"top_cam": b"\xff\xd8fake-top", "left_cam": b"\xff\xd8fake-left", "right_cam": b"\xff\xd8fake-right"}
    item = build_observation_item("goal", q14, eef, 2948, 52, frames)
    assert item["role"] == "user" and len(item["content"]) == 7
    assert item["content"][1] == {"type": "input_text", "text": "camera 'top_cam' (step 52):"}
    assert item["content"][2]["type"] == "input_image" and item["content"][2]["detail"] == "high"
    assert item["content"][2]["image_url"].startswith("data:image/jpeg;base64,")
    red = redact_images(item)
    assert red["content"][2]["image_url"].startswith("data:image/jpeg;base64,$blob:")
    assert "base64," + "" not in json.dumps(red).replace("base64,$blob", "")
    assert item["content"][2]["image_url"] != red["content"][2]["image_url"]  # original untouched


def test_prune_image_history_keeps_last_n():
    q14, eef = _parse_reference()
    frames = {"top_cam": b"\xff\xd8x"}
    items = [{"role": "system", "content": "s"}]
    for step in range(4):
        items.append(build_observation_item("g", q14, eef, 3000 - step, step, frames))
        items.append({"type": "function_call", "call_id": f"c{step}", "name": "move_to", "arguments": "{}"})
    assert count_images(items) == 4
    pruned = prune_image_history(items, 2)
    assert count_images(pruned) == 2 and count_images(items) == 4  # original untouched
    assert pruned[1]["content"][2] == {"type": "input_text", "text": IMAGE_OMITTED_TEXT}
    assert prune_image_history(items, 0) is items


def test_tilt_note_only_when_released():
    from astra_yam.embodiment import tilt_note
    path = str(ROOT / "configs" / "SYSTEM_PROMPT.md")
    assert build_system_prompt(path, 100, bounds=Bounds()).startswith(EXAMPLE["input"][0]["content"])
    released = build_system_prompt(path, 100, bounds=Bounds(pitch=(-0.7, 0.7), roll=(-0.7, 0.7)))
    assert released.endswith(tilt_note()) and "pitch" in released
    tools = build_tools(Bounds(pitch=(-0.7, 0.7), roll=(-0.7, 0.7)))
    assert "left_pitch: [-0.7, 0.7]" in tools[0]["description"]
