"""The model-facing text lives in configs/, not in the code."""
import pytest
import yaml
from conftest import ROOT

from astra_yam.config import Bounds, PipelineConfig
from astra_yam.embodiment import build_tools
from astra_yam.observation import build_observation_item, observation_text
from astra_yam.prompts import load_prompts, prompt

PROMPTS = str(ROOT / "configs" / "PROMPTS.yaml")


def test_default_paths_point_into_configs():
    cfg = PipelineConfig()
    for path in (cfg.prompts_path, cfg.system_prompt_path, cfg.tilt_note_path):
        assert path.startswith(str(ROOT / "configs")) and ROOT.joinpath(path).exists()


def test_dotted_lookup_and_missing_keys():
    assert prompt("tools.give_up.description").startswith("Stop trying")
    with pytest.raises(KeyError):
        prompt("tools.nope")
    with pytest.raises(TypeError):
        prompt("tools")                      # a subtree, not text
    with pytest.raises(KeyError):
        prompt("tools.move_to.description")  # {bounds} not supplied


def test_tool_descriptions_are_single_line():
    """`>-` folds the wrapped YAML back into one line; a stray blank line would leak a newline into the API."""
    tools = build_tools(Bounds())
    texts = [t["description"] for t in tools]
    texts += [p["description"] for t in tools for p in t["parameters"]["properties"].values()
              if "description" in p]
    assert texts and all("\n" not in t and "  " not in t for t in texts)


def test_every_prompt_string_is_used_somewhere():
    """A key nobody reads is dead copy; a key read with the wrong name fails the tests above."""
    used = {
        "tools.move_to.description", "tools.move_to.targets", "tools.move_to.note",
        "tools.done.description", "tools.give_up.description", "tools.hindsight",
        "observation.text", "observation.camera_label", "observation.image_omitted",
        "session.goal", "session.operator_feedback", "session.tool_call_reminder",
    }

    def keys(node, prefix=""):
        for k, v in node.items():
            path = f"{prefix}{k}"
            yield from keys(v, f"{path}.") if isinstance(v, dict) else iter([path])

    assert set(keys(load_prompts(PROMPTS))) == used


def test_text_comes_from_the_file_not_the_code(tmp_path):
    """Editing the YAML changes the request; nothing is hard-coded as a fallback."""
    data = yaml.safe_load(open(PROMPTS))
    data["tools"]["done"]["description"] = "STOP HERE"
    data["observation"]["text"] = "OBS {goal} {joint_pos} {eef_state} {waypoints_remaining}"
    data["observation"]["camera_label"] = "IMG {camera} {step}"
    edited = tmp_path / "PROMPTS.yaml"
    edited.write_text(yaml.safe_dump(data))

    import numpy as np

    eef = {f"{a}_{d}": 0.0 for a in ("left", "right")
           for d in ("x", "y", "z", "yaw", "pitch", "roll", "gripper")}
    q14 = np.zeros(14)
    tools = build_tools(Bounds(), str(edited))
    assert [t for t in tools if t["name"] == "done"][0]["description"] == "STOP HERE"
    assert observation_text("g", q14, eef, 7, str(edited)).startswith("OBS g [0.0")
    item = build_observation_item("g", q14, eef, 7, 3, {"top_cam": b"\xff\xd8x"}, "high", str(edited))
    assert item["content"][1] == {"type": "input_text", "text": "IMG top_cam 3"}
    assert build_tools(Bounds())[1]["description"] != "STOP HERE"      # the real file is untouched
