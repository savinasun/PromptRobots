import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from astra_yam.config import REFERENCE_HOME_JOINTS
from astra_yam.kinematics import ArmKinematics, relative_ypr, rotation_from_ypr, unwrap_angle

from conftest import ROOT, reference_transcript
EXAMPLE = reference_transcript("0002_example_input.json")


def _reference_observations():
    """(joint_pos[14], eef dict) pairs from the reference transcript."""
    out = []
    for item in EXAMPLE["input"]:
        if item.get("role") != "user" or not isinstance(item.get("content"), list):
            continue
        text = item["content"][0]["text"]
        jp = json.loads(text.split("state[joint_pos]: ")[1].split("\n")[0])
        eef_line = text.split("state[eef_state]: ")[1].split("\n")[0]
        eef = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in eef_line.split(" ")}
        out.append((np.array(jp), eef))
    return out


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics()


def test_fk_grasp_site_matches_reference_within_1cm(kin):
    obs = _reference_observations()
    assert len(obs) >= 2
    for q14, eef in obs:
        for arm, sl in (("left", slice(0, 6)), ("right", slice(7, 13))):
            pos, _ = kin.fk(q14[sl])
            ref = np.array([eef[f"{arm}_x"], eef[f"{arm}_y"], eef[f"{arm}_z"]])
            err = np.linalg.norm(pos - ref)
            assert err < 0.012, f"{arm}: fk {pos} vs reference {ref} (err {err*1000:.1f} mm)"


def test_fk_frame_is_forward_left_up(kin):
    pos, rot = kin.fk(np.array(REFERENCE_HOME_JOINTS))
    assert pos[0] > 0.25 and abs(pos[1]) < 0.01 and pos[2] > 0.15   # forward, centered, above the base
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)


def test_ik_roundtrip_from_perturbed_seed(kin):
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(40):
        q = rng.uniform(kin.lower + 0.15, kin.upper - 0.15)
        pos, rot = kin.fk(q)
        seed = np.clip(q + rng.normal(0, 0.15, size=6), kin.lower, kin.upper)
        res = kin.ik(pos, rot, seed, max_iters=200, pos_tol=1e-3, ori_tol=5e-3)
        if res.converged:
            n_ok += 1
            p2, r2 = kin.fk(res.q)
            assert np.linalg.norm(p2 - pos) < 1e-3
            assert kin.within_limits(res.q)
    assert n_ok >= 36, f"only {n_ok}/40 IK problems converged"


def test_ik_unreachable_reports_failure(kin):
    _, rot = kin.fk(np.array(REFERENCE_HOME_JOINTS))
    res = kin.ik(np.array([1.2, 0.0, 0.3]), rot, np.array(REFERENCE_HOME_JOINTS), max_iters=100)
    assert not res.converged and res.pos_err_m > 0.1


def test_relative_ypr_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        start = Rotation.random(random_state=rng).as_matrix()
        yaw = rng.uniform(-3.0, 3.0)
        rot = rotation_from_ypr(start, yaw, 0.0, 0.0)
        ypr = relative_ypr(rot, start)
        assert abs(ypr[0] - yaw) < 1e-9 and abs(ypr[1]) < 1e-9 and abs(ypr[2]) < 1e-9
    assert np.allclose(relative_ypr(start, start), 0.0)


def test_positive_yaw_is_counterclockwise_from_above():
    start = np.eye(3)
    rot = rotation_from_ypr(start, 0.5)
    x_axis = rot @ np.array([1.0, 0.0, 0.0])
    assert x_axis[1] > 0  # +x rotated toward +y (left) = counter-clockwise seen from above


def test_unwrap_angle():
    assert abs(unwrap_angle(-3.1, 3.0) - (2 * np.pi - 3.1)) < 1e-12
    assert unwrap_angle(0.2, 0.0) == 0.2
