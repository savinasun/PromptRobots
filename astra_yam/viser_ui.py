"""3D visualization + operator UI for the Astra <-> YAM pipeline, built on viser (https://viser.studio).

* Kinematics come from the user's bimanual URDF (`skild_yam_v2.urdf`, left arm at the origin, right arm at
  y = -0.61 m). Link geometry comes from the i2rt YAM MJCF (`yam.xml` + `assets/*_visual.stl`) when available,
  otherwise the arms are drawn as skeletons. The grasp point, jaws (opening = gripper * 9.5 cm), the
  per-arm Cartesian bounds, the planned grasp-point path, the table, the sim objects and the three camera
  poses are all shown.
* In simulation the agent's camera images can be rendered by the connected browser from the same three
  camera poses (`ClientHandle.get_render`), so Astra sees the 3D scene instead of the schematic drawings.
* The GUI carries the trial status, Astra's latest note, the last camera frames, a Start/STOP button, a
  feedback box (lines go to Astra as "Operator feedback: ..."), and drag gizmos to arrange the sim objects.
* With the real robot (`--robot zmq --viser`) the same scene acts as a digital twin fed with the commanded
  and measured joint positions.
"""
from __future__ import annotations

import queue
import re
import select
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from astra_yam.config import ARMS, NUM_DOFS, PipelineConfig
from astra_yam.embodiment import ARM_GRIPPER_INDEX, ARM_SLICES
from astra_yam.kinematics import ArmKinematics

LINK_NAMES = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
GRASP_OFFSET_M = 0.1347          # tcp_site -> grasp_site along the tool z axis (yam.xml)
JAW_MAX_OPENING_M = 0.095
CAMERA_NAMES = ("top_cam", "left_cam", "right_cam")
BASE_DISTANCE = 0.61
# Wrist camera rigidly mounted in the grasp-site frame: 7 cm "above" the housing (site -x, which is world-up
# at the home pose) and 6 cm ahead of the flange; looks 10 cm past the grasp point. Sight lines to the grasp
# point and both jaw tips clear the link_6 mesh from here (checked against the STL).
WRIST_CAM_OFFSET_SITE = np.array([-0.07, 0.0, 0.06])
WRIST_CAM_LOOKAHEAD_M = 0.10
WRIST_CAM_FOV = 1.0
TOP_CAM_FOV = 1.05
ARM_COLORS = {"left": (185, 185, 190), "right": (150, 150, 160)}
PLAN_COLORS = {"left": (30, 120, 255), "right": (255, 120, 30)}


# ---------------------------------------------------------------------------
# small math helpers
# ---------------------------------------------------------------------------
def wxyz_from_matrix(rot: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(np.asarray(rot)).as_quat()
    return np.array([w, x, y, z])


def look_at_wxyz(position: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera orientation (viser/OpenCV convention: +Z forward, +X right, +Y down) looking from position at target."""
    position, target, up = (np.asarray(v, dtype=float) for v in (position, target, up))
    forward = target - position
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:                       # looking straight along `up`
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return wxyz_from_matrix(np.column_stack([right, down, forward]))


def arm_offset(arm: str) -> np.ndarray:
    return np.zeros(3) if arm == "left" else np.array([0.0, -BASE_DISTANCE, 0.0])


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# AirPods case meshes (Apple's AR Quick Look model; see scripts/build_airpods_asset.py)
# ---------------------------------------------------------------------------
CASE_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "airpods"
_CASE_MESH_CACHE: Optional[Dict[str, object]] = None
CASE_LID_RGB = (228, 228, 233)


def case_meshes() -> Dict[str, object]:
    """Body and lid of the AirPods case in SimCase's standing frame, or {} if the asset is missing."""
    global _CASE_MESH_CACHE
    if _CASE_MESH_CACHE is None:
        import trimesh

        out = {}
        for part in ("body", "lid"):
            path = CASE_ASSET_DIR / f"case_{part}.ply"
            if path.exists():
                out[part] = trimesh.load(str(path), process=False)
        if len(out) != 2:
            print(f"[viser] no AirPods case meshes in {CASE_ASSET_DIR}; drawing boxes instead")
            out = {}
        _CASE_MESH_CACHE = out
    return _CASE_MESH_CACHE


def _tinted(mesh, rgb: Sequence[int]):
    m = mesh.copy()
    m.visual.face_colors = (*rgb, 255)
    return m


# ---------------------------------------------------------------------------
# link meshes from the i2rt MJCF
# ---------------------------------------------------------------------------
class LinkMeshes:
    """Visual meshes per link: (trimesh, position, wxyz) expressed in the link frame, parsed from yam.xml."""

    def __init__(self, mjcf_path: Path):
        import trimesh

        mjcf_path = Path(mjcf_path)
        root = ET.parse(mjcf_path).getroot()
        compiler = root.find("compiler")
        meshdir = mjcf_path.parent / (compiler.get("meshdir", "assets") if compiler is not None else "assets")
        files = {m.get("name"): m.get("file") for m in root.iter("mesh")}
        self.items: Dict[str, Tuple["trimesh.Trimesh", np.ndarray, np.ndarray]] = {}

        def add(link: str, geom: ET.Element) -> None:
            fname = files.get(geom.get("mesh"))
            if not fname:
                return
            path = meshdir / fname
            if not path.exists():
                path = meshdir / fname.replace("_collision", "_visual")
            if not path.exists():
                return
            mesh = trimesh.load(str(path), force="mesh")
            pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
            quat = np.array([float(v) for v in geom.get("quat", "1 0 0 0").split()])
            quat = quat / np.linalg.norm(quat)
            self.items[link] = (mesh, pos, quat)

        worldbody = root.find("worldbody")
        if worldbody is None:
            return
        for geom in worldbody.findall("geom"):
            if geom.get("type") == "mesh":
                add("base_link", geom)
        for body in worldbody.iter("body"):
            for geom in body.findall("geom"):
                if geom.get("type") == "mesh":
                    add(body.get("name"), geom)

    @staticmethod
    def default_dir(gello_software_path: Optional[str]) -> Optional[Path]:
        if not gello_software_path:
            return None
        p = Path(gello_software_path) / "third_party" / "robot_models" / "yam"
        return p if (p / "yam.xml").exists() else None


# ---------------------------------------------------------------------------
# the visualizer
# ---------------------------------------------------------------------------
class ViserVisualizer:
    def __init__(self, cfg: PipelineConfig, kin: ArmKinematics, world=None, mode: str = "sim"):
        import viser
        import yourdfpy

        self.cfg = cfg
        self.vc = cfg.viz
        self.kin = kin
        self.world = world
        self.mode = mode
        self.server = viser.ViserServer(host=self.vc.host, port=self.vc.port, label="Astra x YAM", verbose=False)
        self.server.scene.set_up_direction("+z")
        self.server.gui.configure_theme(show_share_button=False, brand_color=(30, 110, 190))
        main_panel = getattr(self.server.gui, "main_panel", None)
        if main_panel is not None:
            main_panel.dock_right()

        self.urdf = yourdfpy.URDF.load(self.vc.urdf_path, load_meshes=False, build_scene_graph=True)
        self._cfg_index: List[int] = []
        for name in self.urdf.actuated_joint_names:
            arm = "left" if name.startswith("left") else "right"
            k = int(re.search(r"(\d+)$", name).group(1)) - 1
            self._cfg_index.append(ARM_SLICES[arm].start + k)
        mesh_dir = Path(self.vc.mesh_dir) if self.vc.mesh_dir else LinkMeshes.default_dir(cfg.robot.gello_software_path)
        self.meshes: Optional[LinkMeshes] = None
        if mesh_dir is not None and (Path(mesh_dir) / "yam.xml").exists():
            try:
                self.meshes = LinkMeshes(Path(mesh_dir) / "yam.xml")
            except Exception as e:  # noqa: BLE001
                print(f"[viser] could not load link meshes from {mesh_dir}: {e}; drawing skeletons")

        self._lock = threading.RLock()
        self._op_queue: "queue.Queue[str]" = queue.Queue()
        self._start_event = threading.Event()
        # Set by whoever owns the motion (cmd_run wires TrialRunner.request_estop, the manual bench its
        # gateway's event) so the EMERGENCY STOP button can halt a motion that is already streaming.
        self.on_estop: Optional[Callable[[], None]] = None
        self.on_reobserve: Optional[Callable[[], None]] = None
        self._last_update = 0.0
        self._last_q: Optional[np.ndarray] = None
        self._pending_q: Optional[np.ndarray] = None   # pose dropped by rate limiting, flushed before renders
        self._link_frames: Dict[Tuple[str, str], object] = {}
        self._skeleton: Dict[str, object] = {}
        self._tool_frames: Dict[str, object] = {}
        self._tool_axes: Dict[str, object] = {}
        self._grasp_markers: Dict[str, object] = {}
        self._arm_axes: List[object] = []
        self._label_specs: Dict[str, Tuple[str, Callable[[], np.ndarray]]] = {}   # node -> (text, position getter)
        self._label_handles: Dict[str, object] = {}
        self._jaws: Dict[str, list] = {}
        self._cam_frustums: Dict[str, object] = {}
        self._bounds: Dict[str, object] = {}
        self._object_handles: Dict[str, tuple] = {}
        self._gizmos: Dict[str, object] = {}
        self._plan_handles: list = []
        self._status: Dict[str, object] = {"goal": "", "phase": "idle", "llm_calls": 0, "max_calls": cfg.limits.max_llm_calls,
                                           "waypoints": 0, "remaining": cfg.limits.max_waypoints, "last_result": ""}
        self._build_scene()
        self._build_gui()
        self._apply_display_defaults()

    # ------------------------------------------------------------------ misc
    def url(self) -> str:
        host = lan_ip() if self.vc.host in ("0.0.0.0", "") else self.vc.host
        return f"http://{host}:{self.server.get_port()}"

    def has_client(self) -> bool:
        return len(self.server.get_clients()) > 0

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------- scene
    def _table_z(self) -> float:
        return float(self.world.table_z) if self.world is not None else 0.0

    def _build_scene(self) -> None:
        sc = self.server.scene
        tz = self._table_z()
        sc.add_box("/table", color=(205, 170, 125), dimensions=(1.2, 1.6, 0.02), position=(0.45, -0.305, tz - 0.01),
                   cast_shadow=False)
        sc.add_grid("/table_grid", width=1.2, height=1.6, plane="xy", cell_size=0.1, section_size=0.5,
                    cell_color=(160, 130, 95), section_color=(120, 90, 60), position=(0.45, -0.305, tz + 0.001))
        for arm in ARMS:
            off = arm_offset(arm)
            sc.add_frame(f"/robot/{arm}", show_axes=False, position=off)
            self._arm_axes.append(sc.add_frame(f"/robot/{arm}/base_axes", show_axes=True, axes_length=0.08, axes_radius=0.003))
            self._label_specs[f"/labels/{arm}"] = (f"{arm} arm base", lambda off=off: off + np.array([-0.12, 0.0, 0.03]))
            b = self.cfg.bounds
            (x0, x1), (y0, y1), (z0, z1) = b.for_dim("x"), b.for_dim("y"), b.for_dim("z")
            self._bounds[arm] = sc.add_box(f"/bounds/{arm}", color=(60, 160, 255), wireframe=True, opacity=0.35,
                                           dimensions=(x1 - x0, y1 - y0, z1 - z0),
                                           position=off + np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]),
                                           visible=False)
            for link in LINK_NAMES:
                h = sc.add_frame(f"/robot/{arm}/{link}", show_axes=False)
                self._link_frames[(arm, link)] = h
                if self.meshes is not None and link in self.meshes.items:
                    mesh, pos, wxyz = self.meshes.items[link]
                    mesh = mesh.copy()
                    mesh.visual.face_colors = (*ARM_COLORS[arm], 255)
                    sc.add_mesh_trimesh(f"/robot/{arm}/{link}/mesh", mesh, position=pos, wxyz=wxyz)
            if self.meshes is None or len(self.meshes.items) < 2:
                self._skeleton[arm] = sc.add_line_segments(f"/robot/{arm}/skeleton", points=np.zeros((6, 2, 3)),
                                                            colors=(50, 50, 60), thickness=0.03)
            tool = sc.add_frame(f"/robot/{arm}/tool", show_axes=False)
            self._tool_frames[arm] = tool
            self._tool_axes[arm] = sc.add_frame(f"/robot/{arm}/tool/axes", show_axes=True, axes_length=0.05, axes_radius=0.0025)
            self._jaws[arm] = [
                sc.add_box(f"/robot/{arm}/tool/jaw_{i}", color=(45, 45, 50), dimensions=(0.012, 0.008, 0.06),
                           position=(0.0, sgn * JAW_MAX_OPENING_M / 2, -0.03))
                for i, sgn in enumerate((1.0, -1.0))
            ]
            self._grasp_markers[arm] = sc.add_icosphere(f"/robot/{arm}/tool/grasp_point", radius=0.006, color=(255, 60, 60))
            self._cam_frustums[f"{arm}_cam"] = sc.add_camera_frustum(f"/cameras/{arm}_cam", fov=WRIST_CAM_FOV, aspect=4 / 3,
                                                                     scale=0.06, color=(255, 140, 0))
        pos, wxyz = self._top_camera_pose()
        self._cam_frustums["top_cam"] = sc.add_camera_frustum("/cameras/top_cam", fov=TOP_CAM_FOV, aspect=4 / 3,
                                                              scale=0.12, color=(255, 140, 0), position=pos, wxyz=wxyz)
        self._build_objects()
        self._create_labels()

    # 3D labels are HTML overlays in the viewer: toggling `visible` leaves their background planes in renders and
    # removal is applied asynchronously (renders captured right after `remove()` still showed white streaks).
    # They are therefore off by default (object names live in the GUI) and, when enabled, removed
    # `label_settle_s` before each agent render and recreated afterwards.
    def _create_labels(self) -> None:
        if not self.vc.show_labels:
            return
        for node, (text, get_pos) in self._label_specs.items():
            if node not in self._label_handles:
                self._label_handles[node] = self.server.scene.add_label(node, text, position=np.asarray(get_pos()))

    def _remove_labels(self) -> None:
        for node, h in list(self._label_handles.items()):
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._label_handles = {}

    def _build_objects(self) -> None:
        if self.world is None:
            return
        import trimesh

        from astra_yam.sim import SimBowl, SimCase

        sc = self.server.scene
        self._case_lids: Dict[str, object] = getattr(self, "_case_lids", {})
        for name, obj in self.world.objects.items():
            node = f"/objects/{_slug(name)}"
            if isinstance(obj, SimCase):
                meshes = case_meshes()
                if meshes:
                    body = sc.add_mesh_trimesh(node + "/body", _tinted(meshes["body"], obj.color_rgb),
                                               position=obj.body_center())
                    self._case_lids[name] = sc.add_mesh_trimesh(node + "/lid", _tinted(meshes["lid"], CASE_LID_RGB),
                                                                position=obj.lid_center())
                else:
                    body = sc.add_box(node + "/body", color=obj.color_rgb, dimensions=self._case_body_dims(obj),
                                      position=obj.body_center())
                    self._case_lids[name] = sc.add_box(node + "/lid", color=CASE_LID_RGB,
                                                       dimensions=obj.lid_dims(), position=obj.lid_center())
                self._label_specs[node + "_label"] = (obj.name, lambda o=obj: o.pos + np.array([0, 0, o.height / 2 + 0.03]))
                self._object_handles[name] = (body, node + "_label")
                continue
            if obj.shape == "box":
                h = sc.add_box(node, color=obj.color_rgb, dimensions=obj.size, position=obj.pos)
            else:
                if isinstance(obj, SimBowl):
                    ring = trimesh.creation.annulus(r_min=obj.inner_radius, r_max=obj.size[0], height=obj.height, sections=48)
                    bottom = trimesh.creation.cylinder(radius=obj.inner_radius, height=obj.floor_thickness, sections=48)
                    bottom.apply_translation([0, 0, -obj.height / 2 + obj.floor_thickness / 2])
                    mesh = trimesh.util.concatenate([ring, bottom])
                else:
                    mesh = trimesh.creation.cylinder(radius=obj.size[0], height=obj.size[1], sections=48)
                mesh.visual.face_colors = (*obj.color_rgb, 255)
                h = sc.add_mesh_trimesh(node, mesh, position=obj.pos)
            self._label_specs[node + "_label"] = (obj.name, lambda o=obj: o.pos + np.array([0, 0, o.height / 2 + 0.03]))
            self._object_handles[name] = (h, node + "_label")

    def _set_edit_objects(self, enabled: bool) -> None:
        if self.world is None:
            return
        with self._lock:
            if not enabled:
                for g in self._gizmos.values():
                    g.remove()
                self._gizmos = {}
                return
            for name, obj in self.world.objects.items():
                if name in self._gizmos:
                    continue
                g = self.server.scene.add_transform_controls(f"/gizmos/{_slug(name)}", scale=0.12,
                                                             disable_rotations=True, position=obj.pos)

                def _moved(event, _name=name):
                    self.world.set_object_position(_name, np.asarray(event.target.position))
                    self.update_objects()
                    if self.cfg.reactive.enabled:
                        self._on_reobserve()

                g.on_update(_moved)
                self._gizmos[name] = g

    # ---------------------------------------------------------------- cameras
    def _top_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array([-0.25, -0.305, 0.95])
        return pos, look_at_wxyz(pos, (0.35, -0.305, self._table_z()))

    def _wrist_camera_pose(self, arm: str, q14: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Camera fixed to the gripper (grasp-site frame offsets), image 'up' = the top of the gripper housing."""
        grasp, rot = self.kin.fk(q14[ARM_SLICES[arm]])
        tcp = grasp - GRASP_OFFSET_M * rot[:, 2]
        pos = tcp + rot @ WRIST_CAM_OFFSET_SITE + arm_offset(arm)
        target = grasp + WRIST_CAM_LOOKAHEAD_M * rot[:, 2] + arm_offset(arm)
        return pos, look_at_wxyz(pos, target, up=-rot[:, 0])

    def camera_pose(self, name: str, q14: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        if name == "top_cam":
            pos, wxyz = self._top_camera_pose()
            return pos, wxyz, TOP_CAM_FOV
        arm = "left" if name.startswith("left") else "right"
        q = q14 if q14 is not None else self._last_q
        if q is None:
            raise RuntimeError("robot pose unknown; call update_robot first")
        pos, wxyz = self._wrist_camera_pose(arm, q)
        return pos, wxyz, WRIST_CAM_FOV

    def _overlay_handles(self) -> list:
        """Helper geometry that exists only in the viewer, hidden while rendering the agent's images."""
        handles = list(self._cam_frustums.values()) + list(self._bounds.values()) + list(self._tool_axes.values())
        handles += list(self._grasp_markers.values()) + list(self._arm_axes) + list(self._plan_handles)
        handles += list(self._gizmos.values())
        return [h for h in handles if h is not None]

    def render_cameras(self, names: Sequence[str] = CAMERA_NAMES) -> Optional[Dict[str, bytes]]:
        """JPEG renders from the connected browser, or None when unavailable (caller falls back)."""
        if not self._render_cb.value:
            return None
        clients = self.server.get_clients()
        if not clients:
            return None
        client = clients[min(clients)]
        out: Dict[str, bytes] = {}
        self.flush()                       # the render must show the pose the observation reports
        with self._lock:
            hidden = []
            with self.server.atomic():
                for h in self._overlay_handles():
                    try:
                        if getattr(h, "visible", True):
                            h.visible = False
                            hidden.append(h)
                    except Exception:  # noqa: BLE001 - e.g. a handle removed meanwhile
                        pass
                had_labels = bool(self._label_handles)
                self._remove_labels()
            if had_labels:
                time.sleep(self.vc.label_settle_s)
            try:
                for name in names:
                    pos, wxyz, fov = self.camera_pose(name)
                    img = client.get_render(self.vc.render_height, self.vc.render_width, wxyz=wxyz, position=pos, fov=fov,
                                            transport_format="jpeg", timeout=self.vc.render_timeout_s)
                    bgr = cv2.cvtColor(np.asarray(img)[..., :3], cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.cameras.jpeg_quality)])
                    if not ok:
                        return None
                    out[name] = buf.tobytes()
            except Exception as e:  # noqa: BLE001 - timeouts, disconnects
                print(f"[viser] render failed ({type(e).__name__}: {e}); using fallback cameras")
                return None
            finally:
                with self.server.atomic():
                    for h in hidden:
                        try:
                            h.visible = True
                        except Exception:  # noqa: BLE001
                            pass
                    self._create_labels()
        return out

    # ------------------------------------------------------------ robot pose
    def update_robot(self, q14: np.ndarray, force: bool = False) -> None:
        q14 = np.asarray(q14, dtype=float)
        if q14.shape[0] != NUM_DOFS:
            return
        now = time.perf_counter()
        if not force and now - self._last_update < 1.0 / max(self.vc.update_hz, 1.0):
            self._pending_q = q14.copy()
            return
        with self._lock:
            self._pending_q = None
            self._last_update = now
            self._last_q = q14.copy()
            cfg = np.array([q14[i] for i in self._cfg_index])
            self.urdf.update_cfg(cfg)
            with self.server.atomic():
                for arm in ARMS:
                    origins = []
                    for link in LINK_NAMES:
                        T = self.urdf.get_transform(f"{arm}_{link}", f"{arm}_base_link")
                        h = self._link_frames[(arm, link)]
                        h.position = T[:3, 3]
                        h.wxyz = wxyz_from_matrix(T[:3, :3])
                        origins.append(T[:3, 3])
                    if arm in self._skeleton:
                        pts = np.stack([np.stack([origins[i], origins[i + 1]]) for i in range(len(origins) - 1)])
                        pts = np.concatenate([pts, self._tool_segment(q14, arm)[None]], axis=0)
                        self._skeleton[arm].points = pts
                    grasp, rot = self.kin.fk(q14[ARM_SLICES[arm]])
                    tool = self._tool_frames[arm]
                    tool.position = grasp
                    tool.wxyz = wxyz_from_matrix(rot)
                    half = JAW_MAX_OPENING_M * float(np.clip(q14[ARM_GRIPPER_INDEX[arm]], 0.0, 1.0)) / 2
                    self._jaws[arm][0].position = (0.0, half, -0.03)
                    self._jaws[arm][1].position = (0.0, -half, -0.03)
                    pos, wxyz = self._wrist_camera_pose(arm, q14)
                    fr = self._cam_frustums[f"{arm}_cam"]
                    fr.position, fr.wxyz = pos, wxyz
                if self.world is not None:
                    self._update_objects_unlocked()

    def flush(self) -> None:
        """Push a pose that was dropped by rate limiting (e.g. the last waypoint of a fast motion)."""
        pending = self._pending_q
        if pending is not None:
            self.update_robot(pending, force=True)

    def _tool_segment(self, q14: np.ndarray, arm: str) -> np.ndarray:
        grasp, rot = self.kin.fk(q14[ARM_SLICES[arm]])
        return np.stack([grasp - GRASP_OFFSET_M * rot[:, 2], grasp])

    def update_objects(self) -> None:
        if self.world is None:
            return
        with self._lock, self.server.atomic():
            self._update_objects_unlocked()

    def _objects_markdown(self) -> str:
        rows = ["| object | shape | size (m) | position x, y, z (left-arm frame) | state |", "|---|---|---|---|---|"]
        for name, o in self.world.objects.items():
            size = f"r={o.size[0]:.3f} h={o.size[1]:.3f}" if o.shape == "cylinder" else "x".join(f"{v:.3f}" for v in o.size)
            state = f"held by {o.held_by}" if o.held_by else "-"
            extra = getattr(o, "lid_angle", None)
            if extra is not None:
                state += f", lid {np.degrees(extra):.0f} deg{' (open)' if o.is_open else ''}{', hanging' if o.hanging else ''}"
            if getattr(o, "poured_in_bowl", None) is not None:
                state += f", tilt {np.degrees(o.tilt_rad):.0f} deg, poured {o.poured_in_bowl:.0%}, spilled {o.spilled:.0%}"
            rows.append(f"| {name} | {o.shape} | {size} | {o.pos[0]:.3f}, {o.pos[1]:.3f}, {o.pos[2]:.3f} | {state} |")
        return "\n".join(rows)

    def _update_powder(self, name: str, can, poured: float) -> None:
        """A growing dark-red disc inside the target bowl shows how much has been poured."""
        bowl = self.world.objects.get(getattr(can, "target", ""))
        if bowl is None or poured <= 0:
            return
        import trimesh

        node = f"/objects/{_slug(name)}_powder"
        h = max(0.003, 0.03 * poured)
        pos = bowl.pos + np.array([0, 0, -bowl.height / 2 + h / 2 + 0.002])
        handle = getattr(self, "_powder_handles", {}).get(name)
        if not hasattr(self, "_powder_handles"):
            self._powder_handles = {}
        if handle is None or abs(getattr(handle, "_powder_h", -1) - h) > 0.002:
            if handle is not None:
                handle.remove()
            mesh = trimesh.creation.cylinder(radius=bowl.footprint_radius * 0.8, height=h, sections=32)
            mesh.visual.face_colors = (150, 30, 20, 255)
            handle = self.server.scene.add_mesh_trimesh(node, mesh, position=pos)
            handle._powder_h = h
            self._powder_handles[name] = handle

    @staticmethod
    def _case_body_dims(case) -> Tuple[float, float, float]:
        return (case.H, case.W, case.D - case.LID) if case.hanging else (case.D - case.LID, case.W, case.H)

    def _update_case(self, name: str, case, body_handle) -> None:
        # the meshes are modelled standing; laying the case down is +90 deg about y
        lay = np.eye(3) if case.hanging else Rotation.from_euler("y", np.pi / 2).as_matrix()
        meshes = bool(case_meshes())
        body_handle.position = case.body_center()
        if meshes:
            body_handle.wxyz = wxyz_from_matrix(lay)
        lid = self._case_lids.get(name)
        if not meshes:
            try:
                body_handle.dimensions = self._case_body_dims(case)
                if lid is not None:
                    lid.dimensions = case.lid_dims()
            except Exception:  # noqa: BLE001 - older viser without settable dimensions
                pass
        if lid is None:
            return
        hinge = case.hinge()
        sign = -1.0 if case.hanging else 1.0
        rot = Rotation.from_euler("y", sign * case.lid_angle).as_matrix()
        lid.position = hinge + rot @ (case.lid_center() - hinge)
        lid.wxyz = wxyz_from_matrix(rot @ lay if meshes else rot)

    def _update_objects_unlocked(self) -> None:
        from astra_yam.sim import SimCase

        if self._objects_md is not None and time.perf_counter() - self._objects_md_t > 0.5:
            self._objects_md_t = time.perf_counter()
            try:
                self._objects_md.content = self._objects_markdown()
            except Exception:  # noqa: BLE001
                pass
        for name, obj in self.world.objects.items():
            if name not in self._object_handles:
                continue
            h, label_node = self._object_handles[name]
            if isinstance(obj, SimCase):
                self._update_case(name, obj, h)
            else:
                h.position = obj.pos
                try:
                    h.wxyz = wxyz_from_matrix(obj.rot)
                except Exception:  # noqa: BLE001
                    pass
                poured = getattr(obj, "poured_in_bowl", None)
                if poured is not None:
                    self._update_powder(name, obj, poured)
            label = self._label_handles.get(label_node)
            if label is not None:
                label.position = obj.pos + np.array([0, 0, obj.height / 2 + 0.03])
            g = self._gizmos.get(name)
            if g is not None and np.linalg.norm(np.asarray(g.position) - obj.pos) > 1e-4 and obj.held_by is not None:
                g.position = obj.pos

    # ------------------------------------------------------------------ plan
    def show_plan(self, plan) -> None:
        self.clear_plan()
        sc = self.server.scene
        stride = max(1, len(plan.q_path) // 150)
        with self._lock:
            for arm in ARMS:
                pts = [self.kin.fk(plan.start_q[ARM_SLICES[arm]])[0] + arm_offset(arm)]
                pts += [self.kin.fk(q[ARM_SLICES[arm]])[0] + arm_offset(arm) for q in plan.q_path[::stride]]
                pts.append(self.kin.fk(plan.q_path[-1][ARM_SLICES[arm]])[0] + arm_offset(arm))
                pts = np.asarray(pts)
                if np.linalg.norm(pts[-1] - pts[0]) < 1e-4:
                    continue
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                self._plan_handles.append(sc.add_line_segments(f"/plan/{arm}_path", points=segs, colors=PLAN_COLORS[arm],
                                                               thickness=0.004))
                self._plan_handles.append(sc.add_icosphere(f"/plan/{arm}_goal", radius=0.01, color=PLAN_COLORS[arm],
                                                           position=pts[-1], opacity=0.8))
            for h in self._plan_handles:
                h.visible = self._show_plan_cb.value

    def _apply_display_defaults(self) -> None:
        self._set_visible(self._bounds.values(), self._show_bounds_cb.value)

    def clear_plan(self) -> None:
        with self._lock:
            for h in self._plan_handles:
                try:
                    h.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._plan_handles = []

    # ------------------------------------------------------------------- gui
    def _build_gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Trial"):
            self._status_md = gui.add_markdown(self._status_markdown())
            self._progress = gui.add_progress_bar(0.0, color="blue")
            self._note_md = gui.add_markdown("_Astra's notes appear here._")
        self._objects_md = None
        if self.world is not None:
            with gui.add_folder("Scene objects", expand_by_default=False):
                self._objects_md = gui.add_markdown(self._objects_markdown())
                self._objects_md_t = 0.0
        with gui.add_folder("Operator"):
            self._start_btn = gui.add_button("Start trial", color="green", hint="Begin (or continue to) the next goal")
            self._estop_btn = gui.add_button("EMERGENCY STOP", color="red",
                                             hint="Halts the motion in flight (within one control tick), holds "
                                                  "position, and ends the session")
            self._stop_btn = gui.add_button("Stop after this motion", color="orange",
                                            hint="Lets the current motion finish, then ends the session; the arms hold")
            self._reobserve_btn = gui.add_button("Scene moved — reobserve", visible=self.cfg.reactive.enabled,
                                                hint="Pause the current motion, keep the grasp, and obtain a fresh observation")
            self._fb_text = gui.add_text("Feedback to Astra", "", hint="Sent before the next Astra call")
            self._send_btn = gui.add_button("Send feedback")
            self._edit_cb = gui.add_checkbox("Edit objects (drag gizmos)", False, visible=self.world is not None)
            self._reset_btn = gui.add_button("Reset objects", visible=self.world is not None)
        with gui.add_folder("Cameras"):
            self._render_cb = gui.add_checkbox("Agent sees this 3D render", bool(self.vc.render_observations and self.mode == "sim"),
                                               hint="Needs a connected browser; otherwise schematic images are used")
            blank = np.full((120, 160, 3), 40, np.uint8)
            self._cam_images = {name: gui.add_image(blank, label=name) for name in CAMERA_NAMES}
        with gui.add_folder("Display", expand_by_default=False):
            big = any(self.cfg.bounds.for_dim(d)[1] - self.cfg.bounds.for_dim(d)[0] > 1.0 for d in ("x", "y", "z"))
            self._show_bounds_cb = gui.add_checkbox("Workspace bounds", not big,
                                                    hint="off by default when the bounds span the whole reach envelope")
            self._show_plan_cb = gui.add_checkbox("Planned path", True)
            self._show_cams_cb = gui.add_checkbox("Camera frustums", True)
            self._show_tool_cb = gui.add_checkbox("Tool frames", True)

        self._start_btn.on_click(lambda _e: self._on_start())
        self._estop_btn.on_click(lambda _e: self._on_estop())
        self._stop_btn.on_click(lambda _e: self._on_stop())
        self._reobserve_btn.on_click(lambda _e: self._on_reobserve())
        self._send_btn.on_click(lambda _e: self._on_send())
        self._edit_cb.on_update(lambda _e: self._set_edit_objects(self._edit_cb.value))
        self._reset_btn.on_click(lambda _e: self._on_reset_objects())
        self._show_bounds_cb.on_update(lambda _e: self._set_visible(self._bounds.values(), self._show_bounds_cb.value))
        self._show_plan_cb.on_update(lambda _e: self._set_visible(self._plan_handles, self._show_plan_cb.value))
        self._show_cams_cb.on_update(lambda _e: self._set_visible(self._cam_frustums.values(), self._show_cams_cb.value))
        self._show_tool_cb.on_update(lambda _e: self._set_visible(list(self._tool_axes.values()) + list(self._grasp_markers.values()),
                                                                  self._show_tool_cb.value))

    @staticmethod
    def _set_visible(handles, visible: bool) -> None:
        for h in list(handles):
            try:
                h.visible = visible
            except Exception:  # noqa: BLE001
                pass

    # GUI callbacks (also called directly by tests)
    def _on_start(self) -> None:
        self._start_event.set()

    def _on_stop(self) -> None:
        self._op_queue.put("/stop")
        self.set_status(phase="stop requested (after this motion)")

    def _on_reobserve(self) -> None:
        if self.on_reobserve is not None:
            self.on_reobserve()
        else:
            self._op_queue.put("/reobserve")
        self.set_status(phase="scene changed; reobservation requested")

    def _on_estop(self) -> None:
        """Emergency stop: halt the motion now, then end the session.

        `on_estop` is wired to `TrialRunner.request_estop` (or the manual bench's gateway) and stops a motion
        that is already streaming; the queued `/stop` ends the session even when nothing is moving and no
        runner is attached.
        """
        if self.on_estop is not None:
            try:
                self.on_estop()
            except Exception as e:  # noqa: BLE001 - the queued stop below is the fallback
                print(f"[viser] e-stop callback failed: {type(e).__name__}: {e}")
        self._op_queue.put("/stop")
        self.set_note("**EMERGENCY STOP** - motion halted, arms holding position.")
        self.set_status(phase="EMERGENCY STOP")

    def _on_send(self) -> None:
        text = self._fb_text.value.strip()
        if text:
            self._op_queue.put(text)
            self._fb_text.value = ""
            self.set_note(f"**Operator feedback queued:** {text}")

    def _on_reset_objects(self) -> None:
        if self.world is not None:
            self.world.reset_objects()
            self.update_objects()

    def poll_operator(self) -> List[str]:
        lines = []
        while True:
            try:
                lines.append(self._op_queue.get_nowait())
            except queue.Empty:
                return lines

    def wait_for_start(self, timeout: Optional[float] = None, allow_stdin: bool = True) -> bool:
        """Block until the Start button is pressed (or Enter in an interactive terminal). Returns False on timeout.

        A press that happened before this call counts (the event is consumed on return), so clicking Start while
        the robot is still homing is not lost."""
        self.set_status(phase="waiting for Start")
        t0 = time.perf_counter()
        while True:
            if self._start_event.is_set():
                self._start_event.clear()
                return True
            if allow_stdin and sys.stdin is not None and sys.stdin.isatty():
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r:
                    sys.stdin.readline()
                    return True
            if timeout is not None and time.perf_counter() - t0 > timeout:
                return False
            time.sleep(0.1)

    # ------------------------------------------------------------- status
    def _status_markdown(self) -> str:
        s = self._status
        return (f"**Goal:** {s['goal'] or '-'}  \n"
                f"**Phase:** {s['phase']}  \n"
                f"**Astra calls:** {s['llm_calls']} / {s['max_calls']}  \n"
                f"**Waypoints:** {s['waypoints']} executed, {s['remaining']} remaining  \n"
                f"**Last gateway result:** {s['last_result'] or '-'}")

    def set_status(self, **kw) -> None:
        self._status.update({k: v for k, v in kw.items() if v is not None})
        try:
            self._status_md.content = self._status_markdown()
            self._progress.value = 100.0 * float(self._status["llm_calls"]) / max(1, int(self._status["max_calls"]))
        except Exception:  # noqa: BLE001
            pass

    def set_note(self, text: str) -> None:
        try:
            self._note_md.content = text
        except Exception:  # noqa: BLE001
            pass

    def set_frames(self, frames: Dict[str, bytes], preview_width: int = 320) -> None:
        for name, jpeg in frames.items():
            handle = self._cam_images.get(name)
            if handle is None:
                continue
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w > preview_width:
                img = cv2.resize(img, (preview_width, int(h * preview_width / w)))
            handle.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------- trial hooks
    def on_trial_start(self, goal: str, q0: np.ndarray) -> None:
        self.clear_plan()
        self._status.update({"llm_calls": 0, "waypoints": 0, "remaining": self.cfg.limits.max_waypoints, "last_result": ""})
        self.set_status(goal=goal, phase="observing")
        self.set_note("_Astra's notes appear here._")
        self.update_robot(q0, force=True)

    def on_observation(self, step: int, q: np.ndarray, eef: Dict[str, float], frames: Dict[str, bytes], remaining: int) -> None:
        self.update_robot(q, force=True)
        self.set_frames(frames)
        self.set_status(waypoints=step, remaining=remaining)

    def on_astra_call(self, i: int, max_calls: int, n_items: int, n_images: int) -> None:
        self.set_status(phase=f"Astra thinking (call {i}, {n_items} items, {n_images} images)", llm_calls=i, max_calls=max_calls)

    def on_astra_response(self, resp) -> None:
        self.set_status(phase=f"Astra answered in {resp.elapsed_s:.1f}s")

    def on_plan(self, plan) -> None:
        self.show_plan(plan)
        self.set_status(phase=f"executing {plan.steps} waypoints")

    def on_tool_call(self, name: str, args: dict, payload: Optional[dict], plan) -> None:
        args = args or {}
        if name == "move_to":
            note = args.get("note") or ""
            targets = ", ".join(f"{k}={v}" for k, v in (args.get("targets") or {}).items())
            self.set_note(f"**move_to** `{targets}`  \n{note}")
            if payload is not None:
                status = payload.get("status")
                text = f"{status} ({payload.get('steps', 0)} steps)" if payload.get("ok") else f"{status}: {payload.get('reason', '')}"
                self.set_status(last_result=text, phase="observing")
        else:
            self.set_note(f"**{name}**: {args.get('summary') or args.get('reason') or ''}  \n_hindsight:_ {args.get('hindsight', '')}")

    def on_end(self, outcome) -> None:
        self.set_status(phase=f"finished: {outcome.status}", llm_calls=outcome.llm_calls, waypoints=outcome.waypoints)


# ---------------------------------------------------------------------------
# adapters used by the CLI
# ---------------------------------------------------------------------------
class ViserCameraSource:
    """Camera source that renders the agent's images in the browser, falling back to another source."""

    def __init__(self, viz: ViserVisualizer, fallback, names: Sequence[str] = CAMERA_NAMES):
        self.viz = viz
        self.fallback = fallback
        self.names = list(names)
        self.last_source = "none"

    def read_jpeg_frames(self) -> Dict[str, bytes]:
        frames = self.viz.render_cameras(self.names)
        if frames is not None and len(frames) == len(self.names):
            self.last_source = "viser"
            return frames
        self.last_source = "fallback"
        return self.fallback.read_jpeg_frames()

    def close(self) -> None:
        self.fallback.close()


class MirroredRobot:
    """Robot backend proxy that mirrors every command/measurement into the visualizer (digital twin)."""

    def __init__(self, inner, viz: ViserVisualizer):
        self.inner = inner
        self.viz = viz

    def num_dofs(self) -> int:
        return self.inner.num_dofs()

    def get_joint_positions(self) -> np.ndarray:
        q = self.inner.get_joint_positions()
        self.viz.update_robot(q)
        return q

    def command_joint_positions(self, q14: np.ndarray) -> None:
        self.inner.command_joint_positions(q14)
        self.viz.update_robot(np.asarray(q14, dtype=float))

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, item):
        return getattr(self.inner, item)


# ---------------------------------------------------------------------------
# manual test bench: drag a target gizmo per arm and run it through the gateway
# ---------------------------------------------------------------------------
def run_manual_bench(cfg: PipelineConfig, kin: ArmKinematics, world, robot, viz: ViserVisualizer) -> None:
    """Interactive IK/gateway test without Astra: gizmo targets + move buttons. Blocks until Ctrl-C."""
    from astra_yam.gateway import GatewayRejection, SafetyGateway

    q0 = robot.get_joint_positions()
    start_rot = {arm: kin.fk(q0[ARM_SLICES[arm]])[1] for arm in ARMS}
    gateway = SafetyGateway(cfg, kin, robot, start_rot, realtime=False)
    gateway.on_plan = viz.show_plan
    gateway.estop = threading.Event()
    viz.on_estop = gateway.estop.set
    viz.update_robot(q0, force=True)
    gui = viz.server.gui
    sc = viz.server.scene
    gizmos, yaw_sliders, grip_sliders, move_btns = {}, {}, {}, {}
    with gui.add_folder("Manual bench (no Astra)"):
        gui.add_markdown("Drag a target gizmo, then press *Move*. Targets go through the same gateway as Astra's calls.")
        for arm in ARMS:
            grasp, _ = kin.fk(q0[ARM_SLICES[arm]])
            gizmos[arm] = sc.add_transform_controls(f"/bench/{arm}_target", scale=0.15, disable_rotations=True,
                                                    position=grasp + arm_offset(arm))
            yaw_sliders[arm] = gui.add_slider(f"{arm} yaw (rad)", -3.14, 3.14, 0.01, 0.0)
            grip_sliders[arm] = gui.add_slider(f"{arm} gripper", 0.0, 1.0, 0.01, float(q0[ARM_GRIPPER_INDEX[arm]]))
            move_btns[arm] = gui.add_button(f"Move {arm} arm to target")
        result_md = gui.add_markdown("_result_")

    def make_cb(arm):
        def _cb(_e):
            target = np.asarray(gizmos[arm].position) - arm_offset(arm)
            targets = {f"{arm}_x": float(target[0]), f"{arm}_y": float(target[1]), f"{arm}_z": float(target[2]),
                       f"{arm}_yaw": float(yaw_sliders[arm].value), f"{arm}_gripper": float(grip_sliders[arm].value)}
            payload, plan, res = gateway.move_to(targets, 10 ** 9)
            gateway.estop.clear()          # the bench has no session to end: re-arm for the next move
            viz.update_robot(robot.get_joint_positions(), force=True)
            result_md.content = f"`{targets}`  \n**{payload}**"
            viz.set_status(last_result=str(payload), phase="manual bench")
        return _cb

    for arm in ARMS:
        move_btns[arm].on_click(make_cb(arm))
    print(f"[viser] manual bench ready at {viz.url()} - Ctrl-C to quit")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
