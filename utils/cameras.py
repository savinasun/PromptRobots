"""Camera sources: the station config's own cameras (via gello's initialize_cameras), RealSense
directly, or none. The simulator lives in sim.py."""
from __future__ import annotations

import json
import time
from typing import Dict, Optional, Protocol

import cv2
import numpy as np

from utils.config import CameraConfig
from utils.robot_interface import ensure_gello_on_path


class CameraSource(Protocol):
    def read_jpeg_frames(self) -> Dict[str, bytes]: ...
    def close(self) -> None: ...


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class NoCameraSource:
    def read_jpeg_frames(self) -> Dict[str, bytes]:
        return {}

    def close(self) -> None:
        pass


def load_station_cameras(station_config_path: str) -> Dict[str, str]:
    """{station camera name: RealSense serial} from metadata/station_config.json."""
    with open(station_config_path, "r") as f:
        cfg = json.load(f)
    out = {}
    for name, entry in cfg.get("camera_ids", {}).items():
        serial = entry.get("device_id") if isinstance(entry, dict) else entry
        out[name] = str(serial)
    return out


class RealSenseSource:
    """Opens the station's RealSense cameras in the fast (background-thread) mode used by run_env.py."""

    def __init__(self, cfg: CameraConfig, gello_software_path: Optional[str] = None):
        ensure_gello_on_path(gello_software_path)
        from gello.cameras.realsense_camera import RealSenseCameraFast, get_device_ids

        self.cfg = cfg
        serials = load_station_cameras(cfg.station_config_path)
        if cfg.reset_on_start:
            available = get_device_ids()  # hardware-resets every device and sleeps 5 s (gello convention)
        else:
            import pyrealsense2 as rs

            available = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
        self._cams = {}
        for station_name, model_name in cfg.names.items():
            if station_name not in serials:
                raise RuntimeError(f"camera '{station_name}' not in station config {cfg.station_config_path}")
            serial = serials[station_name]
            if serial not in available:
                raise RuntimeError(f"camera '{station_name}' (serial {serial}) not detected; found {available}")
            self._cams[model_name] = RealSenseCameraFast(device_id=serial, depth=False, hz=cfg.hz)
        time.sleep(cfg.warmup_seconds)

    def read_jpeg_frames(self) -> Dict[str, bytes]:
        frames = {}
        for name, cam in self._cams.items():
            if hasattr(cam, "has_error") and cam.has_error():
                raise RuntimeError(f"camera '{name}' reported an error: {cam.get_error()}")
            bgr, _ = cam.read()
            frames[name] = encode_jpeg(bgr, self.cfg.jpeg_quality)
        return frames

    def close(self) -> None:
        for cam in self._cams.values():
            try:
                cam.stop()
            except Exception:  # noqa: BLE001
                pass
        self._cams = {}


class StationSource:
    """Opens exactly the cameras the station config describes, through gello's own
    ``camera_utils.initialize_cameras`` - the same construction path as demo collection
    (experiments/run_env.py) and policy eval (eval/eval_bimanual.py).

    Prefer this over RealSenseSource: it dispatches per camera on the station config's
    ``cam_type``, so a station may mix RealSense with ZED / ZED X One. ZED entries that carry
    an ``ip`` are network-streamed from the station's Orin (the ZED SDK's
    ``set_from_stream(ip, port)``), which means nothing is enumerable on the local USB bus and
    the Orin's streaming container must be running before this source is built.
    """

    # eval_bimanual.warmup_and_verify_cameras waits 8 s: a ZED stream takes several seconds after
    # set_from_stream() to deliver a first frame, far longer than a local RealSense. Treat
    # cfg.warmup_seconds (default 1.5, tuned for RealSense) as a floor rather than the deadline.
    MIN_WARMUP_SECONDS = 8.0

    def __init__(self, cfg: CameraConfig, gello_software_path: Optional[str] = None):
        ensure_gello_on_path(gello_software_path)
        from gello.cameras.camera_utils import initialize_cameras

        self.cfg = cfg
        with open(cfg.station_config_path, "r") as f:
            station_cfg = json.load(f)

        declared = station_cfg.get("camera_ids", {})
        missing = [name for name in cfg.names if name not in declared]
        if missing:
            raise RuntimeError(
                f"camera(s) {missing} not in station config {cfg.station_config_path}; "
                f"it declares {sorted(declared)}"
            )

        # initialize_cameras opens every camera in the file, not just the ones we map. Keep the
        # extras so close() shuts their capture threads down too, but never read them.
        self._opened = initialize_cameras(station_cfg)
        unopened = [name for name in cfg.names if name not in self._opened]
        if unopened:
            self.close()
            raise RuntimeError(
                f"gello did not open camera(s) {unopened}; check their 'cam_type' in "
                f"{cfg.station_config_path} (initialize_cameras skips types it does not know)"
            )
        self._cams = {model_name: self._opened[station_name] for station_name, model_name in cfg.names.items()}
        self._warmup()

    def _read_bgr(self, name: str, cam) -> Optional[np.ndarray]:
        """Newest BGR frame, or None when the reader has nothing buffered yet."""
        if hasattr(cam, "has_error") and cam.has_error():
            raise RuntimeError(f"camera '{name}' reported an error: {cam.get_error()}")
        frame = cam.read()
        # RealSenseCameraFast and the ZED adapter both return (image, timestamp); the ZED adapter
        # returns (None, None) until its first grab lands.
        bgr = frame[0] if isinstance(frame, tuple) else frame
        return bgr

    def _warmup(self) -> None:
        """Block until every mapped camera yields a non-black frame.

        A failed capture thread leaves read() returning zeros instead of raising, and Astra cannot
        tell a black frame from a dark scene - it would just plan badly. Same guard as
        eval_bimanual.warmup_and_verify_cameras.
        """
        deadline = time.time() + max(self.cfg.warmup_seconds, self.MIN_WARMUP_SECONDS)
        pending = set(self._cams)
        while pending and time.time() < deadline:
            for name in sorted(pending):
                bgr = self._read_bgr(name, self._cams[name])
                if bgr is not None and int(bgr.max()) > 0:
                    pending.discard(name)
            if pending:
                time.sleep(0.1)
        if pending:
            self.close()
            raise RuntimeError(
                f"camera(s) {sorted(pending)} produced no non-black frame within "
                f"{max(self.cfg.warmup_seconds, self.MIN_WARMUP_SECONDS):.1f} s. A capture thread "
                f"likely failed; for ZED-over-Orin cameras check that the Orin's streaming "
                f"container is up and reachable."
            )

    def read_jpeg_frames(self) -> Dict[str, bytes]:
        frames = {}
        for name, cam in self._cams.items():
            bgr = self._read_bgr(name, cam)
            if bgr is None:
                raise RuntimeError(f"camera '{name}' returned no frame; its capture thread may have stopped")
            frames[name] = encode_jpeg(bgr, self.cfg.jpeg_quality)
        return frames

    def close(self) -> None:
        for cam in getattr(self, "_opened", {}).values():
            for method in ("close", "stop"):   # ZED adapter has both; RealSenseCameraFast only stop()
                fn = getattr(cam, method, None)
                if fn is None:
                    continue
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
                break
        self._opened = {}
        self._cams = {}


def make_camera_source(cfg: CameraConfig, gello_software_path: Optional[str] = None, sim_world=None) -> CameraSource:
    if cfg.backend == "station":
        return StationSource(cfg, gello_software_path)
    if cfg.backend == "realsense":
        return RealSenseSource(cfg, gello_software_path)
    if cfg.backend == "sim":
        from utils.sim import SimCameraSource

        if sim_world is None:
            raise ValueError("camera backend 'sim' needs a SimWorld (use robot backend 'sim' too)")
        return SimCameraSource(sim_world, list(cfg.names.values()), cfg.jpeg_quality)
    if cfg.backend == "none":
        return NoCameraSource()
    raise ValueError(f"unknown camera backend '{cfg.backend}'")
