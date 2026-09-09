"""Camera sources: RealSense (via gello's RealSenseCameraFast) or none. The simulator lives in sim.py."""
from __future__ import annotations

import json
import time
from typing import Dict, Optional, Protocol

import cv2
import numpy as np

from astra_yam.config import CameraConfig
from astra_yam.robot_interface import ensure_gello_on_path


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


def make_camera_source(cfg: CameraConfig, gello_software_path: Optional[str] = None, sim_world=None) -> CameraSource:
    if cfg.backend == "realsense":
        return RealSenseSource(cfg, gello_software_path)
    if cfg.backend == "sim":
        from astra_yam.sim import SimCameraSource

        if sim_world is None:
            raise ValueError("camera backend 'sim' needs a SimWorld (use robot backend 'sim' too)")
        return SimCameraSource(sim_world, list(cfg.names.values()), cfg.jpeg_quality)
    if cfg.backend == "none":
        return NoCameraSource()
    raise ValueError(f"unknown camera backend '{cfg.backend}'")
