"""Observation freshness and bounded episode memory; no model or robot state oracle."""
from __future__ import annotations

import cv2
import numpy as np

from astra_yam.config import ReactiveConfig


def validate_reactive(cfg: ReactiveConfig) -> None:
    if not np.isfinite(cfg.max_motion_seconds) or not 0 < cfg.max_motion_seconds <= 10:
        raise ValueError("reactive.max_motion_seconds must be in (0, 10]")
    if not np.isfinite(cfg.change_fraction) or not 0 < cfg.change_fraction <= 1:
        raise ValueError("reactive.change_fraction must be in (0, 1]")
    if not np.isfinite(cfg.pixel_difference) or not 0 < cfg.pixel_difference < 255:
        raise ValueError("reactive.pixel_difference must be in (0, 255)")


def changed_fraction(before: dict, after: dict, cfg: ReactiveConfig) -> float:
    """Compare a fixed camera while the robot is stationary during inference.

    Conservative image change heuristic, not an object tracker or collision sensor.
    Missing/invalid camera data fails closed rather than authorizing blind motion.
    """
    def decode(frames):
        data = frames.get(cfg.camera_name)
        if not data:
            raise RuntimeError(f"reactive camera {cfg.camera_name!r} is missing")
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"reactive camera {cfg.camera_name!r} returned an invalid image")
        return cv2.GaussianBlur(cv2.resize(img, (320, 240)), (5, 5), 0).astype(float)
    delta = np.max(np.abs(decode(before) - decode(after)), axis=2)
    return float(np.mean(delta > cfg.pixel_difference))


class EpisodeMemory:
    """Tentative model lessons, reset on every run; never imported demonstrations."""
    def __init__(self):
        self.lessons: list[str] = []

    def add(self, lesson) -> None:
        if isinstance(lesson, str) and lesson.strip():
            lesson = lesson.strip()[:400]
            if lesson not in self.lessons:
                self.lessons.append(lesson)
                self.lessons = self.lessons[-6:]
