"""Observation messages for Astra (text + camera images), matching the reference transcript format."""
from __future__ import annotations

import base64
import copy
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from astra_yam.embodiment import format_eef_state, format_joint_pos
from astra_yam.prompts import prompt

IMAGE_OMITTED_TEXT = prompt("observation.image_omitted")


def observation_text(goal: str, q14: np.ndarray, eef: Dict[str, float], waypoints_remaining: int,
                     prompts_path: Optional[str] = None) -> str:
    return prompt(
        "observation.text",
        prompts_path,
        goal=goal,
        joint_pos=format_joint_pos(q14),
        eef_state=format_eef_state(eef),
        waypoints_remaining=int(waypoints_remaining),
    )


def jpeg_data_uri(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


def build_observation_item(
    goal: str,
    q14: np.ndarray,
    eef: Dict[str, float],
    waypoints_remaining: int,
    step: int,
    frames: Dict[str, bytes],
    detail: str = "high",
    prompts_path: Optional[str] = None,
) -> dict:
    """One `user` message: observation text followed by (label, image) pairs per camera."""
    text = observation_text(goal, q14, eef, waypoints_remaining, prompts_path)
    content: List[dict] = [{"type": "input_text", "text": text}]
    for cam_name, jpeg in frames.items():
        label = prompt("observation.camera_label", prompts_path, camera=cam_name, step=int(step))
        content.append({"type": "input_text", "text": label})
        content.append({"type": "input_image", "image_url": jpeg_data_uri(jpeg), "detail": detail})
    return {"role": "user", "content": content}


def blob_id(jpeg_bytes: bytes) -> str:
    return hashlib.sha1(jpeg_bytes).hexdigest()[:11]


def redact_images(obj: Any) -> Any:
    """Deep-copy `obj` replacing base64 image payloads with `$blob:<sha1>` placeholders (reference log style)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "image_url" and isinstance(v, str) and v.startswith("data:image/"):
                header, _, payload = v.partition(",")
                try:
                    digest = blob_id(base64.b64decode(payload))
                except Exception:
                    digest = hashlib.sha1(payload.encode()).hexdigest()[:11]
                out[k] = f"{header},$blob:{digest}"
            else:
                out[k] = redact_images(v)
        return out
    if isinstance(obj, list):
        return [redact_images(v) for v in obj]
    return copy.deepcopy(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj


def images_kept_before(n_observations: int, keep_last_n: int) -> int:
    """How many of the oldest observations have their images dropped, given `n_observations` so far.

    Dropping images strictly "older than the last N" would move the boundary every single turn, and every
    turn the request prefix would diverge one item earlier - which is exactly where the provider's prompt
    cache stops matching, so the whole image-heavy tail gets re-prefilled each call. Instead the boundary is
    floored to a multiple of N, so it only moves once every N turns and the images in view float between N
    and 2N observations: the same request prefix is reused for N-1 turns out of N.
    """
    if keep_last_n <= 0 or n_observations <= keep_last_n:
        return 0
    return max(0, ((n_observations - keep_last_n) // keep_last_n) * keep_last_n)


def prune_image_history(items: List[dict], keep_last_n: int, prompts_path: Optional[str] = None) -> List[dict]:
    """Drop camera images from the oldest observation messages (0 = keep every image).

    The `input_image` parts are replaced by a short text placeholder so the model still sees that an
    observation happened. How many are dropped comes from `images_kept_before`, which keeps the request
    prefix stable for the provider's prompt cache. Returns a new list; `items` is not modified.
    """
    if keep_last_n <= 0:
        return items
    obs_indices = [
        i
        for i, it in enumerate(items)
        if isinstance(it, dict)
        and it.get("role") == "user"
        and isinstance(it.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "input_image" for p in it["content"])
    ]
    to_prune = set(obs_indices[: images_kept_before(len(obs_indices), keep_last_n)])
    if not to_prune:
        return items
    out = list(items)
    for i in to_prune:
        new_content = []
        for part in out[i]["content"]:
            if isinstance(part, dict) and part.get("type") == "input_image":
                new_content.append({"type": "input_text",
                                    "text": prompt("observation.image_omitted", prompts_path)})
            else:
                new_content.append(part)
        out[i] = {**out[i], "content": new_content}
    return out


def count_images(items: List[dict]) -> int:
    n = 0
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("content"), list):
            n += sum(1 for p in it["content"] if isinstance(p, dict) and p.get("type") == "input_image")
    return n


def summarize_items(items: List[dict]) -> Tuple[int, int]:
    """(number of items, number of images) for logging."""
    return len(items), count_images(items)
