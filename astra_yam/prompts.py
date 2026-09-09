"""Loader for the model-facing text in `configs/PROMPTS.yaml`.

Nothing the model reads is hard-coded in this package: the tool descriptions, the observation message and
the lines the session injects all come from that file (the two long prose files, `SYSTEM_PROMPT.md` and
`TILT_NOTE.md`, are loaded by `astra_yam.embodiment`). Keys are dotted paths into the YAML tree, e.g.
`prompt("tools.move_to.description", bounds=...)`.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import yaml

from astra_yam.config import DEFAULT_PROMPTS


@lru_cache(maxsize=8)
def load_prompts(path: str = DEFAULT_PROMPTS) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping, got {type(data).__name__}")
    return data


def prompt(key: str, path: str | None = None, **fmt: Any) -> str:
    """The text at dotted `key`, with `{...}` placeholders filled from `fmt`.

    Raises KeyError for a missing key and, via `str.format`, for a placeholder the caller did not supply -
    a typo in the YAML fails the next request instead of silently shipping a broken prompt.
    """
    node: Any = load_prompts(path or DEFAULT_PROMPTS)
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"no prompt '{key}' in the prompts file (missing '{part}')")
        node = node[part]
    if not isinstance(node, str):
        raise TypeError(f"prompt '{key}' is a {type(node).__name__}, not text")
    return node.format(**fmt)          # always: an unfilled {placeholder} must fail, not ship as literal text
