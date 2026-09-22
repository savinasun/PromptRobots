"""Per-trial logging: redacted request/response JSON, frames, plain-text transcripts, a notes log, and a summary."""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.observation import redact_images


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return (s[:max_len] or "trial").rstrip("_")


# A trial directory is named <yyyymmdd>_<HHmmss>_<success|fail> and is filed under one of these when it
# finishes. Only a run killed before `finish` is left loose in the log root.
SUCCEEDED_DIR = "succeeded"
RECYCLED_DIR = "recycled"

# Each filing folder keeps an index of what has been filed into it, beside it in the log root.
MAP_FILES = {SUCCEEDED_DIR: ".succeeded_map.json", RECYCLED_DIR: ".recycled_map.json"}


def record_filing(log_root: Path, folder: str, name: str) -> None:
    """Append a newly filed trial directory to its folder's index.

    Best-effort: a missing, empty or corrupt index is rebuilt from this call onward rather than failing a
    trial that has already finished. Use `rebuild_map` to restore one from the directory itself.
    """
    path = Path(log_root) / MAP_FILES[folder]
    try:
        entries = json.loads(path.read_text())
        if not isinstance(entries, list):
            entries = []
    except (OSError, ValueError):
        entries = []
    if name not in entries:
        entries.append(name)
    try:
        path.write_text(json.dumps(entries, indent=2))
    except OSError:
        pass


def rebuild_map(log_root: Path, folder: str) -> list:
    """Rewrite a folder's index from what is actually on disk, sorted; returns the entries."""
    root = Path(log_root)
    entries = sorted(d.name for d in (root / folder).iterdir() if d.is_dir()) if (root / folder).is_dir() else []
    (root / MAP_FILES[folder]).write_text(json.dumps(entries, indent=2))
    return entries


def unique_dir(root: Path, name: str) -> Path:
    """`root/name`, suffixed _2, _3, ... if runs collide within the same second."""
    candidate, n = root / name, 2
    while candidate.exists():
        candidate, n = root / f"{name}_{n}", n + 1
    return candidate


class _Encoder(json.JSONEncoder):
    def default(self, o):  # numpy scalars / arrays
        try:
            import numpy as np

            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
        except ImportError:
            pass
        return str(o)


class TrialLogger:
    def __init__(self, root: str, goal: str, name: Optional[str] = None):
        # The directory is born `_fail` and is promoted to `_success` only by a clean `done` in `finish`.
        # Naming it pessimistically is what makes a killed run self-describing: SIGKILL runs none of our
        # code, so whatever the directory is called at that moment is the name it keeps.
        self.stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.auto_named = name is None
        Path(root).mkdir(parents=True, exist_ok=True)
        self.dir = Path(root) / name if name else unique_dir(Path(root), f"{self.stamp}_fail")
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "frames").mkdir()
        (self.dir / "requests").mkdir()
        (self.dir / "responses").mkdir()
        self._t0 = time.perf_counter()
        self._transcript = open(self.dir / "transcript.jsonl", "a", buffering=1)
        self._notes = open(self.dir / "notes.md", "a", buffering=1)
        self._notes.write(f"# Trial: {goal}\n\nStarted {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
        # transcript.txt is the whole conversation as plain text - no base64, no JSON envelopes - so a run can
        # be read start to finish, including Astra's reasoning summaries.
        self._text = open(self.dir / "transcript.txt", "a", buffering=1)
        # transcript_notes.txt is Astra's words and nothing else: one numbered entry per `note`, then the
        # closing summary/hindsight - the run as narration, for reading without the machinery around it.
        self._notes_only = open(self.dir / "transcript_notes.txt", "a", buffering=1)
        self._notes_only.write(f"# {goal}\n")
        self._note_index = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def event(self, kind: str, **payload: Any) -> None:
        rec = {"t": dt.datetime.now().isoformat(timespec="milliseconds"), "elapsed_s": round(self.elapsed, 3),
               "kind": kind, **payload}
        self._transcript.write(json.dumps(redact_images(rec), cls=_Encoder) + "\n")

    def note(self, text: str) -> None:
        self._notes.write(f"- [{self.elapsed:7.1f}s] {text}\n")

    # ------------------------------------------------------------ transcript.txt
    WIDTH = 100

    def text_section(self, title: str, body: str = "", stamp: bool = True) -> None:
        """One block of the plain-text transcript: a ruled header, then `body` verbatim."""
        head = f"{title} [{self.elapsed:.1f}s]" if stamp else title
        rule = "-" * max(4, self.WIDTH - len(head) - 5)
        self._text.write(f"\n--- {head} {rule}\n")
        if body:
            self._text.write(body.rstrip("\n") + "\n")

    def text_block(self, body: str) -> None:
        """More text under the current section (no header)."""
        if body:
            self._text.write(body.rstrip("\n") + "\n")

    def astra_note(self, text: str, label: str) -> None:
        """One numbered entry in the note-only transcript. `label` anchors it to the full transcript."""
        self._note_index += 1
        head = f"{self._note_index:3d}. [{label}] "
        self._notes_only.write(self.indent((text or "").strip() or "(no note)", head) + "\n")

    def astra_note_extra(self, text: str, label: str) -> None:
        """An aligned continuation of the entry just written (the closing `hindsight`). Skipped when empty."""
        if text and text.strip():
            self._notes_only.write(self.indent(text.strip(), f"{'':5}{label}: ") + "\n")

    def compare_note(self, call_index: int, lines: Dict[str, str]) -> None:
        """notes_compare.txt: per call, one entry per model (controlling model first) - the side-by-side view."""
        if not hasattr(self, "_compare"):
            self._compare = open(self.dir / "notes_compare.txt", "a", buffering=1)
        self._compare.write(f"\n## call {call_index} [{self.elapsed:.1f}s]\n")
        width = max(len(k) for k in lines)
        for model, text in lines.items():
            self._compare.write(self.indent(text, f"{model:<{width}} : ") + "\n")

    @staticmethod
    def indent(text: str, prefix: str) -> str:
        """`prefix` on the first line, aligned blanks on the rest - keeps multi-line reasoning readable."""
        lines = (text or "").rstrip("\n").splitlines() or [""]
        pad = " " * len(prefix)
        return "\n".join((prefix if i == 0 else pad) + line for i, line in enumerate(lines))

    def save_frames(self, step: int, frames: Dict[str, bytes], sequence: Optional[int] = None) -> Dict[str, str]:
        paths = {}
        for cam, jpeg in frames.items():
            suffix = f"_obs_{sequence:05d}" if sequence is not None else ""
            p = self.dir / "frames" / f"step_{int(step):05d}{suffix}_{cam}.jpg"
            p.write_bytes(jpeg)
            paths[cam] = str(p.relative_to(self.dir))
        return paths

    def log_request(self, call_index: int, request: dict) -> None:
        p = self.dir / "requests" / f"request_{call_index:04d}.json"
        p.write_text(json.dumps(redact_images(request), indent=2, cls=_Encoder, ensure_ascii=False))

    def log_response(self, call_index: int, payload: dict) -> None:
        p = self.dir / "responses" / f"response_{call_index:04d}.json"
        p.write_text(json.dumps(payload, indent=2, cls=_Encoder, ensure_ascii=False))

    def write_json(self, name: str, payload: Any) -> None:
        (self.dir / name).write_text(json.dumps(payload, indent=2, cls=_Encoder, ensure_ascii=False))

    def settle(self, status: Optional[str]) -> Path:
        """Rename the trial directory to its outcome and file it under `succeeded/` or `recycled/`.

        Only `done` counts as success: a `give_up`, a budget exhaustion or a crash all leave `_fail`. A run
        killed mid-flight never reaches this and stays loose in the log root under the `_fail` name it was
        born with, which is how a kill is told apart from a finished failure. A directory named explicitly
        by the caller is left alone.
        """
        if not self.auto_named:
            return self.dir
        success = status == "done"
        root = self.dir.parent / (SUCCEEDED_DIR if success else RECYCLED_DIR)
        root.mkdir(parents=True, exist_ok=True)
        desired = f"{self.stamp}_{'success' if success else 'fail'}"
        self.dir = Path(self.dir.rename(unique_dir(root, desired)))
        record_filing(root.parent, root.name, self.dir.name)
        return self.dir

    def finish(self, summary: dict) -> None:
        self._notes.write(f"\n## Outcome\n\n```\n{json.dumps(summary, indent=2, cls=_Encoder)}\n```\n")
        self.text_section("OUTCOME", json.dumps(summary, indent=2, cls=_Encoder))
        self._notes.close()
        self._transcript.close()
        self._text.close()
        self._notes_only.close()
        # Rename first, so summary.json records where the run actually ended up.
        self.settle(summary.get("status"))
        self.write_json("summary.json", {**summary, "log_dir": str(self.dir)})
