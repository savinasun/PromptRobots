"""Per-trial logging: redacted request/response JSON, frames, plain-text transcripts, a notes log, and a summary."""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from astra_yam.observation import redact_images


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return (s[:max_len] or "trial").rstrip("_")


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
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / (name or f"{stamp}_{slugify(goal)}")
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

    @staticmethod
    def indent(text: str, prefix: str) -> str:
        """`prefix` on the first line, aligned blanks on the rest - keeps multi-line reasoning readable."""
        lines = (text or "").rstrip("\n").splitlines() or [""]
        pad = " " * len(prefix)
        return "\n".join((prefix if i == 0 else pad) + line for i, line in enumerate(lines))

    def save_frames(self, step: int, frames: Dict[str, bytes]) -> Dict[str, str]:
        paths = {}
        for cam, jpeg in frames.items():
            p = self.dir / "frames" / f"step_{int(step):05d}_{cam}.jpg"
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

    def finish(self, summary: dict) -> None:
        self.write_json("summary.json", summary)
        self._notes.write(f"\n## Outcome\n\n```\n{json.dumps(summary, indent=2, cls=_Encoder)}\n```\n")
        self.text_section("OUTCOME", json.dumps(summary, indent=2, cls=_Encoder))
        self._notes.close()
        self._transcript.close()
        self._text.close()
        self._notes_only.close()
