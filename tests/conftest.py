import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def reference_transcript(name: str):
    """Load one of the reference-trial JSON transcripts, or skip the module if it is not in the repo.

    They used to live in `prompts/`; that directory was reorganized into `configs/`, so both are searched.
    """
    import json

    import pytest

    for folder in ("configs", "prompts"):
        path = ROOT / folder / name
        if path.exists():
            return json.loads(path.read_text())
    pytest.skip(f"reference transcript {name} not found in configs/ or prompts/", allow_module_level=True)
