"""Which simulations have been run: a small JSON file, `.fbh-progress.json` in the current directory.

Set FBH_PROGRESS to keep it somewhere else (running `fbh` from another folder otherwise starts a fresh one).
"""
import json
import os
import sys
from pathlib import Path


def path() -> Path:
    return Path(os.environ.get("FBH_PROGRESS", ".fbh-progress.json"))


def load() -> list[str]:
    """Ids run so far, in the order they were first run. A missing file means none; an unreadable or
    malformed one also means none, but says so on stderr (the next `fbh run` rewrites it)."""
    target = path()
    try:
        ids = json.loads(target.read_text())["run"]
        if not isinstance(ids, list):
            raise TypeError("run is not a list")
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError, TypeError):
        print(f"warning: {target} is not a valid progress file, counting nothing as run; "
              "the next `fbh run` rewrites it", file=sys.stderr)
        return []
    return [i for i in ids if isinstance(i, str)]


def mark(sid: str) -> None:
    ids = load()
    if sid not in ids:
        ids.append(sid)
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"run": ids}) + "\n")
