"""Which simulations have been run: a small JSON file, `.fbh-progress.json` unless FBH_PROGRESS says otherwise."""
import json
import os
from pathlib import Path


def path() -> Path:
    return Path(os.environ.get("FBH_PROGRESS", ".fbh-progress.json"))


def load() -> list[str]:
    """Ids run so far, in the order they were first run. A missing or unreadable file means none."""
    try:
        ids = json.loads(path().read_text())["run"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [i for i in ids if isinstance(i, str)] if isinstance(ids, list) else []


def mark(sid: str) -> None:
    ids = load()
    if sid not in ids:
        ids.append(sid)
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"run": ids}) + "\n")
