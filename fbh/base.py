"""What every simulation shares: the options it may read, the result it returns, and the errors it may raise."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import plot


@dataclass
class Options:
    q: float | None = None
    r: float | None = None
    gap: tuple[int, int] | None = None
    seed: int | None = None
    sigma_r: float | None = None            # lesson 2: the radar's range noise, m
    sigma_theta: float | None = None        # lesson 2: the radar's bearing noise, rad
    lang: str | None = None                 # None is "en"


@dataclass
class Result:
    table: list[dict]
    summary: dict
    plot: Callable[..., Path]               # plot(out_dir, lang=None) -> <out_dir>/<id>.png
    digits: dict = field(default_factory=dict)   # summary keys the CLI prints with more than its usual 3 decimals


class OptionError(ValueError):
    """An option the simulation cannot honour (a gap that does not fit, ...); the CLI shows it as a usage error."""


def _pick(value, default):
    return default if value is None else value


def _plain(obj):
    """numpy scalars and arrays to plain Python, so tables and summaries can go straight to JSON."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _result(sid: str, data: dict, table: list[dict], summary: dict, opts: Options, digits: dict | None = None) -> Result:
    def draw(out_dir, lang=None):
        return plot.save(sid, data, Path(out_dir), lang or opts.lang or "en")
    return Result(_plain(table), _plain(summary), draw, dict(digits or {}))
