"""Seeded scenarios: the truth and the noise a simulation runs on, as plain lists."""
import numpy as np


def tracking(seed: int = 7, n: int = 60, gap: tuple[int, int] | None = (20, 27), maneuver: int = 36) -> dict:
    """A target the camera follows: 10 px/frame, then 14 px/frame from frame `maneuver`.

    Frames are 1-based, p_k = p_{k-1} + v_k from p_0 = 200. `gap` is the inclusive range of frames
    where the detector finds nothing (None = never). `noise` is unit normal; a simulation scales it
    by sqrt(R), so changing R rescales the very same draws.
    """
    v = [10.0 if k < maneuver else 14.0 for k in range(1, n + 1)]
    p, pos = [], 200.0
    for vk in v:
        pos += vk
        p.append(pos)
    return {
        "p": p,
        "v": v,
        "noise": np.random.default_rng(seed).standard_normal(n).tolist(),
        "gap": None if gap is None else [int(gap[0]), int(gap[1])],
        "maneuver": maneuver,
        "x0": [200, 7],                    # the filter starts 3 px/frame low, and P0 admits it
        "P0": [[16, 0], [0, 9]],
    }
