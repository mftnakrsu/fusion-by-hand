"""Seeded scenarios: the truth and the noise a simulation runs on, as plain lists."""
import numpy as np

from .kf import cv_model


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


def model_truth(seed: int = 1, n: int = 400, q: float = 0.01, maneuver: int | None = None, dv: float = 4.0,
                gap: tuple[int, int] | None = None) -> dict:
    """A target that obeys exactly the model the filter assumes: x_k = F x_{k-1} + w_k, w_k ~ N(0, Q(q)).

    Same keys as `tracking`. The start is drawn from the filter's own prior N(x0, P0), so a filter
    with R equal to the sensor's is consistent from frame 1 and its NIS has mean 1 for any seed.
    With `maneuver` set, the velocity also jumps by `dv` px/frame at that frame; the random draws
    stay the same, so that truth differs from the calm one only by that deterministic step.
    """
    F, _, Q, _ = cv_model(1.0, q, 1.0)
    x0, P0 = [200, 7], [[16, 0], [0, 9]]
    rng = np.random.default_rng(seed)
    start, w, noise = rng.standard_normal(2), rng.standard_normal((n, 2)), rng.standard_normal(n)
    x = np.array(x0, dtype=float) + np.linalg.cholesky(np.array(P0, dtype=float)) @ start
    L = np.linalg.cholesky(Q)
    p, v = [], []
    for k in range(1, n + 1):
        x = F @ x + L @ w[k - 1]
        if k == maneuver:
            x = x + np.array([0.0, dv])
        p.append(float(x[0]))
        v.append(float(x[1]))
    return {"p": p, "v": v, "noise": noise.tolist(), "gap": None if gap is None else [int(gap[0]), int(gap[1])],
            "maneuver": maneuver, "x0": x0, "P0": P0, "q": q}
