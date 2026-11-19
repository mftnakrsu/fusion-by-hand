"""`fbh export <dir>`: the numbers the essay's browser figures draw, written as compact JSON.

Python is the single source: the browser reruns the filter on tracking.json and a test compares it
with kf-cases.json, so the two can never drift apart. tracking.json and kf-cases.json are written
at full float precision; the figures' display data (derivative, covariance, nis) at 10 digits.
README.md, section "Export format", is the same text with all the keys; a test writes a filter from
that wording alone and checks it against kf-cases.json.

What a port must do (state = [position, velocity], dt = 1, frames are 1-based, JSON arrays are 0-based:
the array index is frame - 1):

- F = [[1, 1], [0, 1]], H = [1, 0], R = r, Q = q * [[1/3, 1/2], [1/2, 1]].
- `noise` in tracking.json is unit normal; the measurement of frame k is z = p[k - 1] + sqrt(r) * noise[k - 1].
- Start from the frame-0 prior x0, P0. Every frame, frame 1 included, first predicts:
  x = F x, P = F P Fᵀ + Q.
- Then, unless the frame is in the gap, updates: y = z - H x, S = H P Hᵀ + R, K = P Hᵀ / S,
  x = x + K y, P = (I - K H) P (the plain form, not Joseph, and P is not symmetrised afterwards).
- `gap` [a, b] is inclusive: frames a to b are predict-only, so z, y, S, K and NIS do not exist there
  and x, P are the prediction.
- kf-cases.json exports only x, P and K, one entry per frame (index = frame - 1); there are no z, y, S or
  NIS arrays. x and P are after the update (the prediction in a gap frame); K is null in a gap frame.
- Floats agree to about 1e-9 relative, not bit for bit; compare with a tolerance (P comes out
  symmetric only up to the last digits).
"""
import json
from pathlib import Path

from . import scenarios, sims

KF_Q = (0.001, 0.01, 1, 10)
KF_R = (1, 4, 25)
SAMPLES = 2000                              # error samples in covariance.json


def _digits(obj, n=10):
    """Floats rounded to n significant digits, recursively (compact files for display data)."""
    if isinstance(obj, float):
        return float(f"{obj:.{n}g}")
    if isinstance(obj, dict):
        return {k: _digits(v, n) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_digits(v, n) for v in obj]
    return obj


def tracking_json() -> dict:
    scn = scenarios.tracking()
    return {"n": len(scn["p"]), **{k: scn[k] for k in ("gap", "maneuver", "p", "v", "noise", "x0", "P0")}}


def kf_cases_json() -> dict:
    """x, P (after each frame's update; the prediction in a gap frame) and K (null in a gap frame)."""
    with_gap, without_gap = scenarios.tracking(), scenarios.tracking(gap=None)
    cases = []
    for q in KF_Q:
        for r in KF_R:
            for gap in (True, False):
                run = sims.run_cv(with_gap if gap else without_gap, q, r)
                cases.append({"q": q, "r": r, "gap": gap, "x": run["x"], "P": run["P"], "K": run["K"]})
    return {"cases": cases}


def derivative_json() -> dict:
    d = sims.derivative_vs_filter()
    summary = sims.SIMS["05-derivative-vs-filter"](sims.Options()).summary
    return _digits({"k": d["k"], "truth_p": d["truth_p"], "truth_v": d["truth_v"], "z": d["z"].tolist(),
                    "derivative": d["derivative"].tolist(), "filter": d["filter"].tolist(), "summary": summary})


def covariance_json() -> dict:
    c = sims.covariance_cloud(n=SAMPLES)
    return _digits({"n": SAMPLES, "F": c["F"].tolist(), "before": c["e_before"].tolist(), "after": c["e_after"].tolist(),
                    "P_before": c["P_before_formula"].tolist(), "P_after_formula": c["P_after_formula"].tolist()})


def nis_json() -> dict:
    d = sims.nis()
    summary = sims.SIMS["14-nis"](sims.Options()).summary
    return _digits({"k": d["k"], "bound": d["bound"], "maneuver": d["maneuver"], "window": d["window"],
                    "honest": d["honest"].tolist(), "overconfident": d["overconfident"].tolist(),
                    "q_small": d["q_small"].tolist(), "summary": summary})


FILES = {
    "tracking.json": tracking_json,
    "kf-cases.json": kf_cases_json,
    "derivative.json": derivative_json,
    "covariance.json": covariance_json,
    "nis.json": nis_json,
}


def write_all(out_dir) -> list[Path]:
    """Write the five files into out_dir (created if needed) and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in FILES.items():
        path = out_dir / name
        path.write_text(json.dumps(build(), separators=(",", ":"), allow_nan=False) + "\n")
        paths.append(path)
    return paths
