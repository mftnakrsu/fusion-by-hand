"""`fbh export <dir> --lesson 2`: the numbers the browser figures of lesson 2 draw, as compact JSON.

Python is the single source here too. The fly-by and the crossing are exported as finished runs, so the
browser only draws them. The single updates (sections 2, 3, 5 and 9) are small enough to compute in the
browser, and ekf-cases.json holds the same calculations from Python, so a test on the site can compare the
two. Lengths are metres, angles radians, scans are 1-based and the JSON arrays 0-based (index = scan - 1).
The state is [px, py, vx, vy]; a report is [range, bearing] with the bearing in [-pi, pi).

- flyby.json: `n`, `dt`, `seed`, `q`, `sigma_r`, `sigma_theta`, the truth `p` and `v`, the prior `x0`, `P0`,
  and per scan `z` (the report), `raw` (the report as a point), `z_hat` (the report the prediction expected),
  `xm`, `Pm` (the prediction), `x`, `P` (after the update), `y`, `S`, `K`, `nis`, `error`, `claimed`,
  `sigma_along`, `sigma_across`; `bound` (5.99) and `summary`.
- wrap.json: `n`, `p`, `v`, `x0`, `P0`, `z`, `jump_scan`, `jump_scans`, `raw`, `wrapped`, and the two runs
  `with_wrap` and `without_wrap`, each with per scan `z_hat`, `xm`, `x`, `P`, `y`, `nis`, `error`; `summary`.
- ekf-cases.json: `tangent` (section 2), `range` (section 3), `update` (sections 5 and 9, the EKF's answer)
  and `exact` (the same updates by Bayes' rule on a polar grid), each a list of inputs with their outputs.
"""
import json
from pathlib import Path

import numpy as np

from . import ekf, sims2
from .base import Options, _plain
from .export import _digits


def flyby_json() -> dict:
    d = sims2.flyby()
    run, scn = d["run"], d["scn"]
    summary = sims2.SIMS["ekf-07-flyby"](Options()).summary
    return _digits(_plain({
        "n": len(scn["p"]), "dt": 1, "seed": d["seed"], "q": d["q"], "sigma_r": d["sigma_r"],
        "sigma_theta": d["sigma_theta"], "bound": d["bound"],
        **{key: scn[key] for key in ("p", "v", "x0", "P0")}, "raw": d["raw"],
        **{key: run[key] for key in ("z", "z_hat", "xm", "Pm", "x", "P", "y", "S", "K", "nis")},
        **{key: d[key] for key in ("error", "claimed", "sigma_along", "sigma_across")}, "summary": summary}))


def wrap_json() -> dict:
    d = sims2.wrap_demo()
    scn = d["scn"]
    summary = sims2.SIMS["ekf-08-wrap"](Options()).summary
    pick = lambda run, error: {**{key: run[key] for key in ("z_hat", "xm", "x", "P", "y", "nis")}, "error": error}
    return _digits(_plain({
        "n": len(scn["p"]), **{key: scn[key] for key in ("p", "v", "x0", "P0")}, "z": d["wrapped_run"]["z"],
        **{key: d[key] for key in ("jump_scan", "jump_scans", "raw", "wrapped")},
        "with_wrap": pick(d["wrapped_run"], d["error_wrapped"]),
        "without_wrap": pick(d["plain_run"], d["error_plain"]), "summary": summary}))


def exact_radar_update(x, prior_sigma: float, dr: float, dtheta: float, sigma_r: float, sigma_theta: float,
                       n: int = 161, span: float = 6.0) -> dict:
    """The update of sims2.radar_update by Bayes' rule, without the tangent: mean and covariance of
    prior times likelihood, integrated on an n by n polar grid of +-span sigmas around the report."""
    xv = np.array(x, dtype=float)
    r_z = np.hypot(*xv) + dr
    th_z = np.arctan2(xv[1], xv[0]) + dtheta
    r = r_z + sigma_r * np.linspace(-span, span, n)
    th = th_z + sigma_theta * np.linspace(-span, span, n)
    rr, tt = np.meshgrid(r, th, indexing="ij")
    px, py = rr * np.cos(tt), rr * np.sin(tt)
    w = (np.exp(-0.5 * ((px - xv[0]) ** 2 + (py - xv[1]) ** 2) / prior_sigma**2
                - 0.5 * ((rr - r_z) / sigma_r) ** 2 - 0.5 * ((tt - th_z) / sigma_theta) ** 2) * rr)
    w = np.where(rr > 0, w, 0.0)
    w /= w.sum()
    mean = np.array([np.sum(w * px), np.sum(w * py)])
    ex, ey = px - mean[0], py - mean[1]
    cov = np.array([[np.sum(w * ex * ex), np.sum(w * ex * ey)], [np.sum(w * ex * ey), np.sum(w * ey * ey)]])
    return {"n": n, "span": span, "mean": mean, "cov": cov, "range_of_mean": float(np.hypot(*mean))}


UPDATE_GRID = {
    "x": ((300.0, 400.0), (-350.0, 2.5)),
    "prior_sigma": (10.0, 100.0, 300.0),
    "reading": ((10.0, 0.02), (0.0, 0.4), (-20.0, -0.1)),            # (dr, dtheta) beyond the expected report
    "sigmas": ((5.0, 0.02), (0.5, 0.1)),                              # (sigma_r, sigma_theta)
}


def cases_json() -> dict:
    tangent = [{key: d[key] for key in ("x_hat", "sigma", "offset", "h", "H", "lin_std", "true_mean", "true_std")}
               for d in (sims2.tangent(x_hat, sigma) for x_hat in (300.0, 100.0, 30.0, 0.0, -300.0)
                         for sigma in (10.0, 50.0, 100.0))]
    range_keys = ("x_minus", "P_minus", "z", "R", "offset", "h", "H", "y", "S", "K", "x_plus", "P_plus",
                  "exact_mean", "exact_var")
    ranges = [{key: d[key] for key in range_keys}
              for d in (sims2.range_only_update(x=x, P=P, z=float(np.hypot(x, sims2.OFFSET)) + dz, sigma_r=sr)
                        for x in (300.0, 30.0) for P in (100.0, 2500.0) for dz in (6.0, -4.0) for sr in (2.0, 5.0))]
    update, exact = [], []
    keys = ("H", "z_hat", "z", "y", "S", "K", "x_plus", "P_plus", "nis", "var_along", "var_across", "range_plus")
    for x in UPDATE_GRID["x"]:
        for prior_sigma in UPDATE_GRID["prior_sigma"]:
            for dr, dtheta in UPDATE_GRID["reading"]:
                for sigma_r, sigma_theta in UPDATE_GRID["sigmas"]:
                    given = {"x": list(x), "prior_sigma": prior_sigma, "dr": dr, "dtheta": dtheta,
                             "sigma_r": sigma_r, "sigma_theta": sigma_theta}
                    d = sims2.radar_update(x, prior_sigma, dr, dtheta, sigma_r, sigma_theta)
                    update.append({**given, **{key: d[key] for key in keys}})
                    exact.append({**given, **exact_radar_update(x, prior_sigma, dr, dtheta, sigma_r, sigma_theta)})
    return _digits(_plain({"tangent": tangent, "range": ranges, "update": update, "exact": exact}))


FILES = {
    "flyby.json": flyby_json,
    "wrap.json": wrap_json,
    "ekf-cases.json": cases_json,
}


def write_all(out_dir) -> list[Path]:
    """Write the three files into out_dir (created if needed) and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in FILES.items():
        path = out_dir / name
        path.write_text(json.dumps(build(), separators=(",", ":"), allow_nan=False) + "\n")
        paths.append(path)
    return paths
