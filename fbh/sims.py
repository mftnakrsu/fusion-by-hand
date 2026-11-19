"""The simulations of lesson 1. Each one is a seeded run of the filter in kf.py.

Every id in SIMS maps an Options to a Result (table, summary, plot). A simulation reads only the
options it has a use for and ignores the others; None always means "the simulation's own default".
The functions below it (velocity_learning, nis, ...) are the same runs with their raw numbers.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import plot, scenarios
from .kf import cv_model, fuse, predict, update

P_EXAMPLE = [[2.5, 1.0], [1.0, 0.5]]        # the covariance worked out by hand in section 8
CHI2_95_1DOF = 3.84                         # 95 % bound of a chi-square with one degree of freedom


@dataclass
class Options:
    q: float | None = None
    r: float | None = None
    gap: tuple[int, int] | None = None
    seed: int | None = None
    lang: str | None = None                 # None is "en"


@dataclass
class Result:
    table: list[dict]
    summary: dict
    plot: Callable[..., Path]               # plot(out_dir, lang=None) -> <out_dir>/<id>.png


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


def _result(sid: str, data: dict, table: list[dict], summary: dict, opts: Options) -> Result:
    def draw(out_dir, lang=None):
        return plot.save(sid, data, Path(out_dir), lang or opts.lang or "en")
    return Result(_plain(table), _plain(summary), draw)


# --- the filter on a scenario ---------------------------------------------------------

def run_cv(scn: dict, q: float, r: float, r_true: float | None = None) -> dict:
    """Run the constant-velocity filter over a scenario, frames k = 1..n (predict, then update).

    Per frame: xm/Pm = prediction, x/P = after the update (the prediction again in a gap frame),
    z, y, S, K, nis = the measurement and what the update made of it (None in a gap frame).
    The measurement is z = p + sqrt(r_true) * noise, with r_true = r unless the filter is told a
    different R than the sensor really has. Everything returned is plain lists, floats and None.
    """
    F, H, Q, R = cv_model(1.0, q, r)
    sigma = float(np.sqrt(r if r_true is None else r_true))
    x = np.array(scn["x0"], dtype=float).reshape(2, 1)
    P = np.array(scn["P0"], dtype=float)
    gap = scn["gap"]
    out = {key: [] for key in ("xm", "Pm", "x", "P", "y", "S", "K", "nis", "z")}
    for k, (p, e) in enumerate(zip(scn["p"], scn["noise"]), start=1):
        x, P = predict(x, P, F, Q)
        out["xm"].append(x.ravel().tolist())
        out["Pm"].append(P.tolist())
        if gap is not None and gap[0] <= k <= gap[1]:
            z = y = S = K = nis_k = None
        else:
            z = float(p + sigma * e)
            u = update(x, P, np.array([[z]]), H, R)
            x, P = u.x, u.P
            y, S, K, nis_k = float(u.y[0, 0]), float(u.S[0, 0]), u.K.ravel().tolist(), float(u.nis)
        out["x"].append(x.ravel().tolist())
        out["P"].append(P.tolist())
        for key, value in (("z", z), ("y", y), ("S", S), ("K", K), ("nis", nis_k)):
            out[key].append(value)
    return out


# --- the runs behind the figures ------------------------------------------------------

def fuse_two(z1: float = 20.0, var1: float = 1.0, z2: float = 30.0, var2: float = 9.0) -> dict:
    """Section 1: two Gaussian beliefs and their inverse-variance fusion, on a grid."""
    mean, var = fuse(z1, var1, z2, var2)
    s1, s2 = np.sqrt(var1), np.sqrt(var2)
    grid = np.linspace(min(z1 - 4 * s1, z2 - 4 * s2), max(z1 + 4 * s1, z2 + 4 * s2), 601)
    pdf = lambda m, v: np.exp(-0.5 * (grid - m) ** 2 / v) / np.sqrt(2 * np.pi * v)
    return {"z1": z1, "var1": var1, "z2": z2, "var2": var2, "mean": mean, "var": var,
            "grid": grid, "pdf1": pdf(z1, var1), "pdf2": pdf(z2, var2), "pdf_fused": pdf(mean, var)}


def one_d_loop(x: float = 200.0, P: float = 4.0, u: float = 10.0, Q: float = 5.0, R: float = 1.0,
               z: float = 216.0, frames: int = 5) -> dict:
    """Section 3 by hand: frame 1 is the starting belief, the camera sees z in frame 2, then nothing."""
    rows = [{"k": 1, "x_minus": None, "P_minus": None, "z": None, "K": None, "x_plus": x, "P_plus": P}]
    for k in range(2, frames + 1):
        x, P = x + u, P + Q                                   # predict
        row = {"k": k, "x_minus": x, "P_minus": P, "z": None, "K": None}
        if k == 2:                                            # update, only in frame 2
            K = P / (P + R)
            x, P = x + K * (z - x), (1 - K) * P
            row["z"], row["K"] = z, K
        rows.append({**row, "x_plus": x, "P_plus": P})
    return {"rows": rows, "u": u, "Q": Q, "R": R}


def derivative_vs_filter(seed: int = 0, n: int = 200, r: float = 9.0, q: float = 0.01) -> dict:
    """Section 5: a target that does not move at all, seen through a box that jitters by sigma = sqrt(r) px.

    Velocity from the difference of two measurements against the filter's velocity, frames 2..n.
    The difference of two independent errors has std sqrt(2) * sigma / dt (`theory_derivative`);
    `std_position` is the measured scatter of the position itself, which the difference exceeds.
    n is large enough (200) that the measured stds sit close to these values for any seed.
    """
    sigma, truth_p = float(np.sqrt(r)), 250.0
    z = truth_p + sigma * np.random.default_rng(seed).standard_normal(n)
    F, H, Q, R = cv_model(1.0, q, r)
    x, P = np.array([[z[0]], [0.0]]), np.diag([r, r])        # start from the first box, velocity unknown
    filt = []
    for k in range(1, n):
        x, P = predict(x, P, F, Q)
        u = update(x, P, np.array([[z[k]]]), H, R)
        x, P = u.x, u.P
        filt.append(float(x[1, 0]))
    deriv = np.diff(z)
    return {"k": list(range(2, n + 1)), "z": z[1:], "derivative": deriv, "filter": np.array(filt),
            "sigma": sigma, "std_position": float(np.std(z - truth_p)), "theory_derivative": float(np.sqrt(2) * sigma),
            "std_derivative": float(np.std(deriv)), "std_filter": float(np.std(filt)),
            "truth_p": truth_p, "truth_v": 0.0, "r": r, "q": q}


def covariance_cloud(n: int = 20000, seed: int = 0, P=P_EXAMPLE, dt: float = 1.0) -> dict:
    """Sections 8 and 9: errors drawn from N(0, P) (the P of section 8), pushed through F; the cloud's covariance is F P Fᵀ (section 9)."""
    P = np.array(P, dtype=float)
    F = cv_model(dt, 0.0, 1.0)[0]
    e = np.random.default_rng(seed).standard_normal((n, 2)) @ np.linalg.cholesky(P).T
    after = e @ F.T
    return {"e_before": e, "e_after": after, "F": F,
            "P_before_formula": P, "P_before_empirical": e.T @ e / n,
            "P_after_formula": F @ P @ F.T, "P_after_empirical": after.T @ after / n}


def velocity_learning(q: float = 0.01, r: float = 1.0, seed: int = 0, n: int = 30) -> dict:
    """Section 12, the loop as printed: truth v = 13, the filter starts believing 10, camera sees position only."""
    dt = 1.0
    F, H, Q, R = cv_model(dt, q, r)
    rng = np.random.default_rng(seed)
    truth_p, truth_v = 200.0, 13.0
    x = np.array([[200.0], [10.0]])
    P = np.diag([4.0, 4.0])
    start = {"k": 0, "v": float(x[1, 0]), "sigma_v": float(np.sqrt(P[1, 1]))}
    rows = []
    for k in range(1, n + 1):
        truth_p += truth_v * dt
        z = np.array([[truth_p + rng.normal(0, np.sqrt(r))]])
        x, P = predict(x, P, F, Q)
        u = update(x, P, z, H, R)
        x, P = u.x, u.P
        rows.append({"k": k, "v": float(x[1, 0]), "sigma_v": float(np.sqrt(P[1, 1])), "K_v": float(u.K[1, 0])})
    return {"rows": rows, "start": start, "truth_v": truth_v, "q": q, "r": r,
            "std_v_10_30": float(np.std([row["v"] for row in rows if 10 <= row["k"] <= 30]))}


def maneuver_response(seed: int = 7, q_small: float = 0.01, q_big: float = 1.0, r: float = 4.0,
                      gap=None, band: float = 1.0) -> dict:
    """Section 13: the same camera on a target that speeds up from 10 to 14 px/frame, for a calm and an agile filter.

    `settle` counts the frames from the maneuver until the velocity estimate is within `band` of the
    new speed and stays there to the end of the run (None if it is still outside in the last frame).
    `reach` is the first frame it touches the band, however briefly. `calm_std` is the estimate's
    scatter over frames 10..maneuver-1, while the target flies straight.
    """
    scn = scenarios.tracking(seed=seed, gap=gap)
    m, new_v = scn["maneuver"], scn["v"][-1]
    out = {"scn": scn, "r": r, "band": band, "new_v": new_v}
    for name, q in (("small", q_small), ("big", q_big)):
        run = run_cv(scn, q, r)
        v = [x[1] for x in run["x"]]
        inside = [abs(vk - new_v) <= band for vk in v]
        reach = next((k - m for k in range(m, len(v) + 1) if inside[k - 1]), None)
        settle = None
        for k in range(len(v), m - 1, -1):                    # walk back from the end while it stays inside
            if not inside[k - 1]:
                break
            settle = k - m
        out[name] = {"q": q, "v": v, "reach": reach, "settle": settle, "calm_std": float(np.std(v[9:m - 1])),
                     "sigma_v": [float(np.sqrt(P[1][1])) for P in run["P"]]}
    return out


def nis(seed: int = 1, n: int = 400, q: float = 0.01, r: float = 4.0, r_sensor: float = 20.0,
        maneuver: int | None = None, dv: float = 4.0, window: int = 20, gap=None) -> dict:
    """Section 14, three runs of the same filter (q, R = r) on targets that obey its own model.

    honest:         the sensor really has R = r, so NIS has mean 1 (chi-square, one degree of freedom).
    overconfident:  same target, but the sensor really has R = r_sensor: the filter claims more than it knows.
    q_small:        the sensor is right, but the target jumps by dv px/frame at frame `maneuver` (default n/2),
                    which q cannot explain: NIS spikes for about `window` frames after the jump, nowhere else.
    """
    maneuver = n // 2 if maneuver is None else maneuver
    calm = scenarios.model_truth(seed=seed, n=n, q=q, gap=gap)
    jump = scenarios.model_truth(seed=seed, n=n, q=q, maneuver=maneuver, dv=dv, gap=gap)
    runs = {"honest": run_cv(calm, q, r), "overconfident": run_cv(calm, q, r, r_true=r_sensor),
            "q_small": run_cv(jump, q, r)}
    keep = [k for k, v in enumerate(runs["honest"]["nis"]) if v is not None]
    frames = np.array([k + 1 for k in keep])
    out = {"k": frames.tolist(), "bound": CHI2_95_1DOF, "n": len(keep), "q": q, "r": r, "r_sensor": r_sensor,
           "maneuver": maneuver, "dv": dv, "window": [maneuver, maneuver + window - 1]}
    for name, run in runs.items():
        values = np.array([run["nis"][k] for k in keep])
        out[name] = values
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_above"] = float(np.mean(values > CHI2_95_1DOF))
    near = (frames >= maneuver) & (frames < maneuver + window)
    out["q_small_inside_mean"] = float(out["q_small"][near].mean())
    out["q_small_outside_mean"] = float(out["q_small"][~near].mean())
    return out


# --- ids -> Result ----------------------------------------------------------------------

def _sim_fusion(o: Options) -> Result:
    d = fuse_two()
    w1, w2 = 1 / d["var1"], 1 / d["var2"]
    table = [{"source": "z1", "value": d["z1"], "var": d["var1"], "weight": w1 / (w1 + w2)},
             {"source": "z2", "value": d["z2"], "var": d["var2"], "weight": w2 / (w1 + w2)},
             {"source": "fused", "value": d["mean"], "var": d["var"], "weight": 1.0}]
    return _result("01-fusion", d, table, {"mean": d["mean"], "var": d["var"]}, o)


def _sim_one_d_loop(o: Options) -> Result:
    d = one_d_loop(Q=_pick(o.q, 5.0), R=_pick(o.r, 1.0))
    last = d["rows"][-1]
    return _result("03-1d-loop", d, d["rows"],
                   {"Q": d["Q"], "R": d["R"], "x_plus_frame_2": d["rows"][1]["x_plus"],
                    "P_plus_frame_2": d["rows"][1]["P_plus"], "x_last": last["x_plus"], "P_last": last["P_plus"]}, o)


def _sim_derivative(o: Options) -> Result:
    d = derivative_vs_filter(seed=_pick(o.seed, 0), r=_pick(o.r, 9.0), q=_pick(o.q, 0.01))
    table = [{"k": k, "z": z, "derivative": dv, "filter": fv}
             for k, z, dv, fv in zip(d["k"], d["z"], d["derivative"], d["filter"])]
    summary = {"sigma": d["sigma"], "std_position": d["std_position"], "theory_derivative": d["theory_derivative"],
               "std_derivative": d["std_derivative"], "std_filter": d["std_filter"],
               "ratio": d["std_derivative"] / d["std_filter"]}
    return _result("05-derivative-vs-filter", d, table, summary, o)


def _sim_covariance(o: Options) -> Result:
    d = covariance_cloud(seed=_pick(o.seed, 0))
    table = [{"stage": stage, "kind": kind, "var_p": m[0][0], "cov_pv": m[0][1], "var_v": m[1][1]}
             for stage in ("before", "after") for kind in ("formula", "empirical")
             for m in [d[f"P_{stage}_{kind}"]]]
    summary = {key: d[key] for key in ("P_before_formula", "P_before_empirical", "P_after_formula", "P_after_empirical")}
    return _result("08-covariance", d, table, summary, o)


def _sim_velocity_learning(o: Options) -> Result:
    d = velocity_learning(q=_pick(o.q, 0.01), r=_pick(o.r, 1.0), seed=_pick(o.seed, 0))
    last = d["rows"][-1]
    summary = {"q": d["q"], "r": d["r"], "std_v_10_30": d["std_v_10_30"], "v_30": last["v"],
               "sigma_v_1": d["rows"][0]["sigma_v"], "sigma_v_30": last["sigma_v"]}
    return _result("12-velocity-learning", d, d["rows"], summary, o)


def _sim_tradeoff(o: Options) -> Result:
    q_big, r = _pick(o.q, 1.0), _pick(o.r, 1.0)
    a = velocity_learning(q=0.01, r=r, seed=_pick(o.seed, 0))
    b = velocity_learning(q=q_big, r=r, seed=_pick(o.seed, 0))
    man = maneuver_response(seed=_pick(o.seed, 7), q_big=q_big, r=_pick(o.r, 4.0), gap=o.gap)
    data = {"small": a, "big": b, "maneuver": man}
    table = [{"k": ra["k"], "v_small": ra["v"], "K_v_small": ra["K_v"], "v_big": rb["v"], "K_v_big": rb["K_v"]}
             for ra, rb in zip(a["rows"], b["rows"])]
    summary = {"q_small": 0.01, "q_big": q_big, "std_v_10_30_small": a["std_v_10_30"], "std_v_10_30_big": b["std_v_10_30"],
               "K_v_30_small": a["rows"][-1]["K_v"], "K_v_30_big": b["rows"][-1]["K_v"],
               "maneuver_r": man["r"], "reach_small": man["small"]["reach"], "reach_big": man["big"]["reach"],
               "settle_small": man["small"]["settle"], "settle_big": man["big"]["settle"],
               "calm_std_small": man["small"]["calm_std"], "calm_std_big": man["big"]["calm_std"]}
    return _result("13-q-r-tradeoff", data, table, summary, o)


def _sim_nis(o: Options) -> Result:
    r = _pick(o.r, 4.0)
    d = nis(seed=_pick(o.seed, 1), q=_pick(o.q, 0.01), r=r, r_sensor=5 * r, gap=o.gap)
    table = [{"k": k, "nis_honest": a, "nis_overconfident": b, "nis_q_small": c}
             for k, a, b, c in zip(d["k"], d["honest"], d["overconfident"], d["q_small"])]
    keys = ("n", "r", "r_sensor", "maneuver", "bound", "honest_mean", "overconfident_mean", "q_small_mean",
            "q_small_inside_mean", "q_small_outside_mean", "honest_above", "overconfident_above", "q_small_above")
    return _result("14-nis", d, table, {key: d[key] for key in keys}, o)


SIMS: dict[str, Callable[[Options], Result]] = {
    "01-fusion": _sim_fusion,
    "03-1d-loop": _sim_one_d_loop,
    "05-derivative-vs-filter": _sim_derivative,
    "08-covariance": _sim_covariance,
    "12-velocity-learning": _sim_velocity_learning,
    "13-q-r-tradeoff": _sim_tradeoff,
    "14-nis": _sim_nis,
}
