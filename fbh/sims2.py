"""The simulations of lesson 2: the extended Kalman filter in ekf.py on a radar that measures range and bearing.

Same contract as sims.py: every id in SIMS maps an Options to a Result (table, summary, plot), and the
functions above it are the same runs with their raw numbers. Lengths are metres, angles radians, and one
scan of the radar takes one second.
"""
from typing import Callable

import numpy as np

from . import ekf, scenarios
from .base import Options, Result, _pick, _result

OFFSET = 400.0                              # sections 1 to 3: the flight line passes 400 m north of the radar
SIGMA_R, SIGMA_THETA = 5.0, 0.02            # the radar of the lesson: 5 m in range, 0.02 rad (1.1 degrees) in bearing
Q = 0.1                                     # process noise intensity of the tracker, m^2/s^3
CHI2_95_2DOF = 5.99                         # 95 % bound of a chi-square with two degrees of freedom
HAND_X = (300.0, 400.0)                     # the prediction of the hand calculation: range 500 m, a 3-4-5 triangle


# --- section 2: a Gaussian through the range curve -------------------------------------

def tangent(x_hat: float = 300.0, sigma: float = 10.0, offset: float = OFFSET, n: int = 200001) -> dict:
    """Push the belief N(x_hat, sigma^2) about the position along the line through r = sqrt(px^2 + offset^2).

    `true_mean` and `true_std` are the mean and spread of the range that comes out (by numerical integration);
    `lin_mean` and `lin_std` are what the tangent at x_hat says: h(x_hat) and |H| sigma.
    """
    grid = np.linspace(x_hat - 8 * sigma, x_hat + 8 * sigma, n)
    w = np.exp(-0.5 * ((grid - x_hat) / sigma) ** 2)
    w /= w.sum()
    r = np.hypot(grid, offset)
    true_mean = float(w @ r)
    h = float(np.hypot(x_hat, offset))
    H = x_hat / h
    return {"x_hat": x_hat, "sigma": sigma, "offset": offset, "h": h, "H": H,
            "lin_mean": h, "lin_std": abs(H) * sigma,
            "true_mean": true_mean, "true_std": float(np.sqrt(w @ (r - true_mean) ** 2)),
            "grid": grid, "weights": w, "range": r}


TANGENT_CASES = ((300.0, 10.0), (300.0, 100.0), (0.0, 10.0), (0.0, 100.0))


# --- section 3: one update with the range alone ----------------------------------------

def range_only_update(x: float = 300.0, P: float = 100.0, z: float = 506.0, sigma_r: float = 2.0,
                      offset: float = OFFSET) -> dict:
    """The scalar EKF update by hand, and next to it the exact answer of Bayes' rule on a fine grid."""
    R = sigma_r**2
    xv, Pv = np.array([[x]]), np.array([[P]])
    u = ekf.update(xv, Pv, np.array([[z]]), lambda s: ekf.h_range(s, offset), lambda s: ekf.H_range(s, offset),
                   np.array([[R]]))
    span = 8 * np.sqrt(P)
    grid = np.linspace(x - span, x + span, 400001)
    log_post = -0.5 * (grid - x) ** 2 / P - 0.5 * (z - np.hypot(grid, offset)) ** 2 / R
    post = np.exp(log_post - log_post.max())                  # shifted, so a tiny sigma_r cannot underflow every point
    post /= post.sum()
    mean = float(post @ grid)
    return {"x_minus": x, "P_minus": P, "z": z, "R": R, "offset": offset,
            "h": float(np.hypot(x, offset)), "H": float(ekf.H_range(xv, offset)[0, 0]),
            "y": float(u.y[0, 0]), "S": float(u.S[0, 0]), "K": float(u.K[0, 0]),
            "x_plus": float(u.x[0, 0]), "P_plus": float(u.P[0, 0]),
            "exact_mean": mean, "exact_var": float(post @ (grid - mean) ** 2),
            "grid": grid, "posterior": post}


# --- section 5: one update with range and bearing --------------------------------------

def radar_update(x=HAND_X, prior_sigma: float = 10.0, dr: float = 10.0, dtheta: float = 0.02,
                 sigma_r: float = SIGMA_R, sigma_theta: float = SIGMA_THETA) -> dict:
    """One EKF update of a position estimate, as in section 5: the prediction is `x` with covariance
    prior_sigma^2 I, and the radar reads `dr` metres and `dtheta` radians more than the prediction expects.

    `var_along` and `var_across` are the variances of the corrected estimate along the line of sight and
    across it; `range_plus` is the range of the corrected estimate, to compare with the range the radar read.
    """
    xv = np.array(x, dtype=float).reshape(2, 1)
    P = prior_sigma**2 * np.eye(2)
    R = ekf.radar_R(sigma_r, sigma_theta)
    z_hat = ekf.h_radar(xv)
    z = z_hat + np.array([[dr], [dtheta]])
    u = ekf.update(xv, P, z, ekf.h_radar, ekf.H_radar, R, ekf.radar_residual)
    along = xv / np.linalg.norm(xv)                           # unit vector of the line of sight
    across = np.array([[-along[1, 0]], [along[0, 0]]])
    truth = z[0, 0] * np.array([np.cos(z[1, 0]), np.sin(z[1, 0])])   # where a noise-free reading puts the target
    return {"x_minus": xv.ravel(), "P_minus": P, "prior_sigma": prior_sigma, "sigma_r": sigma_r,
            "sigma_theta": sigma_theta, "z_hat": z_hat.ravel(), "z": z.ravel(), "H": ekf.H_radar(xv),
            "y": u.y.ravel(), "S": u.S, "K": u.K, "x_plus": u.x.ravel(), "P_plus": u.P, "nis": u.nis,
            "var_along": (along.T @ u.P @ along).item(), "var_across": (across.T @ u.P @ across).item(),
            "range_plus": float(np.hypot(*u.x.ravel())), "reading_xy": truth,
            "bend": float(prior_sigma**2 / (2 * z_hat[0, 0]))}


# --- the tracker on a scenario ----------------------------------------------------------

def measurements(scn: dict, sigma_r: float, sigma_theta: float) -> np.ndarray:
    """The radar's reports for a scenario, one [range, bearing] row per scan; the bearing lies in [-pi, pi)."""
    p, e = np.array(scn["p"]), np.array(scn["noise"])
    return np.column_stack([np.hypot(p[:, 0], p[:, 1]) + sigma_r * e[:, 0],
                            ekf.wrap(np.arctan2(p[:, 1], p[:, 0]) + sigma_theta * e[:, 1])])


def run_radar(scn: dict, q: float = Q, sigma_r: float = SIGMA_R, sigma_theta: float = SIGMA_THETA,
              wrap: bool = True) -> dict:
    """Run the EKF over a scenario, scans k = 1..n (predict, then update). State [px, py, vx, vy].

    Per scan: xm/Pm = prediction, x/P = after the update, z = the radar's report, z_hat = the report the
    prediction expected, y, S, K, nis = what the update made of it. With wrap=False the bearing innovation
    is the plain difference, the mistake of section 8. Everything returned is plain lists and floats.
    """
    F, Qm = ekf.cv2d_model(1.0, q)
    R = ekf.radar_R(sigma_r, sigma_theta)
    residual = ekf.radar_residual if wrap else np.subtract
    x = np.array(scn["x0"], dtype=float).reshape(4, 1)
    P = np.array(scn["P0"], dtype=float)
    out = {key: [] for key in ("xm", "Pm", "x", "P", "z", "z_hat", "y", "S", "K", "nis")}
    for z in measurements(scn, sigma_r, sigma_theta):
        x, P = ekf.predict(x, P, lambda s: F @ s, lambda s: F, Qm)
        out["xm"].append(x.ravel().tolist())
        out["Pm"].append(P.tolist())
        out["z_hat"].append(ekf.h_radar(x).ravel().tolist())
        u = ekf.update(x, P, z.reshape(2, 1), ekf.h_radar, ekf.H_radar, R, residual)
        x, P = u.x, u.P
        for key, value in (("x", x.ravel()), ("P", P), ("z", z), ("y", u.y.ravel()), ("S", u.S), ("K", u.K)):
            out[key].append(value.tolist())
        out["nis"].append(float(u.nis))
    return out


def _errors(scn: dict, run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per scan: the distance between the estimate and the true position, and the sigma the filter claims for it."""
    e = np.array(run["x"])[:, :2] - np.array(scn["p"])
    P = np.array(run["P"])
    return np.hypot(e[:, 0], e[:, 1]), np.sqrt(P[:, 0, 0] + P[:, 1, 1])


# --- section 7: the fly-by ----------------------------------------------------------------

FLYBY_SEED = 4                              # a typical draw: 3 of the 60 scans above the chi-square bound


def _along_across(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per scan: the filter's sigma along the line of sight to its estimate, and across it."""
    along, across = [], []
    for x, P in zip(run["x"], run["P"]):
        u = np.array(x[:2]) / np.hypot(x[0], x[1])
        t = np.array([-u[1], u[0]])
        Pp = np.array(P)[:2, :2]
        along.append(np.sqrt(u @ Pp @ u))
        across.append(np.sqrt(t @ Pp @ t))
    return np.array(along), np.array(across)


def flyby(seed: int = FLYBY_SEED, q: float = Q, sigma_r: float = SIGMA_R, sigma_theta: float = SIGMA_THETA) -> dict:
    """The fly-by of scenarios.flyby through the tracker, against the raw reports turned into positions.

    `error` is the distance from the estimate to the true position and `claimed` the sigma the filter gives
    for it (the square root of the two position variances added); `raw_error` is the same distance for each
    report turned into a point on its own. The rms values are over scans 10 to n, after the start has settled.
    """
    scn = scenarios.flyby(seed=seed)
    run = run_radar(scn, q, sigma_r, sigma_theta)
    err, claimed = _errors(scn, run)
    z, p = np.array(run["z"]), np.array(scn["p"])
    raw = np.column_stack([z[:, 0] * np.cos(z[:, 1]), z[:, 0] * np.sin(z[:, 1])])   # each report as a point
    raw_err = np.hypot(*(raw - p).T)
    nis = np.array(run["nis"])
    settled = slice(9, None)                                                         # scans 10 to n
    sigma_along, sigma_across = _along_across(run)
    return {"scn": scn, "run": run, "seed": seed, "q": q, "sigma_r": sigma_r, "sigma_theta": sigma_theta,
            "error": err, "claimed": claimed, "raw": raw, "raw_error": raw_err,
            "sigma_along": sigma_along, "sigma_across": sigma_across,
            "rms_raw": float(np.sqrt(np.mean(raw_err[settled] ** 2))),
            "rms_filter": float(np.sqrt(np.mean(err[settled] ** 2))),
            "v_last": run["x"][-1][2:], "nis_mean": float(nis.mean()),
            "nis_above": float(np.mean(nis > CHI2_95_2DOF)), "bound": CHI2_95_2DOF}


# --- section 8: the bearing wraps -----------------------------------------------------------

def wrap_demo(seed: int = 0, q: float = Q, sigma_r: float = SIGMA_R, sigma_theta: float = SIGMA_THETA) -> dict:
    """The crossing of scenarios.crossing, tracked twice: with the bearing innovation wrapped, and without.

    `jump_scan` is the first scan where the plain difference of the two bearings is more than pi away from
    zero (None if the reading and the prediction never land on opposite sides of the cut); `raw` and
    `wrapped` are that scan's bearing innovation both ways, in radians.
    """
    scn = scenarios.crossing(seed=seed)
    good = run_radar(scn, q, sigma_r, sigma_theta, wrap=True)
    bad = run_radar(scn, q, sigma_r, sigma_theta, wrap=False)
    raw = np.array(good["z"])[:, 1] - np.array(good["z_hat"])[:, 1]
    jumps = [k + 1 for k, d in enumerate(raw) if abs(d) > np.pi]
    k = jumps[0] if jumps else None
    err_good, _ = _errors(scn, good)
    err_bad, _ = _errors(scn, bad)
    return {"scn": scn, "wrapped_run": good, "plain_run": bad, "jump_scan": k, "jump_scans": jumps,
            "raw": None if k is None else float(raw[k - 1]),
            "wrapped": None if k is None else float(ekf.wrap(raw[k - 1])),
            "z_theta": None if k is None else good["z"][k - 1][1],
            "z_hat_theta": None if k is None else good["z_hat"][k - 1][1],
            "error_wrapped": err_good, "error_plain": err_bad,
            "max_error_wrapped": float(err_good.max()), "max_error_plain": float(err_bad.max())}


# --- section 9: is the filter honest? -------------------------------------------------------

CASES = (
    # name, the filter's first guess, sigma of that guess in position (m), radar sigmas
    ("baseline", "prior", 30.0, SIGMA_R, SIGMA_THETA),
    ("vague_prior", "prior", 300.0, SIGMA_R, SIGMA_THETA),
    ("first_report", "report", 300.0, SIGMA_R, SIGMA_THETA),
    ("sharp_range", "report", 30.0, 0.5, 0.1),
)


def consistency(seed: int = 0, runs: int = 300, n: int = 60, q: float = Q, cases=CASES) -> dict:
    """Monte Carlo over `runs` targets per case, each obeying the filter's own motion model.

    In every case the targets start scattered around (-600, 400) by the case's position sigma (and by a
    tenth of it in each velocity), then move as x_k = F x_{k-1} + w_k with w_k ~ N(0, Q(q)).
    - init "prior": the filter starts from that centre with exactly that covariance, an honest prior.
    - init "report": the filter starts from the first radar report instead (ekf.radar_to_xy), with the
      velocity unknown (0 with sigma 30 m/s).
    Per case: `bend` = sigma_c^2 / (2 r), the bend of the range circle across the first update's
    cross-range sigma sigma_c (the prior's sigma, or r sigma_theta for a start from a report);
    `nis_early` = mean NIS of scans 2 to 5, `nis_late` of scans 6 to n (an honest filter gives 2);
    `ratio` = the rms position error over the sigma the filter claims for it, pooled over all scans
    (an honest filter gives 1), and `ratio_worst` the same ratio at the scan where it is largest.
    """
    F, Qm = ekf.cv2d_model(1.0, q)
    LQ = np.linalg.cholesky(Qm)
    centre = np.array([[-600.0], [400.0], [20.0], [0.0]])
    r0 = float(np.hypot(-580.0, 400.0))                       # the range at the first scan
    rng = np.random.default_rng(seed)
    out = {"bound": CHI2_95_2DOF, "runs": runs, "n": n, "q": q, "cases": {}}
    for name, init, sigma_p, sigma_r, sigma_theta in cases:
        P0 = np.diag([sigma_p**2, sigma_p**2, (sigma_p / 10) ** 2, (sigma_p / 10) ** 2])
        L0 = np.linalg.cholesky(P0)
        R = ekf.radar_R(sigma_r, sigma_theta)
        nis = np.full((runs, n), np.nan)
        err2 = np.zeros((runs, n))
        claimed2 = np.zeros((runs, n))
        for i in range(runs):
            xt = centre + L0 @ rng.standard_normal((4, 1))
            truth = []
            for _ in range(n):
                xt = F @ xt + LQ @ rng.standard_normal((4, 1))
                truth.append(xt)
            noise = rng.standard_normal((n, 2))
            z_of = lambda k: ekf.h_radar(truth[k]) + np.array([[sigma_r * noise[k, 0]], [sigma_theta * noise[k, 1]]])
            if init == "prior":
                x, P, first = centre.copy(), P0.copy(), 0
            else:
                p, Pp = ekf.radar_to_xy(z_of(0), R)
                x, P, first = np.vstack([p, np.zeros((2, 1))]), np.zeros((4, 4)), 1
                P[:2, :2], P[2:, 2:] = Pp, 30.0**2 * np.eye(2)
                e = x - truth[0]
                err2[i, 0], claimed2[i, 0] = e[0, 0] ** 2 + e[1, 0] ** 2, P[0, 0] + P[1, 1]
            for k in range(first, n):
                x, P = ekf.predict(x, P, lambda s: F @ s, lambda s: F, Qm)
                z = z_of(k)
                z[1, 0] = ekf.wrap(z[1, 0])
                u = ekf.update(x, P, z, ekf.h_radar, ekf.H_radar, R, ekf.radar_residual)
                x, P = u.x, u.P
                e = x - truth[k]
                nis[i, k], err2[i, k], claimed2[i, k] = u.nis, e[0, 0] ** 2 + e[1, 0] ** 2, P[0, 0] + P[1, 1]
        rms, claimed = np.sqrt(err2.mean(axis=0)), np.sqrt(claimed2.mean(axis=0))
        sigma_c = sigma_p if init == "prior" else r0 * sigma_theta
        seen = nis[~np.isnan(nis)]
        out["cases"][name] = {
            "init": init, "sigma_p": sigma_p, "sigma_r": sigma_r, "sigma_theta": sigma_theta,
            "sigma_c": sigma_c, "bend": sigma_c**2 / (2 * r0),
            "nis_early": float(np.nanmean(nis[:, 1:5])), "nis_late": float(np.nanmean(nis[:, 5:])),
            "nis_above": float(np.mean(seen > CHI2_95_2DOF)),
            "rms": rms, "claimed": claimed,
            "ratio": float(np.sqrt(err2.mean() / claimed2.mean())), "ratio_worst": float(np.max(rms / claimed)),
            "rms_first": float(rms[0]), "claimed_first": float(claimed[0]),
        }
    return out


# --- ids -> Result ----------------------------------------------------------------------

def _sim_tangent(o: Options) -> Result:
    rows = [tangent(x_hat, sigma) for x_hat, sigma in TANGENT_CASES]
    keys = ("x_hat", "sigma", "h", "H", "lin_std", "true_mean", "true_std")
    table = [{key: d[key] for key in keys} for d in rows]
    near, flat = rows[0], rows[3]
    summary = {"h": near["h"], "H": near["H"], "lin_std": near["lin_std"], "true_mean": near["true_mean"],
               "true_std": near["true_std"], "flat_lin_std": flat["lin_std"], "flat_true_mean": flat["true_mean"],
               "flat_true_std": flat["true_std"]}
    return _result("ekf-02-tangent", {"rows": rows}, table, summary, o)


def _sim_range_only(o: Options) -> Result:
    d = range_only_update(sigma_r=_pick(o.sigma_r, 2.0))
    keys = ("h", "H", "y", "S", "K", "x_plus", "P_plus", "exact_mean", "exact_var")
    table = [{"quantity": key, "value": d[key]} for key in ("x_minus", "P_minus", "z", "R") + keys]
    return _result("ekf-03-range-only", d, table, {key: d[key] for key in keys}, o)


def _sim_radar_update(o: Options) -> Result:
    d = radar_update(sigma_r=_pick(o.sigma_r, SIGMA_R), sigma_theta=_pick(o.sigma_theta, SIGMA_THETA))
    table = [{"quantity": name, "range_or_x": a, "bearing_or_y": b}
             for name, (a, b) in (("x_minus", d["x_minus"]), ("z_hat", d["z_hat"]), ("z", d["z"]), ("y", d["y"]),
                                  ("S_diagonal", np.diag(d["S"])), ("K_row_x", d["K"][0]), ("K_row_y", d["K"][1]),
                                  ("x_plus", d["x_plus"]))]
    summary = {"H": d["H"], "S": d["S"], "K": d["K"], "x_plus": d["x_plus"], "P_plus": d["P_plus"],
               "var_along": d["var_along"], "var_across": d["var_across"], "nis": d["nis"]}
    return _result("ekf-05-radar-update", d, table, summary, o)


def _sim_flyby(o: Options) -> Result:
    d = flyby(seed=_pick(o.seed, FLYBY_SEED), q=_pick(o.q, Q), sigma_r=_pick(o.sigma_r, SIGMA_R),
              sigma_theta=_pick(o.sigma_theta, SIGMA_THETA))
    run, scn = d["run"], d["scn"]
    table = [{"k": k, "range": z[0], "bearing": z[1], "px": x[0], "py": x[1], "vx": x[2], "vy": x[3],
              "error": e, "claimed": c, "raw_error": raw, "nis": nis}
             for k, (z, x, e, c, raw, nis) in enumerate(zip(run["z"], run["x"], d["error"], d["claimed"],
                                                            d["raw_error"], run["nis"]), start=1)]
    summary = {"q": d["q"], "sigma_r": d["sigma_r"], "sigma_theta": d["sigma_theta"],
               "rms_raw": d["rms_raw"], "rms_filter": d["rms_filter"], "error_45": d["error"][44],
               "claimed_45": d["claimed"][44], "v_last": d["v_last"], "nis_mean": d["nis_mean"],
               "nis_above": d["nis_above"]}
    return _result("ekf-07-flyby", d, table, summary, o)


def _sim_wrap(o: Options) -> Result:
    d = wrap_demo(seed=_pick(o.seed, 0), q=_pick(o.q, Q), sigma_r=_pick(o.sigma_r, SIGMA_R),
                  sigma_theta=_pick(o.sigma_theta, SIGMA_THETA))
    table = [{"k": k, "bearing": z[1], "expected": zh[1], "plain_difference": z[1] - zh[1],
              "error_wrapped": a, "error_plain": b}
             for k, (z, zh, a, b) in enumerate(zip(d["wrapped_run"]["z"], d["wrapped_run"]["z_hat"],
                                                   d["error_wrapped"], d["error_plain"]), start=1)]
    summary = {key: d[key] for key in ("jump_scan", "z_theta", "z_hat_theta", "raw", "wrapped",
                                       "max_error_wrapped", "max_error_plain")}
    # Four decimals for the bearings, so that the two printed ones subtract to the printed difference.
    return _result("ekf-08-wrap", d, table, summary, o,
                   digits={"z_theta": 4, "z_hat_theta": 4, "raw": 4, "wrapped": 4})


def _sim_consistency(o: Options) -> Result:
    d = consistency(seed=_pick(o.seed, 0))
    keys = ("sigma_c", "bend", "sigma_r", "nis_early", "nis_late", "nis_above", "rms_first", "claimed_first",
            "ratio", "ratio_worst")
    table = [{"case": name, **{key: c[key] for key in keys}} for name, c in d["cases"].items()]
    summary = {"runs": d["runs"], "bound": d["bound"]}
    for name, c in d["cases"].items():
        for key in ("bend", "nis_early", "nis_late", "ratio"):
            summary[f"{name}_{key}"] = c[key]
    return _result("ekf-09-consistency", d, table, summary, o)


SIMS: dict[str, Callable[[Options], Result]] = {
    "ekf-02-tangent": _sim_tangent,
    "ekf-03-range-only": _sim_range_only,
    "ekf-05-radar-update": _sim_radar_update,
    "ekf-07-flyby": _sim_flyby,
    "ekf-08-wrap": _sim_wrap,
    "ekf-09-consistency": _sim_consistency,
}
