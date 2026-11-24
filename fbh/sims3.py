"""The simulations of lesson 3, first half: an ICD message decoded, sigma points, and the UKF update in ukf.py.

Same contract as sims.py: every id in SIMS maps an Options to a Result (table, summary, plot), and the
functions above it are the same calculations with their raw numbers. These are single steps worked by hand, so
the CLI prints their table as well as the summary. Lengths are metres, angles radians.
"""
import struct
from typing import Callable

import numpy as np

from . import ekf, sims2, ukf
from .base import Options, Result, _pick, _result

OFFSET = sims2.OFFSET                       # the line of lessons 2 and 3 passes 400 m north of the radar


# --- section 2: reading an ICD ------------------------------------------------------------

# One LOS message of EOTS-X, an electro-optical sensor that does not exist: the ICD is made up for the lesson.
# Big-endian, an 8-byte header and a 24-byte payload; the CRC that follows on the wire is left out.
MESSAGE = bytes.fromhex("a55a 1001 0018 1267"                         # header
                        "0005 0849 1157 5240"                         # t_meas
                        "1a37 f5c5 080a e90e"                         # az, el
                        "007b ffeb 01f4 01 0f")                       # az_rate, el_rate, sigma_los, sensor, valid
ANGLE_LSB = np.pi / 2**31                   # rad per count of az and el: an int32 spans -pi to pi
RATE_LSB = 1e-4                             # rad/s per count of az_rate and el_rate
FIELDS = (
    # name, struct code (big-endian), unit
    ("sync", "H", ""), ("msg_id", "B", ""), ("version", "B", ""), ("length", "H", "B"), ("seq", "H", ""),
    ("t_meas", "Q", "µs"), ("az", "i", "rad"), ("el", "i", "rad"), ("az_rate", "h", "rad/s"),
    ("el_rate", "h", "rad/s"), ("sigma_los", "H", "µrad"), ("sensor", "B", ""), ("valid", "B", ""),
)
MSG_IDS = {0x10: "LOS"}
SENSORS = ("EO", "MWIR", "LWIR")
VALID_BITS = ("az_el_valid", "stabilised", "tracker_locked", "time_synced", "gimbal_at_limit")   # bits 0 to 4
TARGET = (300.0, 400.0, 100.0)              # east, north, up (m): the lesson's line, 100 m up


def decode(msg: bytes = MESSAGE) -> list[dict]:
    """The fields of one EOTS-X LOS message: name, bytes (0-based, inclusive), raw integer, value and unit.

    EOTS-X is a fictional ICD. Header: sync uint16, msg_id uint8, version uint8, length uint16 (of the payload),
    seq uint16. Payload: t_meas uint64 in µs; az and el int32 with an LSB of pi / 2^31 rad; az_rate and el_rate
    int16 with an LSB of 1e-4 rad/s; sigma_los uint16 in µrad; sensor uint8 (0 EO, 1 MWIR, 2 LWIR); valid uint8,
    bits 0 az/el valid, 1 stabilised, 2 tracker locked, 3 time synced to PPS, 4 gimbal at limit.
    """
    rows, at = [], 0
    for name, code, unit in FIELDS:
        size = struct.calcsize(">" + code)
        (raw,) = struct.unpack_from(">" + code, msg, at)
        rows.append({"field": name, "bytes": str(at) if size == 1 else f"{at}-{at + size - 1}", "raw": raw,
                     "value": _value(name, raw), "unit": unit})
        at += size
    return rows


def _value(name: str, raw: int):
    """What a raw field means: an angle or a rate in SI units, a code by its name, the rest as it is."""
    if name in ("az", "el"):
        return raw * ANGLE_LSB
    if name in ("az_rate", "el_rate"):
        return raw * RATE_LSB
    if name == "msg_id":
        return f"0x{raw:02X} {MSG_IDS.get(raw, '?')}"
    if name == "sensor":
        return SENSORS[raw] if raw < len(SENSORS) else str(raw)
    if name == "sync":
        return f"0x{raw:04X}"
    if name == "valid":
        return f"0x{raw:02X}"                     # the bits are spelled out by icd_check
    return raw


def icd_check(rows: list[dict]) -> dict:
    """The decoded angles against the geometry: a target 300 m east, 400 m north and 100 m up.

    Azimuth from north, atan(300 / 400); elevation over the 500 m ground range, atan(100 / 500). The differences
    are in LSB: an angle sent as an int32 is off by at most half a count.
    """
    v = {r["field"]: r for r in rows}
    east, north, up = TARGET
    az, el = np.arctan2(east, north), np.arctan2(up, np.hypot(east, north))
    return {"az_deg": np.degrees(v["az"]["value"]), "atan_300_400_deg": np.degrees(az),
            "az_off_lsb": (v["az"]["value"] - az) / ANGLE_LSB,
            "el_deg": np.degrees(v["el"]["value"]), "atan_100_500_deg": np.degrees(el),
            "el_off_lsb": (v["el"]["value"] - el) / ANGLE_LSB,
            **{bit: v["valid"]["raw"] >> i & 1 for i, bit in enumerate(VALID_BITS)}}


# --- section 3: a belief through y = x^2 ---------------------------------------------------

def square(mu: float = 2.0, var: float = 1.0) -> dict:
    """x ~ N(mu, var) through y = x^2, three ways.

    The tangent at mu (the EKF): mean mu^2, variance (2 mu)^2 var. The three sigma points: mean and variance of the
    points squared. Exact: E[x^2] = mu^2 + var, Var[x^2] = 4 mu^2 var + 2 var^2.
    """
    X, w = ukf.sigma_points(np.array([[mu]]), np.array([[var]]))
    Y = X**2
    mean, cov = ukf.unscented_transform(Y, w)
    grid = np.linspace(mu - 8 * np.sqrt(var), mu + 8 * np.sqrt(var), 200001)   # for the figure: the true spread
    weights = np.exp(-0.5 * (grid - mu) ** 2 / var)
    return {"mu": mu, "var": var, "x": X.ravel(), "w": w, "y": Y.ravel(), "grid": grid,
            "weights": weights / weights.sum(),
            "ekf_mean": mu**2, "ekf_var": (2 * mu) ** 2 * var,
            "ukf_mean": mean.item(), "ukf_var": cov.item(),
            "exact_mean": mu**2 + var, "exact_var": 4 * mu**2 * var + 2 * var**2}


# --- section 5: sigma points ------------------------------------------------------------------

P_PLUS = [[39.2, -14.4], [-14.4, 30.8]]     # lesson 2, section 5: the covariance after the radar update
X_PLUS = [300.8, 409.4]                     # and its estimate


def axes_sqrt(P):
    """A square root of P along its eigen-axes, V diag(sqrt(lambda)), smallest variance first.

    Each axis is turned to point north (a second component of 0 or more), so the lesson's P_plus gives the line of
    sight (0.6, 0.8) and the direction across it (-0.8, 0.6).
    """
    lam, V = np.linalg.eigh(P)
    V = V * np.where(V[-1] < 0, -1.0, 1.0)
    return V * np.sqrt(lam)


def _moments(X, w):
    mean, cov = ukf.unscented_transform(X, w)
    return mean.ravel(), cov


def sigma_sets() -> dict:
    """The sigma points of section 5: N(300, 10^2) in one dimension; in two, 100 I around (300, 400), and the
    tilted P_plus of lesson 2 around (300.8, 409.4) with two square roots, Cholesky and the eigen-axes.

    Each set comes with the mean and covariance it gives back; the one-dimensional set also with its fourth central
    moment, which n + kappa = 3 makes equal to a Gaussian's, 3 sigma^4.
    """
    x1, P1 = np.array([[300.0]]), np.array([[100.0]])
    X1, w1 = ukf.sigma_points(x1, P1)
    x2, P2 = np.array([[300.0], [400.0]]), 100.0 * np.eye(2)
    X2, w2 = ukf.sigma_points(x2, P2)
    xp, Pp = np.array(X_PLUS).reshape(2, 1), np.array(P_PLUS)
    Xc, wc = ukf.sigma_points(xp, Pp)
    Xa, wa = ukf.sigma_points(xp, Pp, sqrt=axes_sqrt)
    lam = np.linalg.eigvalsh(Pp)                              # 20 along the line of sight, 50 across it
    m1, c1 = _moments(X1, w1)
    sets = {"1d": (X1, w1), "round": (X2, w2), "cholesky": (Xc, wc), "axes": (Xa, wa)}
    return {"sets": sets, "x_plus": xp.ravel(), "P_plus": Pp,
            "mean_1d": m1.item(), "var_1d": c1.item(),
            "m4_1d": float(w1 @ (X1[0] - m1.item()) ** 4), "three_sigma4": 3 * P1.item() ** 2,
            "L_cholesky": np.linalg.cholesky(Pp), "axis_var": lam, "axis_dir": (axes_sqrt(Pp) / np.sqrt(lam)).T,
            "axis_offset": np.sqrt(3 * lam),
            **{f"{key}_{name}": value for name in ("round", "cholesky", "axes")
               for key, value in zip(("mean", "P"), _moments(*sets[name]))}}


# --- sections 7 and 8: one update with the range alone ----------------------------------

def range_update(x: float = 300.0, P: float = 100.0, z: float = 506.0, sigma_r: float = 2.0,
                 offset: float = OFFSET) -> dict:
    """The scalar UKF update by hand, next to the EKF and Bayes' rule of lesson 2 (sims2.range_only_update).

    `points` are the three sigma points of N(x, P), `ranges` the same points through r = sqrt(x^2 + offset^2);
    `ekf` is lesson 2's update and exact answer for the same numbers. The `*_off` values are how far each
    estimate is from the exact mean, and `*_off_sigmas` that distance over the sigma the filter claims.
    """
    R = sigma_r**2
    u = ukf.update(np.array([[x]]), np.array([[P]]), np.array([[z]]),
                   lambda s: ekf.h_range(s, offset), np.array([[R]]))
    e = sims2.range_only_update(x, P, z, sigma_r, offset)
    x_plus, P_plus = u.x.item(), u.P.item()
    ekf_sigma, sigma = np.sqrt(e["P_plus"]), np.sqrt(P_plus)
    return {"x_minus": x, "P_minus": P, "z": z, "R": R, "offset": offset,
            "points": u.X.ravel(), "weights": u.w, "ranges": u.Z.ravel(),
            "z_hat": u.z_hat.item(), "spread": u.spread.item(), "S": u.S.item(), "P_xz": u.P_xz.item(),
            "K": u.K.item(), "y": u.y.item(), "x_plus": x_plus, "P_plus": P_plus, "sigma": sigma,
            "ekf": e, "ekf_sigma": ekf_sigma, "exact_sigma": np.sqrt(e["exact_var"]),
            "ekf_off": abs(e["x_plus"] - e["exact_mean"]), "ukf_off": abs(x_plus - e["exact_mean"]),
            "ekf_off_sigmas": _in_sigmas(abs(e["x_plus"] - e["exact_mean"]), ekf_sigma),
            "ukf_off_sigmas": _in_sigmas(abs(x_plus - e["exact_mean"]), sigma)}


def _in_sigmas(off: float, sigma: float):
    """off / sigma; None when the filter claims sigma 0 (a radar of 1e-9 m rounds the EKF's (1 - K H) P to 0)."""
    return off / sigma if sigma > 0 else None


WIDE = {"P": 10_000.0, "z": 560.0}          # section 8: a belief of 300 +- 100 m, and a reading of 560 m


# --- ids -> Result ----------------------------------------------------------------------

def _sim_icd(o: Options) -> Result:
    rows = decode()
    summary = icd_check(rows)
    return _result("ukf-02-icd-decode", {"rows": rows, "message": MESSAGE, **summary}, rows, summary, o,
                   digits={"value": 4}, show_table=True)


def _sim_square(o: Options) -> Result:
    d = square()
    table = [{"i": i, "weight": w, "x": x, "x_squared": y} for i, (w, x, y) in enumerate(zip(d["w"], d["x"], d["y"]))]
    keys = ("ekf_mean", "ekf_var", "ukf_mean", "ukf_var", "exact_mean", "exact_var")
    return _result("ukf-03-square", d, table, {"mu": d["mu"], "var": d["var"], **{k: d[k] for k in keys}}, o,
                   show_table=True)


def _sim_sigma_points(o: Options) -> Result:
    d = sigma_sets()
    table = [{"set": name, "i": i, "weight": w, "x": X[0, i], "y": X[1, i] if X.shape[0] > 1 else None}
             for name, (X, ws) in d["sets"].items() for i, w in enumerate(ws)]
    keys = ("mean_1d", "var_1d", "m4_1d", "three_sigma4", "mean_round", "P_round", "L_cholesky", "mean_cholesky",
            "P_cholesky", "axis_var", "axis_dir", "axis_offset", "mean_axes", "P_axes")
    return _result("ukf-05-sigma-points", d, table, {k: d[k] for k in keys}, o, show_table=True)


def _points_table(d: dict) -> list[dict]:
    return [{"i": i, "weight": w, "x": x, "range": r}
            for i, (w, x, r) in enumerate(zip(d["weights"], d["points"], d["ranges"]))]


def _sim_radar_update(o: Options) -> Result:
    d = range_update(sigma_r=_pick(o.sigma_r, 2.0))
    e = d["ekf"]
    summary = {"z": d["z"], "R": d["R"], **{k: d[k] for k in ("z_hat", "spread", "S", "P_xz", "K", "y", "x_plus",
                                                                "P_plus")},
               **{f"ekf_{k}": e[k] for k in ("h", "H", "S", "K", "x_plus", "P_plus")},
               "exact_mean": e["exact_mean"], "exact_var": e["exact_var"]}
    # Four decimals for the UKF's x_plus: 308.9049 is 308.90, which three decimals (308.905) would hide.
    return _result("ukf-07-radar-update", d, _points_table(d), summary, o, digits={"x_plus": 4}, show_table=True)


def _sim_wide(o: Options) -> Result:
    d = range_update(P=WIDE["P"], z=WIDE["z"], sigma_r=_pick(o.sigma_r, 2.0))
    e = d["ekf"]
    summary = {"z": d["z"], "P_minus": d["P_minus"],
               **{f"ekf_{k}": e[k] for k in ("S", "K", "x_plus")}, "ekf_sigma": d["ekf_sigma"],
               **{k: d[k] for k in ("z_hat", "S", "P_xz", "K", "x_plus", "sigma")},
               "exact_mean": e["exact_mean"], "exact_var": e["exact_var"], "exact_sigma": d["exact_sigma"],
               **{k: d[k] for k in ("ekf_off", "ekf_off_sigmas", "ukf_off", "ukf_off_sigmas")}}
    return _result("ukf-08-wide-belief", d, _points_table(d), summary, o, show_table=True)


SIMS: dict[str, Callable[[Options], Result]] = {
    "ukf-02-icd-decode": _sim_icd,
    "ukf-03-square": _sim_square,
    "ukf-05-sigma-points": _sim_sigma_points,
    "ukf-07-radar-update": _sim_radar_update,
    "ukf-08-wide-belief": _sim_wide,
}
