"""The figures, one function per simulation id. Light paper style, 1600 x 900 px, EN or TR labels.

Figures are built straight on matplotlib's Figure class, so no display and no global pyplot state
is involved. Turkish figures use decimal commas in tick labels and in every number printed on them.
"""
from pathlib import Path

import matplotlib
import matplotlib.patches
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from .i18n import t

PAPER, INK, TERRACOTTA, BLUE = "#fbfaf7", "#2b2d33", "#c0612b", "#3d6fb0"
MUTED, GRID = "#8a8d96", "#e8e5de"
BLUE_LIGHT = "#8fb0d8"
GREEN = "#3f8a72"
SIZE_PX, DPI = (1600, 900), 100

RC = {
    "font.family": "DejaVu Sans", "font.size": 16,
    "figure.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlesize": 21,
    "axes.titleweight": "bold", "axes.labelsize": 17, "axes.titlelocation": "left", "axes.titlepad": 16,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 15, "legend.frameon": False, "lines.linewidth": 2.6,
}


def num(value: float, lang: str, digits: int | None = None) -> str:
    """A number as printed on a figure: shortest form (or `digits` decimals), real minus, comma in Turkish."""
    s = f"{value:.10g}" if digits is None else f"{value:.{digits}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    s = s.replace("-", "−")
    return s.replace(".", ",") if lang == "tr" else s


def tick_formatter(lang: str):
    return lambda value, _pos: num(value, lang)


def _style(ax, lang: str, y_ticks: bool = True, int_x: bool = False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(tick_formatter(lang)))
    ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter(lang)))
    if int_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not y_ticks:
        ax.set_yticks([])
        ax.grid(False, axis="y")


def _legend(ax, loc="best", **kw):
    return ax.legend(loc=loc, handlelength=1.8, labelspacing=0.5, **kw)


# --- 01-fusion --------------------------------------------------------------------------

def fusion(fig, d, lang):
    ax = fig.add_subplot()
    _style(ax, lang, y_ticks=False)
    g = d["grid"]
    var_word = t("variance", lang)
    for pdf, color, ls, name, var, mean in (
            (d["pdf1"], BLUE, "-", t("fusion_good", lang), d["var1"], d["z1"]),
            (d["pdf2"], BLUE_LIGHT, "--", t("fusion_guess", lang), d["var2"], d["z2"])):
        ax.plot(g, pdf, color=color, ls=ls, label=f"{name}: {num(mean, lang)} ({var_word} {num(var, lang)})")
        ax.fill_between(g, pdf, color=color, alpha=0.10)
    ax.plot(g, d["pdf_fused"], color=TERRACOTTA, lw=3.4,
            label=f"{t('fusion_fused', lang)}: {num(d['mean'], lang)} ({var_word} {num(d['var'], lang, 1)})")
    ax.fill_between(g, d["pdf_fused"], color=TERRACOTTA, alpha=0.16)
    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel(t("fusion_x", lang))
    ax.set_ylabel(t("fusion_density", lang))
    ax.set_title(t("fusion_title", lang))
    _legend(ax, "upper right")


# --- 03-1d-loop -------------------------------------------------------------------------

def one_d_loop(fig, d, lang):
    left, right = fig.subplots(1, 2)
    rows = d["rows"]
    ks = np.array([r["k"] for r in rows])
    xp = np.array([r["x_plus"] for r in rows])
    sd = np.sqrt([r["P_plus"] for r in rows])

    _style(left, lang, int_x=True)
    left.fill_between(ks, xp - 2 * sd, xp + 2 * sd, color=TERRACOTTA, alpha=0.15, lw=0, label=t("band_2s", lang))
    left.plot(ks, xp, color=TERRACOTTA)
    pred = [(r["k"], r["x_minus"]) for r in rows if r["x_minus"] is not None]
    left.plot(*zip(*pred), ls="none", marker="o", ms=11, mfc=PAPER, mec=TERRACOTTA, mew=2.2, label=t("prediction", lang))
    meas = [(r["k"], r["z"]) for r in rows if r["z"] is not None]
    left.plot(*zip(*meas), ls="none", marker="X", ms=17, color=BLUE, label=t("measurement", lang), zorder=3)
    upd = [(r["k"], r["x_plus"]) for r in rows if r["z"] is not None]
    left.plot(*zip(*upd), ls="none", marker="o", ms=9, color=TERRACOTTA, label=t("update", lang), zorder=4)
    left.plot(ks[0], xp[0], marker="o", ms=9, color=TERRACOTTA, zorder=4)         # the starting belief
    left.set_xlabel(t("frame", lang))
    left.set_ylabel(t("position_px", lang))
    left.set_title(t("loop_title_pos", lang))
    left.set_xlim(0.7, ks[-1] + 0.3)
    _legend(left, "upper left")

    _style(right, lang, int_x=True)
    pts = [(1, rows[0]["P_plus"])]
    for r in rows[1:]:
        pts.append((r["k"], r["P_minus"]))
        if r["z"] is not None:
            pts.append((r["k"], r["P_plus"]))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        update = x0 == x1                                        # a vertical drop is the update
        right.plot([x0, x1], [y0, y1], color=BLUE if update else TERRACOTTA, lw=3.2 if update else 2.6)
    for x, y in pts:
        right.plot(x, y, marker="o", ms=8, color=INK)
        right.annotate(num(y, lang), (x, y), textcoords="offset points", xytext=(11, 6), fontsize=15)
    right.axvspan(2.5, ks[-1] + 0.6, color=MUTED, alpha=0.10, lw=0)
    right.text((2.5 + ks[-1] + 0.6) / 2, 0.45, t("no_measurement", lang), ha="center", color=MUTED, fontsize=15)
    right.set_xlabel(t("frame", lang))
    right.set_ylabel(t("variance_px2", lang))
    right.set_title(t("loop_title_var", lang))
    right.set_xlim(0.7, ks[-1] + 0.6)
    right.set_ylim(0, None)


# --- 05-derivative-vs-filter ------------------------------------------------------------

def derivative_vs_filter(fig, d, lang):
    ax = fig.add_subplot()
    _style(ax, lang)
    k, sig = d["k"], d["std_position"]
    ax.axhline(d["truth_v"], color=INK, ls="--", lw=1.8, label=t("truth", lang))
    for sign in (1, -1):                                                 # the level of the position noise itself
        ax.axhline(sign * sig, color=MUTED, ls=":", lw=2.2,
                   label=f"±{t('deriv_position_noise', lang)} (σ = {num(sig, lang, 1)})" if sign == 1 else None)
    ax.plot(k, d["derivative"], color=BLUE, lw=1.2, marker="o", ms=3.5,
            label=f"{t('deriv_difference', lang)} (σ = {num(d['std_derivative'], lang, 1)})")
    ax.plot(k, d["filter"], color=TERRACOTTA, lw=3.4,
            label=f"{t('filter', lang)} (σ = {num(d['std_filter'], lang, 1)})")
    lim = 1.15 * float(np.max(np.abs(d["derivative"])))
    ax.set_ylim(-lim, lim)
    ax.set_xlim(k[0] - 1, k[-1] + 1)
    ax.set_xlabel(t("frame", lang))
    ax.set_ylabel(t("deriv_ylabel", lang))
    ax.set_title(t("deriv_title", lang))
    _legend(ax, "upper right", ncols=2, columnspacing=1.6)


# --- 08-covariance ----------------------------------------------------------------------

def _ellipse(P, n=200):
    """The 2σ ellipse of a 2 x 2 covariance around the origin. Through eigh rather than Cholesky, so that a P
    squeezed flat (a radar that is almost exact) still draws, as a line."""
    a = np.linspace(0, 2 * np.pi, n)
    w, V = np.linalg.eigh(np.array(P, dtype=float))
    return 2 * (V * np.sqrt(np.clip(w, 0, None))) @ np.vstack([np.cos(a), np.sin(a)])


def covariance(fig, d, lang):
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax, stage in zip(axes, ("before", "after")):
        _style(ax, lang)
        e = d[f"e_{stage}"][:3000]
        ax.scatter(e[:, 0], e[:, 1], s=14, color=MUTED, alpha=0.28, lw=0, label=t("cov_cloud", lang))
        ex, ey = _ellipse(d[f"P_{stage}_formula"])
        ax.plot(ex, ey, color=TERRACOTTA, lw=3.2, label=t("cov_ellipse", lang))
        ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
        cov = d[f"P_{stage}_formula"][0][1]
        ax.text(0.04, 0.05, f"{t('covariance', lang)}: {num(cov, lang, 2)}", transform=ax.transAxes, fontsize=17)
        ax.set_title(t(f"cov_{stage}", lang))
        ax.set_xlabel(t("pos_error_px", lang))
    axes[0].set_ylabel(t("vel_error", lang))
    _legend(axes[0], "upper left")
    axes[0].set_xlim(-8, 8)
    axes[0].set_ylim(-3.2, 3.2)


# --- 12-velocity-learning ---------------------------------------------------------------

def velocity_learning(fig, d, lang):
    ax = fig.add_subplot()
    _style(ax, lang, int_x=True)
    s, rows = d["start"], d["rows"]
    k = np.array([0] + [r["k"] for r in rows])
    v = np.array([s["v"]] + [r["v"] for r in rows])
    sd = np.array([s["sigma_v"]] + [r["sigma_v"] for r in rows])
    ax.axhline(d["truth_v"], color=INK, ls="--", lw=1.8, label=f"{t('learn_truth', lang)}: {num(d['truth_v'], lang)}")
    ax.fill_between(k, v - 2 * sd, v + 2 * sd, color=TERRACOTTA, alpha=0.16, lw=0, label=t("band_2s", lang))
    ax.plot(k, v, color=TERRACOTTA, marker="o", ms=6, label=t("learn_estimate", lang))
    ax.plot(0, s["v"], marker="o", ms=12, mfc=PAPER, mec=TERRACOTTA, mew=2.4, ls="none",
            label=f"{t('learn_start', lang)}: {num(s['v'], lang)}")
    ax.annotate(f"σ = {num(sd[-1], lang, 2)}", (k[-1], v[-1]), textcoords="offset points", xytext=(-8, -34),
                ha="right", fontsize=16, color=TERRACOTTA)
    ax.set_xlim(-0.5, k[-1] + 0.5)
    ax.set_xlabel(t("frame", lang))
    ax.set_ylabel(t("velocity", lang))
    ax.set_title(t("learn_title", lang))
    _legend(ax, "lower right")


# --- 13-q-r-tradeoff --------------------------------------------------------------------

def tradeoff(fig, d, lang):
    left, right = fig.subplots(1, 2)
    small, big, man = d["small"], d["big"], d["maneuver"]
    names = (f"q = {num(small['q'], lang)}: {t('trade_calm', lang)}", f"q = {num(big['q'], lang)}: {t('trade_agile', lang)}")

    _style(left, lang, int_x=True)
    left.axhline(small["truth_v"], color=INK, ls="--", lw=1.8, label=t("truth", lang))
    for run, color, name in ((big, BLUE, names[1]), (small, TERRACOTTA, names[0])):
        k = [0] + [r["k"] for r in run["rows"]]
        v = [run["start"]["v"]] + [r["v"] for r in run["rows"]]
        left.plot(k, v, color=color, lw=2.6, label=name)
    left.set_xlabel(t("frame", lang))
    left.set_ylabel(t("velocity", lang))
    left.set_title(t("trade_learn", lang))
    _legend(left, "lower right")

    _style(right, lang, int_x=True)
    scn = man["scn"]
    k = np.arange(1, len(scn["v"]) + 1)
    right.step(k, scn["v"], where="post", color=INK, ls="--", lw=1.8, label=t("truth", lang))
    for run, color in ((man["big"], BLUE), (man["small"], TERRACOTTA)):
        right.plot(k, run["v"], color=color, lw=2.6)
    right.axvline(scn["maneuver"], color=MUTED, lw=1.2, ls=":")
    right.text(scn["maneuver"] - 1, right.get_ylim()[0] + 0.4, t("maneuver", lang), ha="right", color=MUTED, fontsize=15)
    right.set_xlabel(t("frame", lang))
    right.set_title(t("trade_maneuver", lang))
    right.sharey(left)


# --- 14-nis -----------------------------------------------------------------------------

def _rolling(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def nis(fig, d, lang):
    axes = fig.subplots(3, 1, sharex=True, sharey=True)
    k, window = np.array(d["k"]), 20
    mean = t("mean", lang)
    m = lambda key: num(d[key], lang, 1)
    panels = (
        ("honest", BLUE, f"{t('nis_honest', lang)}: {mean} {m('honest_mean')}"),
        ("overconfident", TERRACOTTA, f"{t('nis_over', lang)}: {mean} {m('overconfident_mean')}"
         f"  ({t('nis_over_detail', lang).format(r=num(d['r'], lang), r_sensor=num(d['r_sensor'], lang))})"),
        ("q_small", GREEN, f"{t('nis_qsmall', lang)}: " + t("nis_inside", lang).format(
            inside=m("q_small_inside_mean"), outside=m("q_small_outside_mean"))),
    )
    for ax, (key, color, title) in zip(axes, panels):
        _style(ax, lang, int_x=True)
        ax.plot(k, d[key], ls="none", marker="o", ms=4.5, color=color, alpha=0.25, mew=0)
        ax.plot(k[window - 1:], _rolling(d[key], window), color=color, lw=3.2)
        ax.axhline(1, color=INK, ls="--", lw=1.6)
        ax.axhline(d["bound"], color=MUTED, ls=":", lw=2)
        ax.set_ylim(0, 16)
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlim(k[0], k[-1])
        ax.set_title(title, fontsize=17, pad=8)
    lo, hi = d["window"]
    axes[2].axvspan(lo, hi, color=GREEN, alpha=0.14, lw=0)
    axes[2].text(hi + 4, 13.2, t("maneuver", lang), color=MUTED, fontsize=15)
    axes[2].set_xlabel(t("frame", lang))
    axes[1].set_ylabel(t("nis_ylabel", lang))
    handles = [Line2D([], [], ls="none", marker="o", ms=5, color=MUTED, alpha=0.5, label=t("nis_single", lang)),
               Line2D([], [], color=MUTED, lw=3.2, label=t("nis_window", lang)),
               Line2D([], [], color=INK, ls="--", lw=1.6, label=t("nis_expected", lang)),
               Line2D([], [], color=MUTED, ls=":", lw=2, label=f"{t('nis_bound', lang)}: {num(d['bound'], lang, 2)}")]
    axes[0].legend(handles=handles, loc="upper right", ncols=2, fontsize=14, columnspacing=1.4)
    fig.suptitle(t("nis_title", lang), x=0.06, ha="left", fontsize=21, fontweight="bold")


# --- lesson 2: maps in metres, the radar at the origin -----------------------------------

def _short(value: float, lang: str) -> str:
    """A computed value on a figure: one decimal at most, none when it is zero (309, 308.9; Turkish 308,9)."""
    return num(round(float(value), 1), lang)


def _vec(values, lang: str) -> str:
    """A point as printed on a figure, "(300.8, 409.4)"; Turkish "(300,8; 409,4)", so the commas cannot be misread."""
    return "(" + ("; " if lang == "tr" else ", ").join(_short(v, lang) for v in values) + ")"


def _normal(grid, mean, var):
    return np.exp(-0.5 * (grid - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


def _map(ax, lang: str):
    _style(ax, lang)
    ax.set_aspect("equal", adjustable="datalim")                        # one metre east is one metre north
    ax.set_xlabel(t("east_m", lang))
    ax.set_ylabel(t("north_m", lang))


def _frame(ax, lo, hi):
    """Show at least the box from lo to hi (east, north) on a map; the panel's shape widens one side.

    Call it after drawing: the box replaces the data limits, so lines that run off the map do not stretch it.
    """
    ax.ignore_existing_data_limits = True
    ax.update_datalim([lo, hi])
    ax.margins(0)
    ax.autoscale_view()


# --- ekf-02-tangent ---------------------------------------------------------------------

def tangent(fig, d, lang):
    left, right = fig.subplots(1, 2)
    near, wide, flat = d["rows"][0], d["rows"][1], d["rows"][2]         # N(300, 10²), N(300, 100²), N(0, 10²)
    offset = near["offset"]
    px = np.linspace(-600, 600, 601)

    _style(left, lang)
    left.axvspan(wide["x_hat"] - 2 * wide["sigma"], wide["x_hat"] + 2 * wide["sigma"], color=TERRACOTTA, alpha=0.10,
                 lw=0, label=t("tan_belief", lang).format(mean=num(wide["x_hat"], lang), sigma=num(wide["sigma"], lang)))
    left.plot(px, np.hypot(px, offset), color=INK, lw=3, label=t("tan_curve", lang).format(offset=num(offset, lang)))
    for row, color, ls in ((near, TERRACOTTA, "-"), (flat, MUTED, "--")):
        left.plot(px, row["h"] + row["H"] * (px - row["x_hat"]), color=color, ls=ls, lw=2.4,
                  label=t("tan_at", lang).format(x=num(row["x_hat"], lang), slope=num(row["H"], lang)))
        left.plot(row["x_hat"], row["h"], marker="o", ms=10, color=color, zorder=4)
    left.set_xlim(px[0], px[-1])
    left.set_ylim(offset - 60, 1.06 * np.hypot(px[-1], offset))
    left.set_xlabel(t("tan_x", lang))
    left.set_ylabel(t("range_m", lang))
    left.set_title(t("tan_title_curve", lang))
    _legend(left, "upper center")

    _style(right, lang, y_ticks=False)
    edges = np.arange(offset, wide["range"].max() + 2.0, 2.0)          # no range comes out below the offset
    density, _ = np.histogram(wide["range"], bins=edges, weights=wide["weights"], density=True)
    mid = (edges[:-1] + edges[1:]) / 2
    g = np.linspace(wide["lin_mean"] - 4.5 * wide["lin_std"], wide["lin_mean"] + 4.5 * wide["lin_std"], 601)
    bell = lambda r: _normal(r, wide["lin_mean"], wide["lin_std"] ** 2)
    right.fill_between(mid, density, color=MUTED, alpha=0.45, lw=0,
                       label=t("tan_true", lang).format(mean=_short(wide["true_mean"], lang),
                                                        std=_short(wide["true_std"], lang)))
    right.plot(g, bell(g), color=TERRACOTTA, lw=3.2,
               label=t("tan_lin", lang).format(mean=_short(wide["lin_mean"], lang), std=_short(wide["lin_std"], lang)))
    right.vlines(wide["true_mean"], 0, np.interp(wide["true_mean"], mid, density), color=INK, lw=2)  # the two means
    right.vlines(wide["lin_mean"], 0, bell(wide["lin_mean"]), color=TERRACOTTA, lw=2, ls="--")
    right.set_xlim(g[0], g[-1])
    right.set_ylim(0, 1.3 * max(density.max(), bell(wide["lin_mean"])))
    right.set_xlabel(t("range_m", lang))
    right.set_ylabel(t("density", lang))
    right.set_title(t("tan_title_belief", lang).format(mean=num(wide["x_hat"], lang), sigma=num(wide["sigma"], lang)))
    _legend(right, "upper right")


# --- ekf-03-range-only ------------------------------------------------------------------

def range_only(fig, d, lang):
    ax = fig.add_subplot()
    _style(ax, lang, y_ticks=False)
    x0, P0 = d["x_minus"], d["P_minus"]
    lo, hi = x0 - 2 * np.sqrt(P0), x0 + 3 * np.sqrt(P0)
    g = np.linspace(lo, hi, 1001)
    var_word = t("variance", lang)
    ax.plot(g, _normal(g, x0, P0), color=TERRACOTTA, ls="--", lw=2.2,
            label=f"{t('prediction', lang)}: {num(x0, lang)} ({var_word} {num(P0, lang)})")
    keep = (d["grid"] >= lo) & (d["grid"] <= hi)
    step = max(1, int(keep.sum()) // 2000)                              # the grid is far finer than the pixels
    grid = d["grid"][keep][::step]
    density = d["posterior"][keep][::step] / (d["grid"][1] - d["grid"][0])   # grid weights to a density
    ax.fill_between(grid, density, color=MUTED, alpha=0.45, lw=0,
                    label=f"{t('exact', lang)}: {_short(d['exact_mean'], lang)} ({var_word} {_short(d['exact_var'], lang)})")
    ax.plot(g, _normal(g, d["x_plus"], d["P_plus"]), color=TERRACOTTA, lw=3.4,
            label=f"EKF: {_short(d['x_plus'], lang)} ({var_word} {_short(d['P_plus'], lang)})")
    alone = float(np.sqrt(d["z"] ** 2 - d["offset"] ** 2))             # where the reading puts p_x on its own
    ax.axvline(alone, color=BLUE, ls=":", lw=2.6, label=t("ro_reading", lang).format(
        z=num(d["z"], lang), offset=num(d["offset"], lang), x=_short(alone, lang)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, None)
    ax.set_xlabel(t("ro_x", lang))
    ax.set_ylabel(t("density", lang))
    ax.set_title(t("ro_title", lang).format(z=num(d["z"], lang), sigma_r=num(np.sqrt(d["R"]), lang)))
    _legend(ax, "upper left")


# --- ekf-05-radar-update ----------------------------------------------------------------

def radar_update(fig, d, lang):
    ax, side = fig.subplots(1, 2, width_ratios=[1.45, 1])
    _map(ax, lang)
    xm, xp, z, report = (np.asarray(d[key]) for key in ("x_minus", "x_plus", "z", "reading_xy"))
    two_sigma, ring = 2 * d["prior_sigma"], np.linspace(0, 2 * np.pi, 200)
    ellipse = xp[:, None] + _ellipse(d["P_plus"])
    r_hat = float(np.hypot(*xm))
    along = np.array([-1, 1]) * 4 * two_sigma                           # long enough to cross the whole map
    ax.plot(xm[0] / r_hat * (r_hat + along), xm[1] / r_hat * (r_hat + along), color=INK, ls=":", lw=2)
    arc = z[1] + np.linspace(along[0], along[1], 200) / z[0]
    ax.plot(z[0] * np.cos(arc), z[0] * np.sin(arc), color=BLUE, lw=1.8)
    ax.plot(np.cos(z[1]) * (z[0] + along), np.sin(z[1]) * (z[0] + along), color=BLUE, lw=1.8)
    ax.plot(*report, marker="X", ms=17, color=BLUE, zorder=5)
    ax.plot(xm[0] + two_sigma * np.cos(ring), xm[1] + two_sigma * np.sin(ring), color=TERRACOTTA, ls="--", lw=2.2)
    ax.plot(*xm, marker="o", ms=12, mfc=PAPER, mec=TERRACOTTA, mew=2.4, zorder=5)
    ax.plot(*ellipse, color=TERRACOTTA, lw=3.2)
    ax.plot(*xp, marker="o", ms=10, color=TERRACOTTA, zorder=5)
    ax.annotate("", xy=xp, xytext=xm, arrowprops={"arrowstyle": "-|>", "color": TERRACOTTA, "lw": 2,
                                                  "shrinkA": 8, "shrinkB": 6, "mutation_scale": 18})
    pts = np.column_stack([xm - two_sigma, xm + two_sigma, report, ellipse])
    pad = 0.2 * np.ptp(pts, axis=1).max()
    _frame(ax, pts.min(axis=1) - pad, pts.max(axis=1) + pad)
    ax.set_title(t("ru_title", lang))

    side.axis("off")
    handles = [
        Line2D([], [], color=TERRACOTTA, ls="--", lw=2.2, marker="o", ms=11, mfc=PAPER, mec=TERRACOTTA, mew=2.4,
               label=t("ru_prediction", lang).format(x=_vec(xm, lang))),
        Line2D([], [], ls="none", marker="X", ms=15, color=BLUE,
               label=t("ru_report", lang).format(r=_short(z[0], lang), theta=num(z[1], lang, 4))),
        Line2D([], [], color=BLUE, lw=1.8, label=t("ru_circle_line", lang)),
        Line2D([], [], color=TERRACOTTA, lw=3.2, marker="o", ms=10, label=t("ru_corrected", lang).format(x=_vec(xp, lang))),
        Line2D([], [], color=INK, ls=":", lw=2, label=t("line_of_sight", lang)),
    ]
    side.legend(handles=handles, loc="upper left", handlelength=2.4, labelspacing=1.0, borderaxespad=0)
    side.text(0, 0.12, t("ru_variances", lang).format(along=_short(d["var_along"], lang),
                                                      across=_short(d["var_across"], lang),
                                                      prior=num(d["prior_sigma"] ** 2, lang)),
              transform=side.transAxes, va="bottom", fontsize=16, linespacing=1.6)


# --- ekf-07-flyby -----------------------------------------------------------------------

def flyby(fig, d, lang):
    left, right = fig.subplots(1, 2, width_ratios=[1.2, 1])
    p, x, raw = np.array(d["scn"]["p"]), np.array(d["run"]["x"]), np.asarray(d["raw"])

    _map(left, lang)
    left.plot(0, 0, marker="^", ms=15, color=INK, ls="none", label=t("radar", lang), zorder=5)
    left.plot(p[:, 0], p[:, 1], color=INK, ls="--", lw=1.8, label=t("true_path", lang))
    left.plot(raw[:, 0], raw[:, 1], ls="none", marker="o", ms=6, color=BLUE, alpha=0.8, mew=0, label=t("reports", lang))
    left.plot(x[:, 0], x[:, 1], color=TERRACOTTA, lw=2.4, label=t("ekf_estimate", lang))
    _frame(left, (p[:, 0].min() - 60, -60), (p[:, 0].max() + 60, p[:, 1].max() + 60))
    left.set_title(t("fb_title_map", lang).format(north=num(p[0, 1], lang)))
    _legend(left, "lower center", ncols=2, columnspacing=1.6)

    _style(right, lang, int_x=True)
    k = np.arange(1, len(p) + 1)
    right.fill_between(k, 0, 2 * np.asarray(d["claimed"]), color=TERRACOTTA, alpha=0.15, lw=0, label=t("fb_band", lang))
    right.plot(k, d["raw_error"], ls="none", marker="o", ms=6, color=BLUE,
               label=t("fb_raw", lang).format(rms=_short(d["rms_raw"], lang)))
    right.plot(k, d["error"], color=TERRACOTTA, lw=2.6, label=t("fb_filter", lang).format(rms=_short(d["rms_filter"], lang)))
    right.set_xlim(0.5, k[-1] + 0.5)
    right.set_ylim(0, 1.25 * max(np.max(d["raw_error"]), 2 * np.max(d["claimed"])))
    right.set_xlabel(t("scan", lang))
    right.set_ylabel(t("fb_distance", lang))
    right.set_title(t("fb_title_error", lang))
    _legend(right, "upper right")


# --- ekf-08-wrap ------------------------------------------------------------------------

def _unwrapped_gaps(values):
    """Scans 1..n and a wrapped angle, with a gap wherever it jumps by more than pi, so no line crosses the axis."""
    v = np.asarray(values, dtype=float)
    cut = np.flatnonzero(np.abs(np.diff(v)) > np.pi) + 1
    return np.insert(np.arange(1.0, len(v) + 1), cut, np.nan), np.insert(v, cut, np.nan)


def _mark_exit(ax, track, lang):
    """Point at the edge where `track` (one [east, north] row per scan) first leaves the map, and name the scan."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    on_map = (track[:, 0] > x0) & (track[:, 0] < x1) & (track[:, 1] > y0) & (track[:, 1] < y1)
    k = int(np.argmin(on_map))                                           # the first scan off the map, 0-based
    if on_map.all() or k == 0:
        return
    a, b = track[k - 1], track[k]
    s = min((lim - a[i]) / (b[i] - a[i]) for i, lims in ((0, (x0, x1)), (1, (y0, y1))) for lim in lims
            if b[i] != a[i] and 0 < (lim - a[i]) / (b[i] - a[i]) <= 1)
    edge = a + s * (b - a)                                               # where the segment crosses the frame
    east, north = (edge[0] - x0) / (x1 - x0) > 0.6, (edge[1] - y0) / (y1 - y0) > 0.5
    note = ax.annotate(t("wr_leaves", lang).format(k=k + 1), xy=edge, textcoords="offset points",
                       xytext=(-20 if east else 20, -40 if north else 40), ha="right" if east else "left",
                       va="top" if north else "bottom", color=GREEN, fontsize=15,
                       arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.6})
    note.set_in_layout(False)                                            # it lies inside the map


def wrap(fig, d, lang):
    left, right = fig.subplots(1, 2)
    p = np.array(d["scn"]["p"])
    good, bad = (np.array(d[run]["x"])[:, :2] for run in ("wrapped_run", "plain_run"))

    _map(left, lang)
    west = p[:, 0].min() - 60
    left.plot(0, 0, marker="^", ms=15, color=INK, ls="none", label=t("radar", lang), zorder=5)
    left.plot([west - 1000, 0], [0, 0], color=MUTED, ls=":", lw=2.2, label=t("wr_cut", lang))
    left.plot(p[:, 0], p[:, 1], color=INK, ls="--", lw=1.8, label=t("true_path", lang))
    left.plot(*good.T, color=TERRACOTTA, lw=2.6, marker="o", ms=5, zorder=3,
              label=t("wr_with", lang).format(error=_short(d["max_error_wrapped"], lang)))
    left.plot(*bad.T, color=GREEN, lw=2.4, marker="o", ms=5,
              label=t("wr_without", lang).format(error=_short(d["max_error_plain"], lang)))
    _frame(left, (west, p[:, 1].min() - 60), (60, p[:, 1].max() + 60))  # the path and the radar, not the lost track
    left.set_title(t("wr_title_map", lang))
    _legend(left, "best")                                                # clear of the lost track, wherever it goes

    _style(right, lang, int_x=True)
    z, z_hat = np.array(d["wrapped_run"]["z"])[:, 1], np.array(d["wrapped_run"]["z_hat"])[:, 1]
    right.plot(*_unwrapped_gaps(z_hat), color=TERRACOTTA, lw=2.6, label=t("wr_expected", lang))
    right.plot(np.arange(1, len(z) + 1), z, ls="none", marker="o", ms=8, color=BLUE, label=t("wr_reported", lang), zorder=4)
    k = d["jump_scan"]
    if k is None:
        right.text(0.5, 0.5, t("wr_no_jump", lang), transform=right.transAxes, ha="center", va="center", fontsize=15)
    else:
        right.annotate("", xy=(k, d["z_theta"]), xytext=(k, d["z_hat_theta"]),
                       arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6, "shrinkA": 7, "shrinkB": 7})
        jump = t("wr_jump", lang).format(k=k, raw=num(d["raw"], lang, 4), wrapped=num(d["wrapped"], lang, 4))
        roomy_right = len(z) + 0.5 - k >= k - 0.5                       # write the numbers on the wider side
        right.text(k + 0.8 if roomy_right else k - 0.8, 0, jump, ha="left" if roomy_right else "right", va="center",
                   fontsize=15, linespacing=1.5)
    right.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    right.set_ylim(-np.pi - 0.25, np.pi + 0.25)
    right.set_xlim(0.5, len(z) + 0.5)
    right.set_xlabel(t("scan", lang))
    right.set_ylabel(t("wr_bearing", lang))
    right.set_title(t("wr_title_bearing", lang))
    _legend(right, "upper right", bbox_to_anchor=(1, 0.86))             # below the readings near +pi

    fig.draw_without_rendering()                                         # the layout and the equal scale settle here
    _mark_exit(left, bad, lang)


# --- ekf-09-consistency -----------------------------------------------------------------

def consistency(fig, d, lang):
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, (name, c) in zip(axes.ravel(), d["cases"].items()):
        _style(ax, lang, int_x=True)
        k = np.arange(1, len(c["rms"]) + 1)
        ax.plot(k, c["rms"], color=TERRACOTTA, lw=2.6)
        ax.plot(k, c["claimed"], color=INK, ls="--", lw=2)
        ax.set_title(t("cs_panel", lang).format(name=t(f"cs_{name}", lang), bend=_short(c["bend"], lang),
                                                sigma_r=num(c["sigma_r"], lang)), fontsize=16, pad=8)
    ax.set_yscale("log")                                                 # shared by all four panels
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(tick_formatter(lang)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes.ravel():
        ax.tick_params(which="minor", length=0)
    for ax in axes[1]:
        ax.set_xlabel(t("scan", lang))
    for ax in axes[:, 0]:
        ax.set_ylabel(t("cs_error", lang))
    axes[0, 0].set_xlim(1, len(k))
    handles = [Line2D([], [], color=TERRACOTTA, lw=2.6, label=t("cs_rms", lang).format(runs=d["runs"])),
               Line2D([], [], color=INK, ls="--", lw=2, label=t("cs_claimed", lang))]
    axes[0, 1].legend(handles=handles, loc="upper right", fontsize=14)
    fig.suptitle(t("cs_title", lang), x=0.06, ha="left", fontsize=21, fontweight="bold")


# --- lesson 3 ---------------------------------------------------------------------------

def _gauss_label(key, mean, var, lang):
    return t(key, lang).format(mean=_short(mean, lang), var=_short(var, lang))


# --- ukf-02-icd-decode ------------------------------------------------------------------

def _icd_value(row, d, lang):
    """A decoded field as written under its bytes: the angle with its degrees, a number with its unit."""
    v, unit = row["value"], row["unit"]
    if row["field"] in ("az", "el"):
        return f"{num(v, lang, 4)} rad = {num(d[row['field'] + '_deg'], lang, 2)}°"
    if isinstance(v, float):
        return f"{num(v, lang, 4)} {unit}"
    return f"{v} {unit}".strip()                                         # a whole number has no decimal to localise


def icd(fig, d, lang):
    ax = fig.add_subplot()
    ax.axis("off")
    msg, colors = d["message"], (BLUE_LIGHT, "#e3b494")
    for k, row in enumerate(d["rows"]):
        first, last = (int(b) for b in (row["bytes"].split("-") * 2)[:2])
        line, col = divmod(first, 8)
        y = -1.25 * line
        for b in range(first, last + 1):
            ax.add_patch(matplotlib.patches.Rectangle((b % 8 + 0.04, y), 0.92, 1, color=colors[k % 2], alpha=0.55, lw=0))
            ax.text(b % 8 + 0.5, y + 0.78, f"{msg[b]:02x}", ha="center", va="center", fontsize=18, family="DejaVu Sans Mono")
        mid = (first + last + 1) / 2 - 8 * line
        ax.text(mid, y + 0.45, row["field"], ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(mid, y + 0.17, _icd_value(row, d, lang), ha="center", va="center", fontsize=14)
    for line in range(4):
        ax.text(-0.15, -1.25 * line + 0.5, t("icd_header" if line == 0 else "icd_payload", lang), ha="right",
                va="center", fontsize=15, color=MUTED)
    ax.text(0, -4.1, t("icd_note", lang), fontsize=14, color=MUTED)
    ax.set_xlim(-1.4, 8.05)
    ax.set_ylim(-4.3, 1.1)
    ax.set_title(t("icd_title", lang))


# --- ukf-03-square ----------------------------------------------------------------------

def square(fig, d, lang):
    left, right = fig.subplots(1, 2)
    mu, var = d["mu"], d["var"]
    sigma = np.sqrt(var)
    x = np.linspace(mu - 3.2 * sigma, mu + 3.2 * sigma, 401)

    _style(left, lang)
    left.axvspan(mu - 2 * sigma, mu + 2 * sigma, color=TERRACOTTA, alpha=0.10, lw=0,
                 label=t("sq_belief", lang).format(mu=num(mu, lang), sigma=num(sigma, lang)))
    left.plot(x, x**2, color=INK, lw=3, label=t("sq_curve", lang))
    left.plot(x, mu**2 + 2 * mu * (x - mu), color=MUTED, ls="--", lw=2.4, label=t("sq_tangent", lang).format(mu=num(mu, lang)))
    left.scatter(d["x"], d["y"], s=[900 * w for w in d["w"]], color=TERRACOTTA, zorder=4, label=t("sigma_points", lang))
    for xi, yi, w in zip(d["x"], d["y"], d["w"]):
        left.annotate(f"({_short(xi, lang)}; {_short(yi, lang)})" if lang == "tr" else f"({_short(xi, lang)}, {_short(yi, lang)})",
                      (xi, yi), textcoords="offset points", xytext=(-14, 12), ha="right", fontsize=14)
    left.set_xlim(x[0], x[-1])
    left.set_xlabel("x")
    left.set_ylabel("y")
    left.set_title(t("sq_title_curve", lang).format(mu=num(mu, lang), var=num(var, lang)))
    _legend(left, "upper left")

    _style(right, lang, y_ticks=False)
    y = d["grid"] ** 2
    edges = np.arange(0, y.max() + 0.25, 0.25)
    density, _ = np.histogram(y, bins=edges, weights=d["weights"], density=True)
    g = np.linspace(-10, 24, 681)
    right.fill_between((edges[:-1] + edges[1:]) / 2, density, color=MUTED, alpha=0.45, lw=0,
                       label=_gauss_label("sq_true", d["exact_mean"], d["exact_var"], lang))
    for key, color, ls in (("ekf", MUTED, "--"), ("ukf", TERRACOTTA, "-")):
        mean, v = d[f"{key}_mean"], d[f"{key}_var"]
        right.plot(g, _normal(g, mean, v), color=color if key == "ukf" else INK, ls=ls, lw=3 if key == "ukf" else 2.4,
                   label=_gauss_label(f"sq_{key}", mean, v, lang))
    right.axvline(d["exact_mean"], color=INK, lw=1.6, ymax=0.72)                 # the true mean, below the legend
    right.set_xlim(g[0], g[-1])
    right.set_ylim(0, 1.35 * density.max())
    right.set_xlabel("y = x²")
    right.set_ylabel(t("density", lang))
    right.set_title(t("sq_title_density", lang))
    _legend(right, "upper right")


# --- ukf-05-sigma-points ----------------------------------------------------------------

def _points(ax, X, w, color, marker, label=None, ms=11):
    ax.plot(X[0, 1:], X[1, 1:], ls="none", marker=marker, ms=ms, color=color, label=label, zorder=4)
    ax.plot(X[0, 0], X[1, 0], ls="none", marker=marker, ms=ms + 5, color=color, zorder=4)


def sigma_points(fig, d, lang):
    one, round_, tilted = fig.subplots(1, 3, width_ratios=[1.15, 1, 1])
    X1, w1 = (np.asarray(a) for a in d["sets"]["1d"])
    mu, var = d["mean_1d"], d["var_1d"]
    sigma = np.sqrt(var)

    _style(one, lang, y_ticks=False)
    g = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 401)
    one.plot(g, _normal(g, mu, var), color=INK, lw=2.4)
    one.fill_between(g, _normal(g, mu, var), color=MUTED, alpha=0.15, lw=0)
    peak = _normal(mu, mu, var)
    for xi, w in zip(X1[0], w1):
        one.vlines(xi, 0, 1.2 * peak * w / w1[0], color=TERRACOTTA, lw=4)
        one.text(xi, 1.2 * peak * w / w1[0] + 0.03 * peak, f"{_short(xi, lang)}\n{t('weight', lang)} {num(w, lang, 3)}",
                 ha="center", va="bottom", fontsize=13)
    one.set_xlim(g[0], g[-1])
    one.set_ylim(0, 1.55 * peak)
    one.set_xlabel(t("sp_position", lang))
    one.set_title(t("sp_title_1d", lang).format(mu=num(mu, lang), var=num(var, lang)))

    for ax, name, P, mean in ((round_, "round", d["P_round"], d["mean_round"]),
                              (tilted, "cholesky", d["P_plus"], d["x_plus"])):
        _map(ax, lang)
        mean = np.asarray(mean)
        ax.plot(*(mean[:, None] + _ellipse(np.asarray(P) * 0.75)), color=INK, lw=1.8, ls=":",
                label=t("sp_contour", lang))                              # 2 sqrt(3/4) = sqrt(3)
        X, w = (np.asarray(a) for a in d["sets"][name])
        _points(ax, X, w, BLUE if name == "cholesky" else TERRACOTTA, "s" if name == "cholesky" else "o",
                t("sp_cholesky", lang) if name == "cholesky" else None)
    Xa, wa = (np.asarray(a) for a in d["sets"]["axes"])
    _points(tilted, Xa, wa, TERRACOTTA, "o", t("sp_axes", lang), ms=10)
    for ax in (round_, tilted):
        _frame(ax, np.asarray(ax.dataLim.min) - 6, np.asarray(ax.dataLim.max) + 6)
    round_.set_title(t("sp_title_round", lang))
    tilted.set_title(t("sp_title_tilted", lang))
    handles, _ = tilted.get_legend_handles_labels()
    w0, wi = num(wa[0], lang, 3), num(wa[1], lang, 3)
    handles += [Line2D([], [], ls="none", marker="o", ms=15, color=MUTED, label=t("sp_mean", lang).format(w=w0)),
                Line2D([], [], ls="none", marker="o", ms=10, color=MUTED, label=t("sp_others", lang).format(w=wi))]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, columnspacing=1.6)


# --- ukf-07-radar-update, ukf-08-wide-belief --------------------------------------------

def ukf_update(fig, d, lang):
    left, right = fig.subplots(1, 2, width_ratios=[1, 1.25])
    e, x0, P0 = d["ekf"], d["x_minus"], d["P_minus"]
    s0 = np.sqrt(P0)

    _style(left, lang)
    x = np.linspace(x0 - 2.6 * s0, x0 + 2.6 * s0, 401)
    left.plot(x, np.hypot(x, d["offset"]), color=INK, lw=3, label=t("uu_curve", lang).format(offset=num(d["offset"], lang)))
    left.scatter(d["points"], d["ranges"], s=[900 * w for w in d["weights"]], color=TERRACOTTA, zorder=4,
                 label=t("sigma_points", lang))
    left.axhline(d["z"], color=BLUE, ls=":", lw=2.4, label=t("uu_reading", lang).format(z=num(d["z"], lang)))
    left.axhline(d["z_hat"], color=TERRACOTTA, ls="--", lw=1.8, label=t("uu_z_hat", lang).format(z_hat=_short(d["z_hat"], lang)))
    left.set_xlim(x[0], x[-1])
    left.set_xlabel(t("uu_x", lang))
    left.set_ylabel(t("range_m", lang))
    left.set_title(t("uu_title_curve", lang))
    _legend(left, "upper left")

    _style(right, lang, y_ticks=False)
    spreads = (d["sigma"], d["ekf_sigma"], d["exact_sigma"])
    means = (d["x_plus"], e["x_plus"], e["exact_mean"])
    lo, hi = min(means) - 4 * max(spreads), max(means) + 4 * max(spreads)
    g = np.linspace(lo, hi, 1001)
    floor = (g[1] - g[0]) ** 2                                          # a bell narrower than one step draws as a spike
    P_ekf, P_ukf = max(e["P_plus"], floor), max(d["P_plus"], floor)
    var_word = t("variance", lang)
    keep = (e["grid"] >= lo) & (e["grid"] <= hi)
    step = max(1, int(keep.sum()) // 2000)
    density = e["posterior"][keep][::step] / (e["grid"][1] - e["grid"][0])
    right.fill_between(e["grid"][keep][::step], density, color=MUTED, alpha=0.45, lw=0,
                       label=f"{t('exact', lang)}: {_short(e['exact_mean'], lang)} ({var_word} {_short(e['exact_var'], lang)})")
    right.plot(g, _normal(g, e["x_plus"], P_ekf), color=INK, ls="--", lw=2.4,
               label=f"EKF: {_short(e['x_plus'], lang)} ({var_word} {_short(e['P_plus'], lang)})")
    right.plot(g, _normal(g, d["x_plus"], P_ukf), color=TERRACOTTA, lw=3.4,
               label=f"UKF: {_short(d['x_plus'], lang)} ({var_word} {_short(d['P_plus'], lang)})")
    right.set_xlim(lo, hi)
    right.set_ylim(0, 1.3 * max(density.max(), _normal(0, 0, min(P_ekf, P_ukf))))
    right.set_xlabel(t("uu_x", lang))
    right.set_ylabel(t("density", lang))
    right.set_title(t("uu_title_post", lang).format(z=num(d["z"], lang), x=num(x0, lang), sigma=_short(s0, lang),
                                                    sigma_r=_short(np.sqrt(d["R"]), lang)), fontsize=18)
    _legend(right, "upper left")


PLOTS = {
    "01-fusion": fusion,
    "03-1d-loop": one_d_loop,
    "05-derivative-vs-filter": derivative_vs_filter,
    "08-covariance": covariance,
    "12-velocity-learning": velocity_learning,
    "13-q-r-tradeoff": tradeoff,
    "14-nis": nis,
    "ekf-02-tangent": tangent,
    "ekf-03-range-only": range_only,
    "ekf-05-radar-update": radar_update,
    "ekf-07-flyby": flyby,
    "ekf-08-wrap": wrap,
    "ekf-09-consistency": consistency,
    "ukf-02-icd-decode": icd,
    "ukf-03-square": square,
    "ukf-05-sigma-points": sigma_points,
    "ukf-07-radar-update": ukf_update,
    "ukf-08-wide-belief": ukf_update,
}


def save(sid: str, data: dict, out_dir: Path, lang: str = "en") -> Path:
    """Draw the figure of simulation `sid` and write <out_dir>/<sid>.png (1600 x 900 px)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sid}.png"
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(SIZE_PX[0] / DPI, SIZE_PX[1] / DPI), dpi=DPI, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.3, h_pad=0.3)
        PLOTS[sid](fig, data, lang)
        fig.savefig(path, dpi=DPI, metadata={"Software": None})
    return path
