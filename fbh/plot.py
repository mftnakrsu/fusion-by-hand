"""The figures, one function per simulation id. Light paper style, 1600 x 900 px, EN or TR labels.

Figures are built straight on matplotlib's Figure class, so no display and no global pyplot state
is involved. Turkish figures use decimal commas in tick labels and in every number printed on them.
"""
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .i18n import t

PAPER, INK, TERRACOTTA, BLUE = "#fbfaf7", "#2b2d33", "#c0612b", "#3d6fb0"
MUTED, GRID = "#8a8d96", "#e8e5de"
BLUE_LIGHT = "#8fb0d8"
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
    _style(ax, lang, int_x=True)
    k = d["k"]
    sigma = "σ"
    ax.axhline(d["truth_v"], color=INK, ls="--", lw=1.8, label=t("truth", lang))
    ax.plot(k, d["derivative"], color=BLUE, lw=1.8, marker="o", ms=6,
            label=f"{t('deriv_difference', lang)} ({sigma} = {num(d['std_derivative'], lang, 1)})")
    ax.plot(k, d["filter"], color=TERRACOTTA, lw=3.4,
            label=f"{t('filter', lang)} ({sigma} = {num(d['std_filter'], lang, 1)})")
    lim = 1.15 * float(np.max(np.abs(d["derivative"])))
    ax.set_ylim(-lim, lim)
    ax.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    ax.set_xlabel(t("frame", lang))
    ax.set_ylabel(t("deriv_ylabel", lang))
    ax.set_title(t("deriv_title", lang))
    _legend(ax, "upper right")


# --- 08-covariance ----------------------------------------------------------------------

def _ellipse(P, n=200):
    a = np.linspace(0, 2 * np.pi, n)
    return 2 * np.linalg.cholesky(np.array(P)) @ np.vstack([np.cos(a), np.sin(a)])


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
    ax = fig.add_subplot()
    _style(ax, lang, int_x=True)
    k, window = np.array(d["k"]), 20
    for key, color, name in (("overconfident", TERRACOTTA, t("nis_over", lang)), ("honest", BLUE, t("nis_honest", lang))):
        ax.plot(k, d[key], ls="none", marker="o", ms=4.5, color=color, alpha=0.22, mew=0)
        ax.plot(k[window - 1:], _rolling(d[key], window), color=color, lw=3.2,
                label=f"{name}: {t('mean', lang)} {num(d[key + '_mean'], lang, 1)}")
    ax.axhline(1, color=INK, ls="--", lw=1.8, label=t("nis_expected", lang))
    ax.axhline(d["bound"], color=MUTED, ls=":", lw=2, label=f"{t('nis_bound', lang)}: {num(d['bound'], lang, 2)}")
    ax.axvline(d["maneuver"], color=MUTED, lw=1.2, ls=":")
    ax.text(d["maneuver"] + 3, 15.2, t("maneuver", lang), color=MUTED, fontsize=15)
    ax.set_ylim(0, 16)
    ax.set_xlim(k[0], k[-1])
    ax.set_xlabel(t("frame", lang))
    ax.set_ylabel(f"{t('nis_ylabel', lang)} ({t('nis_window', lang)})")
    ax.set_title(t("nis_title", lang))
    _legend(ax, "upper right")


PLOTS = {
    "01-fusion": fusion,
    "03-1d-loop": one_d_loop,
    "05-derivative-vs-filter": derivative_vs_filter,
    "08-covariance": covariance,
    "12-velocity-learning": velocity_learning,
    "13-q-r-tradeoff": tradeoff,
    "14-nis": nis,
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
