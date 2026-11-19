import json
import struct

import numpy as np
import pytest

from fbh import ekf, registry, scenarios, sims2
from fbh.base import Options

SEEDS = range(20)


@pytest.fixture(scope="module")
def cases():
    """ekf-09-consistency at its default seed and 300 runs: a few seconds, so it is computed once here."""
    return sims2.consistency()["cases"]


# --- the scenarios ----------------------------------------------------------------------

def test_flyby_scenario_follows_its_docstring():
    scn = scenarios.flyby()
    assert len(scn["p"]) == len(scn["noise"]) == 60 and all(len(e) == 2 for e in scn["noise"])
    assert scn["p"][0] == [-580.0, 400.0] and scn["p"][29] == [0.0, 400.0] and scn["p"][44] == [300.0, 400.0]
    assert scn["v"] == [20.0, 0.0]
    np.testing.assert_allclose(np.diff(scn["p"], axis=0), [[20.0, 0.0]] * 59)               # 20 m/s, one scan a second
    scan_0 = np.array([-600.0, 400.0, 20.0, 0.0])                                         # the truth one scan before scan 1
    np.testing.assert_allclose(np.array(scn["x0"]) - scan_0, [20.0, 20.0, -4.0, 3.0])       # one sigma off ...
    np.testing.assert_allclose(np.sqrt(np.diag(scn["P0"])), [30.0, 30.0, 5.0, 5.0])         # ... and P0 admits it
    np.testing.assert_allclose(scn["noise"], np.random.default_rng(0).standard_normal((60, 2)))
    assert json.loads(json.dumps(scn)) == scn


def test_crossing_scenario_crosses_the_cut_between_scans_10_and_11():
    scn = scenarios.crossing()
    p = np.array(scn["p"])
    assert len(p) == 30 and scn["v"] == [-15.0, -5.0]
    assert (p[:, 0] < 0).all()                                                          # always west of the radar
    assert (p[:10, 1] > 0).all() and (p[10:, 1] < 0).all()                             # north up to scan 10, then south
    bearing = np.arctan2(p[:, 1], p[:, 0])
    assert (np.abs(np.abs(bearing[8:12]) - np.pi) < 0.03).all()                        # scans 9 to 12 hug +-pi
    assert json.loads(json.dumps(scn)) == scn


# --- the tracker --------------------------------------------------------------------------

def test_run_radar_is_json_ready_and_has_the_agreed_keys():
    run = sims2.run_radar(scenarios.flyby())
    assert set(run) == {"xm", "Pm", "x", "P", "z", "z_hat", "y", "S", "K", "nis"}
    assert all(len(v) == 60 for v in run.values())
    assert json.loads(json.dumps(run)) == run                                          # plain lists and floats only
    shapes = {"xm": (60, 4), "Pm": (60, 4, 4), "x": (60, 4), "P": (60, 4, 4), "z": (60, 2), "z_hat": (60, 2),
              "y": (60, 2), "S": (60, 2, 2), "K": (60, 4, 2), "nis": (60,)}
    for key, shape in shapes.items():
        assert np.shape(run[key]) == shape, key
    z = np.array(run["z"])
    assert (z[:, 1] >= -np.pi).all() and (z[:, 1] < np.pi).all()                     # bearings in [-pi, pi)


def test_run_radar_steps_are_predict_then_update():
    scn = scenarios.flyby()
    run = sims2.run_radar(scn)
    F, Q = ekf.cv2d_model(1.0, sims2.Q)
    np.testing.assert_allclose(run["xm"][0], F @ np.array(scn["x0"]))                   # scan 1 predicts from x0
    np.testing.assert_allclose(run["Pm"][0], F @ np.array(scn["P0"]) @ F.T + Q)
    for k in range(1, 60):
        np.testing.assert_allclose(run["xm"][k], F @ np.array(run["x"][k - 1]))
    np.testing.assert_allclose(run["z_hat"], [ekf.h_radar(np.array(x).reshape(4, 1)).ravel() for x in run["xm"]])


def test_changing_a_sigma_rescales_the_same_draws():
    scn = scenarios.flyby()
    p = np.array(scn["p"])
    truth = np.column_stack([np.hypot(p[:, 0], p[:, 1]), np.arctan2(p[:, 1], p[:, 0])])
    a, b = sims2.measurements(scn, 5.0, 0.02), sims2.measurements(scn, 10.0, 0.01)
    np.testing.assert_allclose(b[:, 0] - truth[:, 0], 2 * (a[:, 0] - truth[:, 0]))
    np.testing.assert_allclose(ekf.wrap(b[:, 1] - truth[:, 1]), 0.5 * ekf.wrap(a[:, 1] - truth[:, 1]), atol=1e-12)


# --- section 7: the fly-by ----------------------------------------------------------------

def test_flyby_summary_numbers():
    d = sims2.flyby()
    assert d["seed"] == 4 and (d["q"], d["sigma_r"], d["sigma_theta"]) == (0.1, 5.0, 0.02)
    got = {key: round(value, 3) for key, value in sims2.SIMS["ekf-07-flyby"](Options()).summary.items()
           if isinstance(value, float)}
    assert got == {"q": 0.1, "sigma_r": 5.0, "sigma_theta": 0.02, "rms_raw": 11.091, "rms_filter": 5.039,
                   "error_45": 4.762, "claimed_45": 5.264, "nis_mean": 1.81, "nis_above": 0.05}
    np.testing.assert_allclose(d["v_last"], [20.266, -0.163], atol=5e-4)
    assert int(np.sum(np.array(d["run"]["nis"]) > d["bound"])) == 3 and d["bound"] == 5.99   # 3 of the 60 scans


def test_flyby_raw_and_errors_mean_what_they_say():
    d = sims2.flyby()
    z, p = np.array(d["run"]["z"]), np.array(d["scn"]["p"])
    np.testing.assert_allclose(d["raw"], z[:, :1] * np.column_stack([np.cos(z[:, 1]), np.sin(z[:, 1])]))
    np.testing.assert_allclose(d["raw_error"], np.hypot(*(d["raw"] - p).T))
    x, P = np.array(d["run"]["x"]), np.array(d["run"]["P"])
    np.testing.assert_allclose(d["error"], np.hypot(*(x[:, :2] - p).T))
    np.testing.assert_allclose(d["claimed"] ** 2, P[:, 0, 0] + P[:, 1, 1])
    np.testing.assert_allclose(d["sigma_along"] ** 2 + d["sigma_across"] ** 2, d["claimed"] ** 2)   # a rotation
    assert d["rms_filter"] == pytest.approx(np.sqrt(np.mean(d["error"][9:] ** 2)))                  # scans 10 to 60


def test_the_filter_beats_the_raw_reports_for_every_seed():
    for seed in SEEDS:
        d = sims2.flyby(seed=seed)
        assert d["rms_filter"] < d["rms_raw"], seed


# --- section 8: the bearing wraps ---------------------------------------------------------

def test_wrap_demo_numbers():
    d = sims2.wrap_demo()
    assert d["jump_scan"] == 10 and d["jump_scans"] == [10]
    assert round(d["raw"], 4) == -6.2545 and round(d["wrapped"], 4) == 0.0287
    assert d["wrapped"] == pytest.approx(ekf.wrap(d["raw"])) and d["raw"] == pytest.approx(d["z_theta"] - d["z_hat_theta"])
    assert d["z_theta"] < 0 < d["z_hat_theta"]                                         # opposite sides of the cut
    assert round(d["max_error_wrapped"], 2) == 12.62 and round(d["max_error_plain"], 2) == 1159.48
    np.testing.assert_allclose(d["error_plain"][:9], d["error_wrapped"][:9])           # the same run up to the jump


def test_without_the_wrap_is_never_better_and_far_worse_after_a_jump():
    no_jump = []
    for seed in SEEDS:
        d = sims2.wrap_demo(seed=seed)
        if d["jump_scan"] is None:
            no_jump.append(seed)
            assert d["jump_scans"] == [] and d["raw"] is None and d["wrapped"] is None
            np.testing.assert_array_equal(d["error_plain"], d["error_wrapped"])       # nothing to wrap: one run
        else:
            assert d["max_error_plain"] >= 10 * d["max_error_wrapped"], seed
        assert d["max_error_plain"] >= d["max_error_wrapped"], seed
    assert no_jump == [13, 19]


def test_wrap_figure_draws_without_a_jump(tmp_path):
    res = sims2.SIMS["ekf-08-wrap"](Options(seed=13))
    assert res.summary["jump_scan"] is None
    for lang in ("en", "tr"):
        assert res.plot(tmp_path / lang, lang).stat().st_size > 5000


# --- section 9: where the tangent fails ---------------------------------------------------

def test_consistency_bends(cases):
    assert {name: round(c["bend"], 3) for name, c in cases.items()} == {
        "baseline": 0.639, "vague_prior": 63.87, "first_report": 0.141, "sharp_range": 3.523}
    assert [c["sigma_r"] for c in cases.values()] == [5.0, 5.0, 5.0, 0.5]


def _qualitative_facts(cases, label):
    for name in ("baseline", "first_report"):                                          # honest: NIS 2, ratio 1
        c = cases[name]
        assert abs(c["nis_early"] - 2) < 0.25 and abs(c["nis_late"] - 2) < 0.25, (label, name)
        assert abs(c["ratio"] - 1) < 0.1, (label, name)
    assert cases["vague_prior"]["nis_early"] > 20 and cases["vague_prior"]["ratio"] > 2, label
    assert cases["sharp_range"]["nis_late"] > 2.8 and cases["sharp_range"]["ratio"] > 2, label


def test_consistency_honest_and_overconfident_cases(cases):
    _qualitative_facts(cases, "seed 0")


@pytest.mark.parametrize("seed", [1, 2])
def test_consistency_tells_the_same_story_for_other_seeds(seed):
    d = sims2.consistency(seed=seed, runs=150)
    assert d["runs"] == 150 and all(len(c["rms"]) == 60 for c in d["cases"].values())
    _qualitative_facts(d["cases"], f"seed {seed}")


def test_consistency_series_are_per_scan(cases):
    for name, c in cases.items():
        assert len(c["rms"]) == len(c["claimed"]) == 60, name
        assert c["ratio_worst"] == pytest.approx(np.max(c["rms"] / c["claimed"])), name
        assert c["rms_first"] == c["rms"][0] and c["claimed_first"] == c["claimed"][0], name


# --- every id: runs, draws, JSON ----------------------------------------------------------

@pytest.mark.parametrize("sid", list(sims2.SIMS))
def test_every_lesson_two_id_runs_plots_both_languages_and_is_json_ready(sid, tmp_path):
    res = sims2.SIMS[sid](Options())
    json.dumps(res.table, allow_nan=False), json.dumps(res.summary, allow_nan=False)
    assert res.table and set(res.summary) and all(isinstance(row, dict) for row in res.table)
    pngs = [res.plot(tmp_path / lang, lang) for lang in ("en", "tr")]
    for png in pngs:
        assert png.name == f"{sid}.png"
        assert struct.unpack(">II", png.read_bytes()[16:24]) == (1600, 900), (sid, png)
    assert pngs[0].read_bytes() != pngs[1].read_bytes()
    assert registry.entry(sid)                                                         # and it is in the registry
