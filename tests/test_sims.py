import json
import struct

import numpy as np
import pytest

from fbh import i18n, plot, scenarios, sims, sims2, sims3


def test_velocity_learning_table_q001():
    r = sims.velocity_learning(q=0.01)            # md §12, default_rng(0), truth v=13, start belief 10
    got = {row["k"]: (round(row["v"], 2), round(row["sigma_v"], 2)) for row in r["rows"]}
    assert got[1] == (11.39, 1.49) and got[2] == (12.30, 0.90) and got[3] == (12.96, 0.57)
    assert got[10] == (12.87, 0.20) and got[30] == (12.94, 0.20)


def test_q_tradeoff_table():
    a, b = sims.velocity_learning(q=0.01), sims.velocity_learning(q=1.0)
    kv = lambda r, k: round(r["rows"][k - 1]["K_v"], 2)
    v = lambda r, k: round(r["rows"][k - 1]["v"], 2)
    assert [v(b, k) for k in (1, 3, 10, 20, 30)] == [11.51, 13.31, 12.08, 13.62, 13.45]
    assert [kv(a, k) for k in (1, 3, 10, 20, 30)] == [0.44, 0.37, 0.08, 0.08, 0.08]   # md said 0.45 at k=1: corrected
    assert [kv(b, k) for k in (1, 3, 10, 20, 30)] == [0.48, 0.53, 0.49, 0.49, 0.49]
    assert round(a["std_v_10_30"], 2) == 0.13 and round(b["std_v_10_30"], 2) == 0.44


def test_tracking_gap_grows_uncertainty_and_k_jumps_after():
    scn = scenarios.tracking()
    run = sims.run_cv(scn, q=0.01, r=4.0)
    sd = [np.sqrt(P[0][0]) for P in run["P"]]
    for k in range(20, 27):                 # 1-based frames 20..27 → indices 19..26
        assert sd[k] > sd[k - 1]
    assert run["z"][19] is None and run["K"][27] is not None
    assert run["K"][27][0] > run["K"][18][0]


def test_r_slider_scales_same_noise():
    scn = scenarios.tracking()
    a, b = sims.run_cv(scn, 0.01, 1.0), sims.run_cv(scn, 0.01, 4.0)
    za = np.array([z for z in a["z"] if z is not None]); zb = np.array([z for z in b["z"] if z is not None])
    p = np.array([scn["p"][i] for i, z in enumerate(a["z"]) if z is not None])
    np.testing.assert_allclose(zb - p, 2 * (za - p))


def test_derivative_amplifies_noise():
    r = sims.derivative_vs_filter()
    assert r["std_derivative"] > 3 * r["std_filter"]


def test_covariance_cloud_matches_formula():
    r = sims.covariance_cloud(n=20000, seed=3)
    np.testing.assert_allclose(r["P_after_empirical"], r["P_after_formula"], rtol=0.05, atol=0.05)


def test_nis_honest_vs_overconfident():
    r = sims.nis(seed=1)
    assert 0.85 < r["honest_mean"] < 1.15
    assert r["overconfident_mean"] > 3


def test_every_sim_runs_and_plots(tmp_path):
    for sid, fn in sims.SIMS.items():
        for lang in ("en", "tr"):
            res = fn(sims.Options(lang=lang))
            png = res.plot(tmp_path, lang)
            assert png.exists() and png.stat().st_size > 5000, sid


# --- beyond the brief: contracts other tasks rely on ---------------------------------

def test_scenario_truth_follows_the_brief():
    scn = scenarios.tracking()
    assert len(scn["p"]) == len(scn["v"]) == len(scn["noise"]) == 60
    assert scn["x0"] == [200, 7] and scn["P0"] == [[16, 0], [0, 9]]
    assert scn["gap"] == [20, 27] and scn["maneuver"] == 36
    assert scn["v"][34] == 10 and scn["v"][35] == 14                 # k=35 is still 10, k=36 is 14
    assert scn["p"][0] == 210 and scn["p"][1] == 220                 # p_k = p_{k-1} + v_k from p_0 = 200
    assert scenarios.tracking(gap=None)["gap"] is None
    np.testing.assert_allclose(scn["noise"], np.random.default_rng(7).standard_normal(60))


def test_run_cv_is_json_ready_and_has_the_agreed_keys():
    run = sims.run_cv(scenarios.tracking(), 0.01, 4.0)
    assert set(run) == {"xm", "Pm", "x", "P", "y", "S", "K", "nis", "z"}
    assert all(len(v) == 60 for v in run.values())
    assert json.loads(json.dumps(run)) == run                        # plain lists, floats, None only
    assert all(run[key][k] is None for key in ("z", "y", "S", "K", "nis") for k in range(19, 27))
    assert run["P"][19] == run["Pm"][19]                              # no update in a gap frame


def test_run_cv_without_gap_has_every_measurement():
    run = sims.run_cv(scenarios.tracking(gap=None), 0.01, 4.0)
    assert None not in run["z"] and None not in run["K"]


def test_nis_summary_and_gap_handling():
    r = sims.nis(seed=1)
    assert len(r["honest"]) == len(r["overconfident"]) == len(r["q_small"]) == 400
    assert r["bound"] == pytest.approx(3.84) and r["maneuver"] == 200
    assert r["overconfident_mean"] > 3 * r["honest_mean"]
    # with a gap, the missing frames are simply not counted
    res = sims.SIMS["14-nis"](sims.Options(gap=(20, 27)))
    assert res.summary["n"] == 392


def test_model_truth_follows_the_filters_own_model():
    scn = scenarios.model_truth(seed=3, n=20000, q=0.01)
    assert set(scn) == set(scenarios.tracking()) | {"q"} and scn["maneuver"] is None and scn["gap"] is None
    v = np.array(scn["v"])
    assert np.std(np.diff(v)) == pytest.approx(np.sqrt(0.01), rel=0.03)         # Var(w_v) = q dt
    p = np.array(scn["p"])
    assert np.std(p[1:] - p[:-1] - v[:-1]) == pytest.approx(np.sqrt(0.01 / 3), rel=0.03)   # Var(w_p) = q dt^3/3
    assert json.loads(json.dumps(scn)) == scn


def test_model_truth_maneuver_is_a_pure_velocity_step_on_the_same_draws():
    calm, jump = scenarios.model_truth(seed=5, n=100), scenarios.model_truth(seed=5, n=100, maneuver=60, dv=4.0)
    assert calm["noise"] == jump["noise"]
    dv = np.array(jump["v"]) - np.array(calm["v"])
    np.testing.assert_allclose(dv[:59], 0), np.testing.assert_allclose(dv[59:], 4.0)
    dp = np.array(jump["p"]) - np.array(calm["p"])
    np.testing.assert_allclose(dp[60:], 4.0 * np.arange(1, 41))                 # F carries the new speed from frame 61 on


SEEDS = range(20)


def test_nis_honest_filter_is_honest_for_every_seed():
    means = [sims.nis(seed=s)["honest_mean"] for s in SEEDS]
    # the mean of 400 chi-square(1) draws has sd sqrt(2/400) = 0.07, so 0.85..1.15 is a 2 sigma band: all of these seeds sit in it
    assert all(0.85 < m < 1.15 for m in means), means
    assert 0.97 < np.mean(means) < 1.03                                          # pooled over 8000 frames: sd 0.016


def test_nis_overconfident_filter_is_flagged_for_every_seed():
    means = [sims.nis(seed=s)["overconfident_mean"] for s in SEEDS]
    assert all(m > 3 for m in means), means


def test_nis_q_too_small_spikes_only_around_the_maneuver():
    for s in SEEDS:
        r = sims.nis(seed=s)
        assert r["q_small_inside_mean"] > 3 and r["q_small_inside_mean"] > 3 * r["q_small_outside_mean"], s
        assert 0.7 < r["q_small_outside_mean"] < 1.3, s                          # away from the manoeuvre it is honest
        assert 0.85 < r["honest_mean"] < 1.15                                    # and the manoeuvre-free truth never spikes
    lo, hi = r["window"]
    k, calm, jump = np.array(r["k"]), np.array(r["honest"]), np.array(r["q_small"])
    before = k < r["maneuver"]
    np.testing.assert_allclose(calm[before], jump[before])                       # identical until the step
    assert lo == r["maneuver"] and hi == r["maneuver"] + 19


def test_options_default_to_none_and_overrides_are_used():
    o = sims.Options()
    assert (o.q, o.r, o.gap, o.seed, o.sigma_r, o.sigma_theta, o.lang) == (None,) * 7
    a = sims.SIMS["12-velocity-learning"](sims.Options()).summary
    b = sims.SIMS["12-velocity-learning"](sims.Options(q=1.0)).summary
    assert a["std_v_10_30"] < b["std_v_10_30"]


def test_sim_results_are_json_ready(tmp_path):
    for sid, fn in sims.SIMS.items():
        res = fn(sims.Options())
        assert res.table and isinstance(res.table[0], dict), sid
        json.dumps(res.table); json.dumps(res.summary)               # no numpy types left in the output


def test_png_is_1600_by_900_and_named_after_the_sim(tmp_path):
    for sid, fn in sims.SIMS.items():
        png = fn(sims.Options()).plot(tmp_path, "en")
        assert png.name == f"{sid}.png"
        w, h = struct.unpack(">II", png.read_bytes()[16:24])
        assert (w, h) == (1600, 900), sid


def test_ids_are_the_eighteen_in_the_briefs_lesson_one_first():
    assert list(sims.SIMS) == ["01-fusion", "03-1d-loop", "05-derivative-vs-filter", "08-covariance",
                               "12-velocity-learning", "13-q-r-tradeoff", "14-nis",
                               "ekf-02-tangent", "ekf-03-range-only", "ekf-05-radar-update", "ekf-07-flyby",
                               "ekf-08-wrap", "ekf-09-consistency",
                               "ukf-02-icd-decode", "ukf-03-square", "ukf-05-sigma-points", "ukf-07-radar-update",
                               "ukf-08-wide-belief"]
    assert list(sims.SIMS)[7:13] == list(sims2.SIMS) and all(sims.SIMS[k] is sims2.SIMS[k] for k in sims2.SIMS)
    assert list(sims.SIMS)[13:] == list(sims3.SIMS) and all(sims.SIMS[k] is sims3.SIMS[k] for k in sims3.SIMS)
    assert list(plot.PLOTS) == list(sims.SIMS)                       # one figure per simulation


def test_i18n_has_both_languages_for_every_key_and_rejects_unknown():
    for key in i18n.KEYS:
        assert i18n.t(key, "en") and i18n.t(key, "tr")
    assert i18n.t("frame", "tr") == "kare" and i18n.t("frame", "en") == "frame"
    with pytest.raises(KeyError):
        i18n.t("no-such-key", "en")
    with pytest.raises(KeyError):
        i18n.t("frame", "de")


def test_turkish_numbers_use_decimal_commas():
    assert plot.num(0.2, "tr", 2) == "0,20" and plot.num(0.2, "en", 2) == "0.20"
    assert plot.num(-1.5, "tr", 1) == "−1,5"
    fmt = plot.tick_formatter("tr")
    assert fmt(0.5, 0) == "0,5" and fmt(250.0, 0) == "250" and fmt(-2.0, 0) == "−2"
    assert plot.tick_formatter("en")(0.5, 0) == "0.5"


def test_derivative_std_matches_sqrt2_sigma_and_beats_the_position_noise():
    for seed in SEEDS:
        r = sims.derivative_vs_filter(seed=seed)
        assert r["sigma"] == 3.0 and r["theory_derivative"] == pytest.approx(np.sqrt(2) * 3.0)
        assert abs(r["std_derivative"] / r["theory_derivative"] - 1) < 0.15, seed
        assert r["std_derivative"] > r["std_position"], seed
        assert r["std_derivative"] > 10 * r["std_filter"], seed
    res = sims.SIMS["05-derivative-vs-filter"](sims.Options())
    assert {"std_position", "theory_derivative", "std_derivative", "std_filter"} <= set(res.summary)
    assert len(res.table) == 199


def test_maneuver_settle_means_it_stays_inside_the_band():
    for seed in (7, 0, 2):
        for r in (1.0, 4.0):
            m = sims.maneuver_response(seed=seed, r=r)
            k0, new_v = m["scn"]["maneuver"], m["new_v"]
            for name in ("small", "big"):
                v, settle = np.array(m[name]["v"]), m[name]["settle"]
                inside = np.abs(v - new_v) <= m["band"]
                if settle is None:
                    assert not inside[-1]
                else:
                    assert inside[k0 - 1 + settle:].all()                        # in the band from that frame on...
                    assert settle == 0 or not inside[k0 - 2 + settle]            # ...and not already one frame earlier
                assert m[name]["reach"] <= (settle if settle is not None else 10**6)
