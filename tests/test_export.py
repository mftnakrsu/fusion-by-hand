import json
import re
from pathlib import Path

import numpy as np
import pytest

from fbh import export, scenarios, sims
from fbh.cli import main


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    export.write_all(d)
    return d


def load(out, name):
    def no_constants(token):                      # NaN / Infinity are not JSON
        raise ValueError(token)
    return json.loads((out / name).read_text(), parse_constant=no_constants)


def test_export_writes_the_five_files_compactly(out):
    names = sorted(p.name for p in out.iterdir())
    assert names == ["covariance.json", "derivative.json", "kf-cases.json", "nis.json", "tracking.json"]
    for p in out.iterdir():
        text = p.read_text()
        assert ", " not in text and '": ' not in text, p.name        # compact separators
    assert sum(p.stat().st_size for p in out.iterdir()) < 1_000_000


def test_tracking_json_is_the_scenario_the_brief_describes(out):
    t = load(out, "tracking.json")
    assert list(t) == ["n", "gap", "maneuver", "p", "v", "noise", "x0", "P0"]
    assert t["n"] == 60 and t["gap"] == [20, 27] and t["maneuver"] == 36
    assert t["x0"] == [200, 7] and t["P0"] == [[16, 0], [0, 9]]
    assert len(t["p"]) == len(t["v"]) == len(t["noise"]) == 60
    assert all(isinstance(x, float) for x in t["noise"])
    scn = scenarios.tracking()
    assert t["noise"] == scn["noise"] and t["p"] == scn["p"] and t["v"] == scn["v"]      # lossless


def test_kf_cases_cover_the_grid_in_order(out):
    cases = load(out, "kf-cases.json")["cases"]
    assert len(cases) == 24
    grid = [(c["q"], c["r"], c["gap"]) for c in cases]
    assert grid == [(q, r, g) for q in (0.001, 0.01, 1, 10) for r in (1, 4, 25) for g in (True, False)]
    for c in cases:
        assert list(c) == ["q", "r", "gap", "x", "P", "K"]
        assert len(c["x"]) == len(c["P"]) == len(c["K"]) == 60
        assert all(len(x) == 2 for x in c["x"])
        assert all(len(P) == 2 and len(P[0]) == len(P[1]) == 2 for P in c["P"])


def test_kf_cases_match_run_cv_to_1e_12(out):
    cases = load(out, "kf-cases.json")["cases"]
    for c in cases:
        scn = scenarios.tracking(gap=(20, 27) if c["gap"] else None)
        run = sims.run_cv(scn, c["q"], c["r"])
        np.testing.assert_allclose(c["x"], run["x"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(c["P"], run["P"], rtol=1e-12, atol=0)
        assert [k is None for k in c["K"]] == [k is None for k in run["K"]]
        np.testing.assert_allclose([k for k in c["K"] if k is not None], [k for k in run["K"] if k is not None],
                                   rtol=1e-12, atol=0)


def test_gap_cases_have_null_gain_exactly_in_the_gap(out):
    for c in load(out, "kf-cases.json")["cases"]:
        missing = [k for k, K in enumerate(c["K"], start=1) if K is None]
        assert missing == (list(range(20, 28)) if c["gap"] else [])
        assert all(len(K) == 2 for K in c["K"] if K is not None)


def test_tracking_json_alone_reproduces_every_case(out):
    """What the browser does: take tracking.json, run the filter, compare with kf-cases.json."""
    t = load(out, "tracking.json")
    for c in load(out, "kf-cases.json")["cases"]:
        scn = {**t, "gap": t["gap"] if c["gap"] else None}
        run = sims.run_cv(scn, c["q"], c["r"])
        np.testing.assert_allclose(c["x"], run["x"], rtol=1e-12, atol=0)


def test_derivative_json(out):
    d = load(out, "derivative.json")
    assert list(d) == ["k", "truth_p", "truth_v", "z", "derivative", "filter", "summary"]
    assert d["k"] == list(range(2, 201)) and d["truth_p"] == 250 and d["truth_v"] == 0
    assert len(d["z"]) == len(d["derivative"]) == len(d["filter"]) == 199
    s = d["summary"]
    assert set(s) == {"sigma", "std_position", "theory_derivative", "std_derivative", "std_filter", "ratio"}
    assert s["theory_derivative"] == pytest.approx(3 * np.sqrt(2))
    assert s["std_derivative"] == pytest.approx(np.std(d["derivative"]), rel=1e-9)
    assert s["std_filter"] == pytest.approx(np.std(d["filter"]), rel=1e-9)
    assert np.diff(d["z"]).tolist() == pytest.approx(d["derivative"][1:], abs=1e-6)


def test_covariance_json(out):
    c = load(out, "covariance.json")
    assert list(c) == ["n", "F", "before", "after", "P_before", "P_after_formula"]
    assert c["n"] == 2000 and len(c["before"]) == len(c["after"]) == 2000
    assert all(len(e) == 2 for e in c["before"] + c["after"])
    assert c["F"] == [[1, 1], [0, 1]]
    assert c["P_before"] == [[2.5, 1], [1, 0.5]]
    np.testing.assert_allclose(c["P_after_formula"], [[5, 1.5], [1.5, 0.5]])
    e = np.array(c["before"])
    np.testing.assert_allclose(np.array(c["after"]), e @ np.array(c["F"]).T, rtol=1e-8, atol=1e-9)
    after = np.array(c["after"])
    np.testing.assert_allclose(after.T @ after / 2000, c["P_after_formula"], rtol=0.15, atol=0.15)


def test_nis_json_has_the_three_series_their_means_and_the_bound(out):
    n = load(out, "nis.json")
    assert n["k"] == list(range(1, 401)) and n["bound"] == 3.84 and n["maneuver"] == 200 and n["window"] == [200, 219]
    for key in ("honest", "overconfident", "q_small"):
        assert len(n[key]) == 400 and all(v >= 0 for v in n[key])
        assert n["summary"][f"{key}_mean"] == pytest.approx(np.mean(n[key]), rel=1e-8)
    s = n["summary"]
    assert s["bound"] == 3.84
    assert 0.85 < s["honest_mean"] < 1.15 and s["overconfident_mean"] > 3
    assert s["q_small_inside_mean"] > 3 * s["q_small_outside_mean"]


def test_cli_export_command(tmp_path, capsys):
    assert main(["export", str(tmp_path / "new" / "dir")]) in (0, None)
    assert (tmp_path / "new" / "dir" / "tracking.json").exists()
    printed = capsys.readouterr().out
    assert "kf-cases.json" in printed and "tracking.json" in printed


# --- the README's "Export format" is enough to write a port -------------------------------

def readme_section(title):
    text = (ROOT / "README.md").read_text()
    return re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S).group(1)


def port_from_the_readme(t, q, r, gap):
    """The filter exactly as README "Export format" words it, in plain Python (what the browser will do)."""
    sd = r ** 0.5
    x, P = list(t["x0"]), [list(row) for row in t["P0"]]                  # the frame-0 prior
    out = {"x": [], "P": [], "K": []}
    for k in range(1, t["n"] + 1):
        x = [x[0] + x[1], x[1]]                                           # predict, in frame 1 too: x- = F x
        (a, b), (c, d) = P                                                # P- = F P F^T + Q, F = [[1,1],[0,1]]
        P = [[a + c + b + d + q / 3, b + d + q / 2], [c + d + q / 2, d + q]]
        K = None
        if gap is None or not gap[0] <= k <= gap[1]:                      # gap frames: predict only
            z = t["p"][k - 1] + sd * t["noise"][k - 1]                    # z = p + sqrt(r) noise, frames 1-based
            y, S = z - x[0], P[0][0] + r                                  # H = [1, 0], R = r
            K = [P[0][0] / S, P[1][0] / S]                                # K = P- H^T / S
            x = [x[0] + K[0] * y, x[1] + K[1] * y]
            P = [[(1 - K[0]) * P[0][0], (1 - K[0]) * P[0][1]],            # P = (I - K H) P-, not Joseph, not symmetrised
                 [P[1][0] - K[1] * P[0][0], P[1][1] - K[1] * P[0][1]]]
        out["x"].append(x), out["P"].append([list(row) for row in P]), out["K"].append(K)
    return out


def test_a_port_written_only_from_the_readme_reproduces_every_case_to_1e_9(out):
    t = load(out, "tracking.json")
    for c in load(out, "kf-cases.json")["cases"]:
        mine = port_from_the_readme(t, c["q"], c["r"], t["gap"] if c["gap"] else None)
        np.testing.assert_allclose(mine["x"], c["x"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(mine["P"], c["P"], rtol=1e-9, atol=1e-12)
        assert [k is None for k in mine["K"]] == [k is None for k in c["K"]]
        np.testing.assert_allclose([k for k in mine["K"] if k], [k for k in c["K"] if k], rtol=1e-9, atol=1e-12)


def test_readme_export_section_describes_every_file_and_key_and_the_semantics(out):
    section = readme_section("Export format")
    for name in ("tracking.json", "kf-cases.json", "derivative.json", "covariance.json", "nis.json"):
        assert f"`{name}`" in section, name
        top = load(out, name)
        for key in top:
            assert f"`{key}`" in section, (name, key)
    for key in ("x", "P", "K", "q", "r", "gap"):                                   # inside a kf-cases case
        assert f"`{key}`" in section
    for phrase in ("unit normal", "sqrt(r)", "1-based", "inclusive", "predict", "frame 1", "x0", "P0", "after the update",
                   "[[1, 1], [0, 1]]", "[1, 0]", "[[1/3, 1/2], [1/2, 1]]", "(I - K H) P", "Joseph", "symmetric", "1e-9",
                   "null"):
        assert phrase in section, phrase


def test_module_docstring_states_the_same_semantics():
    doc = export.__doc__
    for phrase in ("unit normal", "1-based", "inclusive", "frame 1", "x0", "P0", "after the update", "Joseph", "1e-9", "null"):
        assert phrase in doc, phrase
