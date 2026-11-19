import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from fbh import export2, sims2
from fbh.cli import main


ROOT = Path(__file__).resolve().parent.parent
FLYBY_KEYS = ["n", "dt", "seed", "q", "sigma_r", "sigma_theta", "bound", "p", "v", "x0", "P0", "raw", "z", "z_hat",
              "xm", "Pm", "x", "P", "y", "S", "K", "nis", "error", "claimed", "sigma_along", "sigma_across", "summary"]
WRAP_KEYS = ["n", "p", "v", "x0", "P0", "z", "jump_scan", "jump_scans", "raw", "wrapped", "with_wrap", "without_wrap",
             "summary"]
RUN_KEYS = ["z_hat", "xm", "x", "P", "y", "nis", "error"]
CASE_KEYS = {
    "tangent": ["x_hat", "sigma", "offset", "h", "H", "lin_std", "true_mean", "true_std"],
    "range": ["x_minus", "P_minus", "z", "R", "offset", "h", "H", "y", "S", "K", "x_plus", "P_plus", "exact_mean",
              "exact_var"],
    "update": ["x", "prior_sigma", "dr", "dtheta", "sigma_r", "sigma_theta", "H", "z_hat", "z", "y", "S", "K", "x_plus",
               "P_plus", "nis", "var_along", "var_across", "range_plus"],
    "exact": ["x", "prior_sigma", "dr", "dtheta", "sigma_r", "sigma_theta", "n", "span", "mean", "cov", "range_of_mean"],
}
INPUTS = ["x", "prior_sigma", "dr", "dtheta", "sigma_r", "sigma_theta"]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("export2")
    export2.write_all(d)
    return d


def load(out, name):
    def no_constants(token):                      # NaN / Infinity are not JSON
        raise ValueError(token)
    return json.loads((out / name).read_text(), parse_constant=no_constants)


def close(a, b):
    """Equal to the 10 significant digits the files keep."""
    np.testing.assert_allclose(np.array(a, dtype=float), np.array(b, dtype=float), rtol=1e-9, atol=1e-12)


def test_export_writes_the_three_files_compactly(out):
    assert sorted(p.name for p in out.iterdir()) == ["ekf-cases.json", "flyby.json", "wrap.json"]
    for p in out.iterdir():
        text = p.read_text()
        assert ", " not in text and '": ' not in text, p.name        # compact separators
    assert sum(p.stat().st_size for p in out.iterdir()) < 300_000


def test_flyby_json_keys_shapes_and_the_run_it_came_from(out):
    f = load(out, "flyby.json")
    assert list(f) == FLYBY_KEYS
    assert (f["n"], f["dt"], f["seed"], f["q"], f["sigma_r"], f["sigma_theta"], f["bound"]) == (60, 1, 4, 0.1, 5, 0.02, 5.99)
    shapes = {"p": (60, 2), "v": (2,), "x0": (4,), "P0": (4, 4), "raw": (60, 2), "z": (60, 2), "z_hat": (60, 2),
              "xm": (60, 4), "Pm": (60, 4, 4), "x": (60, 4), "P": (60, 4, 4), "y": (60, 2), "S": (60, 2, 2),
              "K": (60, 4, 2), "nis": (60,), "error": (60,), "claimed": (60,), "sigma_along": (60,), "sigma_across": (60,)}
    for key, shape in shapes.items():
        assert np.shape(f[key]) == shape, key
    d = sims2.flyby()
    for key in ("p", "v", "x0", "P0"):
        close(f[key], d["scn"][key])
    for key in ("z", "z_hat", "xm", "Pm", "x", "P", "y", "S", "K", "nis"):
        close(f[key], d["run"][key])
    for key in ("raw", "error", "claimed", "sigma_along", "sigma_across"):
        close(f[key], d[key])
    assert list(f["summary"]) == ["q", "sigma_r", "sigma_theta", "rms_raw", "rms_filter", "error_45", "claimed_45",
                                  "v_last", "nis_mean", "nis_above"]
    for key, value in sims2.SIMS["ekf-07-flyby"](sims2.Options()).summary.items():
        close(f["summary"][key], value)
    z = np.array(f["z"])
    assert (z[:, 1] >= -np.pi).all() and (z[:, 1] < np.pi).all()


def test_wrap_json_keys_shapes_and_the_runs_it_came_from(out):
    w = load(out, "wrap.json")
    assert list(w) == WRAP_KEYS
    assert w["n"] == 30 and w["jump_scan"] == 10 and w["jump_scans"] == [10]
    assert round(w["raw"], 4) == -6.2545 and round(w["wrapped"], 4) == 0.0287
    d = sims2.wrap_demo()
    for key in ("p", "v", "x0", "P0"):
        close(w[key], d["scn"][key])
    close(w["z"], d["wrapped_run"]["z"])
    for name, run, error in (("with_wrap", d["wrapped_run"], d["error_wrapped"]),
                             ("without_wrap", d["plain_run"], d["error_plain"])):
        assert list(w[name]) == RUN_KEYS
        for key in RUN_KEYS[:-1]:
            close(w[name][key], run[key])
        close(w[name]["error"], error)
        assert np.shape(w[name]["x"]) == (30, 4) and np.shape(w[name]["P"]) == (30, 4, 4), name
    assert list(w["summary"]) == ["jump_scan", "z_theta", "z_hat_theta", "raw", "wrapped", "max_error_wrapped",
                                  "max_error_plain"]
    close(w["summary"]["max_error_plain"], max(w["without_wrap"]["error"]))
    assert w["summary"]["jump_scan"] == 10


def test_ekf_cases_cover_the_grids_in_order(out):
    c = load(out, "ekf-cases.json")
    assert list(c) == ["tangent", "range", "update", "exact"]
    assert [len(c[name]) for name in c] == [15, 16, 36, 36]
    for name, keys in CASE_KEYS.items():
        assert all(list(case) == keys for case in c[name]), name
    assert [(t["x_hat"], t["sigma"]) for t in c["tangent"]] == [(x, s) for x in (300, 100, 30, 0, -300) for s in (10, 50, 100)]
    g = export2.UPDATE_GRID
    expected = [(list(x), s, dr, dt, sr, st) for x in g["x"] for s in g["prior_sigma"] for dr, dt in g["reading"]
                for sr, st in g["sigmas"]]
    for name in ("update", "exact"):                                   # the same inputs, case for case
        assert [tuple(case[key] for key in INPUTS) for case in c[name]] == expected, name


def test_the_hand_case_comes_first(out):
    hand = load(out, "ekf-cases.json")["update"][0]
    assert [hand[key] for key in INPUTS] == [[300, 400], 10, 10, 0.02, 5, 0.02]
    close(hand["x_plus"], [300.8, 409.4])
    close(hand["K"], [[0.48, -200], [0.64, 150]])
    assert round(hand["var_along"], 9) == 20 and round(hand["var_across"], 9) == 50 and round(hand["nis"], 9) == 1.3


def test_every_case_is_what_the_python_says(out):
    c = load(out, "ekf-cases.json")
    for case in c["tangent"]:
        d = sims2.tangent(case["x_hat"], case["sigma"])
        for key in CASE_KEYS["tangent"]:
            close(case[key], d[key])
    for case in c["range"]:                                            # z is the curve plus 6 or -4 m, rounded
        dz = round(case["z"] - math.hypot(case["x_minus"], case["offset"]), 6)
        assert dz in (6, -4)
        d = sims2.range_only_update(x=case["x_minus"], P=case["P_minus"], z=math.hypot(case["x_minus"], 400.0) + dz,
                                    sigma_r=math.sqrt(case["R"]))
        for key in CASE_KEYS["range"]:
            close(case[key], d[key])
    for upd, ex in zip(c["update"], c["exact"]):
        given = [upd[key] for key in INPUTS]
        d = sims2.radar_update(*given)
        for key in CASE_KEYS["update"][6:]:
            close(upd[key], d[key])
        e = export2.exact_radar_update(*given)
        for key in CASE_KEYS["exact"][6:]:
            close(ex[key], e[key])


def test_cli_export_lesson_two_command(tmp_path, capsys):
    assert main(["export", str(tmp_path / "two"), "--lesson", "2"]) in (0, None)
    assert sorted(p.name for p in (tmp_path / "two").iterdir()) == ["ekf-cases.json", "flyby.json", "wrap.json"]
    printed = capsys.readouterr().out
    assert all(name in printed for name in ("flyby.json", "wrap.json", "ekf-cases.json")) and "tracking.json" not in printed


# --- the README's "Export format", lesson 2, is enough to write a port --------------------

def readme_lesson_two():
    text = (ROOT / "README.md").read_text()
    section = re.search(r"^## Export format\n(.*?)(?=^## |\Z)", text, re.M | re.S).group(1)
    return section.split("### Lesson 2\n")[1]


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi                    # (a + pi) mod 2 pi - pi, into [-pi, pi)


def port_update(x, P, z, R, wrap=True):
    """The update exactly as README "Export format" words it, for a state whose first two entries are px, py."""
    n = len(x)
    px, py = x[0], x[1]
    r = math.hypot(px, py)
    z_hat = np.array([r, math.atan2(py, px)])
    H = np.zeros((2, n))
    H[:, :2] = [[px / r, py / r], [-py / r**2, px / r**2]]
    y = np.array(z) - z_hat
    if wrap:
        y[1] = wrap_angle(y[1])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return {"z_hat": z_hat, "H": H, "y": y, "S": S, "K": K, "x": x + K @ y, "P": (np.eye(n) - K @ H) @ P,
            "nis": y @ np.linalg.inv(S) @ y}


def port_tracker(x0, P0, zs, q, sigma_r, sigma_theta, wrap=True):
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    Q = q * np.array([[1 / 3, 0, 1 / 2, 0], [0, 1 / 3, 0, 1 / 2], [1 / 2, 0, 1, 0], [0, 1 / 2, 0, 1]])
    R = np.diag([sigma_r**2, sigma_theta**2])
    x, P = np.array(x0, dtype=float), np.array(P0, dtype=float)
    out = {"xm": [], "x": [], "P": [], "nis": []}
    for z in zs:                                                       # scan 1 included: predict, then update
        x, P = F @ x, F @ P @ F.T + Q
        out["xm"].append(x)
        u = port_update(x, P, z, R, wrap)
        x, P = u["x"], u["P"]
        out["x"].append(x), out["P"].append(P), out["nis"].append(u["nis"])
    return out


def test_a_port_written_only_from_the_readme_reproduces_both_tracks(out):
    f, w = load(out, "flyby.json"), load(out, "wrap.json")
    mine = port_tracker(f["x0"], f["P0"], f["z"], f["q"], f["sigma_r"], f["sigma_theta"])
    for key in ("xm", "x", "P", "nis"):                                # the README's tolerance: 1e-6 rel and abs
        np.testing.assert_allclose(mine[key], f[key], rtol=1e-6, atol=1e-6, err_msg=key)
    for name, wrap in (("with_wrap", True), ("without_wrap", False)):   # same q and sigmas as the fly-by
        mine = port_tracker(w["x0"], w["P0"], w["z"], f["q"], f["sigma_r"], f["sigma_theta"], wrap)
        for key in ("xm", "x", "P", "nis"):
            np.testing.assert_allclose(mine[key], w[name][key], rtol=1e-6, atol=1e-6, err_msg=f"{name} {key}")


def test_a_port_written_only_from_the_readme_reproduces_every_case(out):
    c = load(out, "ekf-cases.json")
    for case in c["tangent"]:
        h = math.hypot(case["x_hat"], case["offset"])
        assert (case["h"], case["H"], case["lin_std"]) == pytest.approx((h, case["x_hat"] / h, abs(case["x_hat"] / h) * case["sigma"]))
        u = np.linspace(-8, 8, 160001)                                 # integrate over +-8 sigma
        weight = np.exp(-0.5 * u**2)
        r = np.hypot(case["x_hat"] + case["sigma"] * u, case["offset"])
        mean = weight @ r / weight.sum()
        assert case["true_mean"] == pytest.approx(mean, rel=1e-9)
        assert case["true_std"] == pytest.approx(math.sqrt(weight @ (r - mean) ** 2 / weight.sum()), rel=1e-6)
    for case in c["range"]:
        h = math.hypot(case["x_minus"], case["offset"])
        H = case["x_minus"] / h
        y, S = case["z"] - h, H**2 * case["P_minus"] + case["R"]
        K = case["P_minus"] * H / S
        assert [case[key] for key in ("h", "H", "y", "S", "K", "x_plus", "P_plus")] == pytest.approx(
            [h, H, y, S, K, case["x_minus"] + K * y, (1 - K * H) * case["P_minus"]], rel=1e-6, abs=1e-6)
        g = np.linspace(-8, 8, 200001) * math.sqrt(case["P_minus"]) + case["x_minus"]   # Bayes' rule, +-8 sqrt(P)
        weight = np.exp(-0.5 * (g - case["x_minus"]) ** 2 / case["P_minus"]
                        - 0.5 * (case["z"] - np.hypot(g, case["offset"])) ** 2 / case["R"])
        mean = weight @ g / weight.sum()
        assert (case["exact_mean"], case["exact_var"]) == pytest.approx(
            (mean, weight @ (g - mean) ** 2 / weight.sum()), rel=1e-6, abs=1e-6)
    for upd, ex in zip(c["update"], c["exact"]):
        x, P = np.array(upd["x"], dtype=float), upd["prior_sigma"] ** 2 * np.eye(2)
        z = np.array([math.hypot(*x) + upd["dr"], math.atan2(x[1], x[0]) + upd["dtheta"]])   # z = z_hat + [dr, dtheta]
        mine = port_update(x, P, z, np.diag([upd["sigma_r"] ** 2, upd["sigma_theta"] ** 2]))
        u = x / np.linalg.norm(x)
        t = np.array([-u[1], u[0]])
        for key, value in (("H", mine["H"]), ("z_hat", mine["z_hat"]), ("z", z), ("y", mine["y"]), ("S", mine["S"]),
                           ("K", mine["K"]), ("x_plus", mine["x"]), ("P_plus", mine["P"]), ("nis", mine["nis"]),
                           ("var_along", u @ mine["P"] @ u), ("var_across", t @ mine["P"] @ t),
                           ("range_plus", np.linalg.norm(mine["x"]))):
            np.testing.assert_allclose(upd[key], value, rtol=1e-8, atol=1e-10, err_msg=key)
        a = np.linspace(-ex["span"], ex["span"], ex["n"])              # Bayes' rule on the polar grid
        rr, bb = np.meshgrid(z[0] + upd["sigma_r"] * a, z[1] + upd["sigma_theta"] * a, indexing="ij")
        A, B = np.meshgrid(a, a, indexing="ij")
        p = np.stack([rr * np.cos(bb), rr * np.sin(bb)], axis=-1)
        weight = np.exp(-np.sum((p - x) ** 2, axis=-1) / (2 * upd["prior_sigma"] ** 2) - A**2 / 2 - B**2 / 2) * rr
        weight = np.where(rr > 0, weight, 0.0) / np.where(rr > 0, weight, 0.0).sum()
        mean = np.einsum("ij,ijk->k", weight, p)
        e = p - mean
        cov = np.einsum("ij,ijk,ijl->kl", weight, e, e)
        np.testing.assert_allclose(ex["mean"], mean, rtol=1e-8)
        np.testing.assert_allclose(ex["cov"], cov, rtol=1e-6, atol=1e-8)
        assert ex["range_of_mean"] == pytest.approx(np.linalg.norm(mean), rel=1e-8)


def test_readme_lesson_two_part_names_every_file_and_key_and_the_semantics(out):
    section = readme_lesson_two()
    for name in ("flyby.json", "wrap.json", "ekf-cases.json"):
        assert f"`{name}`" in section, name
        top = load(out, name)
        for key, value in top.items():
            assert f"`{key}`" in section, (name, key)
            if isinstance(value, dict):                                # summary, with_wrap, without_wrap
                for inner in value:
                    assert f"`{inner}`" in section, (name, key, inner)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                for inner in value[0]:                                 # the keys of each kind of case
                    assert f"`{inner}`" in section, (name, key, inner)
    for phrase in ("metres", "radians", "[-pi, pi)", "atan2(py, px)", "1-based", "0-based", "scan - 1", "scan 1 included",
                   "x0", "P0", "wrapped into [-pi, pi)", "(I - K H) P", "R = diag(sigma_r², sigma_theta²)",
                   "exact[i]", "update[i]", "the same q, sigma_r and sigma_theta", "plain difference",
                   "null", "scans 10 to 60", "4 × 2", "2 × 2", "1e-6 relative and absolute"):
        assert phrase in section, phrase


def test_module_docstring_states_the_same_semantics():
    doc = export2.__doc__
    for phrase in ("metres", "radians", "1-based", "0-based", "scan - 1", "[-pi, pi)", "flyby.json", "wrap.json",
                   "ekf-cases.json", "Python is the single source"):
        assert phrase in doc, phrase
