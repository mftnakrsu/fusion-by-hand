import numpy as np
import pytest
from fbh import ekf, sims2, sims3, ukf

R3 = np.sqrt(3.0)                                # n + kappa = 3: every offset is sqrt(3) sigma

def test_kappa_makes_n_plus_kappa_three():
    assert (ukf.kappa(1), ukf.kappa(2)) == (2.0, 1.0)
    for n in (1, 2):
        _, w = ukf.sigma_points(np.zeros((n, 1)), np.eye(n))
        assert w.sum() == pytest.approx(1.0) and len(w) == 2 * n + 1

# --- section 2: the ICD ---------------------------------------------------------------------

def test_section2_the_message_is_a_header_and_a_payload():
    msg = sims3.MESSAGE
    assert len(msg) == 32 and msg[:8] == bytes.fromhex("a55a100100181267")
    assert msg[8:] == bytes.fromhex("00 05 08 49 11 57 52 40 1a 37 f5 c5 08 0a e9 0e 00 7b ff eb 01 f4 01 0f")

def test_section2_decode_by_hand():
    v = {r["field"]: r for r in sims3.decode()}
    assert [r["bytes"] for r in v.values()] == ["0-1", "2", "3", "4-5", "6-7", "8-15", "16-19", "20-23", "24-25",
                                                "26-27", "28-29", "30", "31"]
    assert v["sync"]["raw"] == 0xA55A and v["msg_id"]["raw"] == 0x10 and v["msg_id"]["value"] == "0x10 LOS"
    assert v["version"]["raw"] == 1 and v["length"]["raw"] == 24 == 32 - 8 and v["seq"]["raw"] == 4711
    assert v["t_meas"]["raw"] == 1416484800123456
    assert v["az"]["raw"] == 439875013 and v["el"]["raw"] == 134932750
    assert round(v["az"]["value"], 5) == 0.64350 and round(np.degrees(v["az"]["value"]), 2) == 36.87
    assert round(v["el"]["value"], 5) == 0.19740 and round(np.degrees(v["el"]["value"]), 2) == 11.31
    assert v["az_rate"]["raw"] == 123 and v["az_rate"]["value"] == pytest.approx(0.0123)
    assert v["el_rate"]["raw"] == -21 and v["el_rate"]["value"] == pytest.approx(-0.0021)   # 0xffeb, two's complement
    assert v["sigma_los"]["raw"] == 500 and v["sensor"]["value"] == "MWIR"
    assert v["valid"]["raw"] == 0x0F and v["valid"]["value"] == "0x0F"

def test_section2_angles_match_the_geometry_to_a_fraction_of_an_lsb():
    c = sims3.icd_check(sims3.decode())
    assert c["atan_300_400_deg"] == pytest.approx(np.degrees(np.arctan(300 / 400)))
    assert c["atan_100_500_deg"] == pytest.approx(np.degrees(np.arctan(100 / 500)))
    assert abs(c["az_off_lsb"]) < 0.5 and abs(c["el_off_lsb"]) < 0.5                     # rounded to the nearest count
    assert round(np.arctan(0.75) * 2**31 / np.pi) == 439875013 and round(np.arctan(0.2) * 2**31 / np.pi) == 134932750
    bits = [c[name] for name in sims3.VALID_BITS]
    assert bits == [1, 1, 1, 1, 0]                                                        # 0x0F: all but gimbal at limit

# --- section 3: y = x^2 -----------------------------------------------------------------------

def test_section3_three_points_through_the_square():
    d = sims3.square()
    np.testing.assert_allclose(d["x"], [2.0, 2 + R3, 2 - R3])
    np.testing.assert_allclose(d["w"], [2 / 3, 1 / 6, 1 / 6])
    assert (d["ekf_mean"], d["ekf_var"]) == (4.0, 16.0)                                   # the tangent: mu^2, (2 mu)^2
    assert d["ukf_mean"] == pytest.approx(5.0) and d["ukf_var"] == pytest.approx(18.0)
    assert (d["exact_mean"], d["exact_var"]) == (5.0, 18.0)                               # mu^2 + s^2, 4 mu^2 s^2 + 2 s^4
    y = d["grid"] ** 2                                                                     # and by numerical integration
    m = d["weights"] @ y
    assert m == pytest.approx(5.0, abs=1e-6) and d["weights"] @ (y - m) ** 2 == pytest.approx(18.0, abs=1e-5)

# --- section 5: sigma points ------------------------------------------------------------------

def test_section5_one_dimension():
    X, w = ukf.sigma_points(np.array([[300.0]]), np.array([[100.0]]))
    np.testing.assert_allclose(X.ravel(), [300.0, 300 + 10 * R3, 300 - 10 * R3])
    assert np.round(X.ravel(), 2).tolist() == [300.0, 317.32, 282.68]
    np.testing.assert_allclose(w, [2 / 3, 1 / 6, 1 / 6])
    d = sims3.sigma_sets()
    assert d["mean_1d"] == pytest.approx(300.0) and d["var_1d"] == pytest.approx(100.0)
    assert d["m4_1d"] == pytest.approx(3 * 10.0**4) == d["three_sigma4"]                  # a Gaussian's 3 sigma^4

def test_section5_two_dimensions_round():
    X, w = ukf.sigma_points(np.array([[300.0], [400.0]]), 100.0 * np.eye(2))
    s = 10 * R3
    np.testing.assert_allclose(X.T, [[300, 400], [300 + s, 400], [300, 400 + s], [300 - s, 400], [300, 400 - s]])
    np.testing.assert_allclose(w, [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
    assert round(s, 2) == 17.32

def test_section5_tilted_cholesky_and_axes_give_back_the_same_mean_and_P():
    d = sims3.sigma_sets()
    P = np.array([[39.2, -14.4], [-14.4, 30.8]])
    np.testing.assert_allclose(d["axis_var"], [20.0, 50.0])
    np.testing.assert_allclose(d["axis_dir"], [[0.6, 0.8], [-0.8, 0.6]], atol=1e-12)
    np.testing.assert_allclose(d["axis_offset"], [R3 * np.sqrt(20), R3 * np.sqrt(50)])
    assert np.round(d["axis_offset"], 2).tolist() == [7.75, 12.25]
    Xa = d["sets"]["axes"][0] - np.array([[300.8], [409.4]])
    np.testing.assert_allclose(np.round(Xa.T, 2), [[0, 0], [4.65, 6.2], [-9.8, 7.35], [-4.65, -6.2], [9.8, -7.35]],
                               atol=1e-9)
    L = np.linalg.cholesky(P)
    np.testing.assert_allclose(d["sets"]["cholesky"][0][:, 1:3] - np.array([[300.8], [409.4]]), R3 * L)
    for name in ("cholesky", "axes"):
        np.testing.assert_allclose(d[f"mean_{name}"], [300.8, 409.4])
        np.testing.assert_allclose(d[f"P_{name}"], P, atol=1e-9)

# --- section 7: one radar update ---------------------------------------------------------

def test_section7_ukf_range_update_by_hand():
    h = lambda s: ekf.h_range(s, 400.0)
    u = ukf.update(np.array([[300.0]]), np.array([[100.0]]), np.array([[506.0]]), h, np.array([[4.0]]))
    assert np.round(u.X.ravel(), 2).tolist() == [300.0, 317.32, 282.68]
    assert np.round(u.Z.ravel(), 2).tolist() == [500.0, 510.58, 489.8]
    assert round(u.z_hat.item(), 2) == 500.06 and round(u.spread.item(), 2) == 35.98
    assert round(u.S.item(), 2) == 39.98 and round(u.P_xz.item(), 2) == 59.98 and round(u.K.item(), 3) == 1.5
    assert round(u.y.item(), 2) == 5.94 and round(u.x.item(), 2) == 308.9 and round(u.P.item(), 2) == 10.03
    assert round(u.x.item(), 4) == 308.9049
    d = sims3.range_update()
    assert d["x_plus"] == pytest.approx(u.x.item()) and d["P_plus"] == pytest.approx(u.P.item())
    e = d["ekf"]                                                                            # lesson 2, the same numbers
    assert (e["h"], e["H"], e["S"], e["K"]) == (pytest.approx(500.0), pytest.approx(0.6), pytest.approx(40.0),
                                                pytest.approx(1.5))
    assert e["x_plus"] == pytest.approx(309.0) and e["P_plus"] == pytest.approx(10.0)
    assert round(e["exact_mean"], 2) == 308.91 and round(e["exact_var"], 2) == 9.69

def test_ukf_update_on_a_line_is_the_kalman_update():
    """With a straight h the points add nothing: the UKF gives exactly the update of lesson 1."""
    H, R = np.array([[0.6, 0.8]]), np.array([[25.0]])
    x, P, z = np.array([[300.0], [400.0]]), np.array([[100.0, 20.0], [20.0, 50.0]]), np.array([[510.0]])
    u = ukf.update(x, P, z, lambda s: H @ s, R)
    k = ekf.update(x, P, z, lambda s: H @ s, lambda s: H, R)
    np.testing.assert_allclose(u.x, k.x)
    np.testing.assert_allclose(u.P, k.P, atol=1e-10)
    assert u.nis == pytest.approx(k.nis)

# --- section 8: a wide belief ------------------------------------------------------------

def test_section8_wide_belief():
    d = sims3.range_update(P=10_000.0, z=560.0)
    e = d["ekf"]
    assert e["S"] == pytest.approx(3604.0) and round(e["K"], 3) == 1.665
    assert round(e["x_plus"], 1) == 399.9 and round(d["ekf_sigma"], 1) == 3.3
    assert np.round(d["points"], 1).tolist() == [300.0, 473.2, 126.8]
    assert np.round(d["ranges"], 1).tolist() == [500.0, 619.6, 419.6]
    assert round(d["z_hat"], 1) == 506.5 and round(d["S"]) == 3423 and round(d["P_xz"]) == 5774
    assert round(d["K"], 3) == 1.687 and round(d["x_plus"], 1) == 390.2 and round(d["sigma"], 1) == 16.2
    assert round(e["exact_mean"], 1) == 391.8 and round(e["exact_var"], 2) == 8.16 and round(d["exact_sigma"], 1) == 2.9
    assert round(d["ekf_off_sigmas"], 1) == 2.4 and round(d["ukf_off_sigmas"], 1) == 0.1
    assert d["ekf_off"] > 4 * d["ukf_off"]

def test_lesson_two_range_update_is_reused():
    e, mine = sims2.range_only_update(P=10_000.0, z=560.0, sigma_r=3.0), sims3.range_update(P=10_000.0, z=560.0, sigma_r=3.0)
    for key in ("x_plus", "P_plus", "exact_mean", "exact_var"):
        assert mine["ekf"][key] == e[key], key
