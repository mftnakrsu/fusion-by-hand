import numpy as np
import pytest
from fbh import ekf, export2, sims2
from fbh.kf import cv_model

P_XY = np.array([[300.0], [400.0]])              # the prediction of sections 3 to 5: a 3-4-5 triangle, range 500
BEARING = float(np.arctan2(400.0, 300.0))        # 0.927295...

def test_section1_range_and_its_slope():
    x = np.array([[300.0]])
    assert ekf.h_range(x, 400.0).item() == pytest.approx(500.0)
    assert ekf.H_range(x, 400.0).item() == pytest.approx(0.6)
    np.testing.assert_allclose(ekf.numeric_jacobian(lambda s: ekf.h_range(s, 400.0), x), [[0.6]], atol=1e-9)

def test_section3_scalar_update_by_hand():
    # y = z - h(x), S = H P H + R, K = P H / S ; x+ = x + K y, P+ = (1 - K H) P
    h, H = lambda s: ekf.h_range(s, 400.0), lambda s: ekf.H_range(s, 400.0)
    u = ekf.update(np.array([[300.0]]), np.array([[100.0]]), np.array([[506.0]]), h, H, np.array([[4.0]]))
    assert u.y.item() == pytest.approx(6.0) and u.S.item() == pytest.approx(40.0) and u.K.item() == pytest.approx(1.5)
    assert u.x.item() == pytest.approx(309.0) and u.P.item() == pytest.approx(10.0)
    exact = sims2.range_only_update()                                  # Bayes' rule on a fine grid, same numbers
    assert (exact["x_plus"], exact["P_plus"]) == (pytest.approx(309.0), pytest.approx(10.0))
    assert round(exact["exact_mean"], 3) == 308.914 and round(exact["exact_var"], 3) == 9.691

def test_section3_curve_against_tangent_at_the_answer():
    assert round(ekf.h_range(np.array([[309.0]]), 400.0).item(), 3) == 505.451
    assert 500.0 + 0.6 * (309.0 - 300.0) == pytest.approx(505.4)

def test_radar_jacobian_by_hand_and_by_differences():
    H = ekf.H_radar(P_XY)
    np.testing.assert_allclose(H, [[0.6, 0.8], [-0.0016, 0.0012]], atol=1e-12)
    np.testing.assert_allclose(ekf.numeric_jacobian(ekf.h_radar, P_XY), H, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(ekf.h_radar(P_XY).ravel(), [500.0, BEARING])
    step = (ekf.h_radar(P_XY + np.array([[1.0], [0.0]])) - ekf.h_radar(P_XY)).ravel()   # one metre east
    assert round(step[0], 4) == 0.6006 and round(step[1], 6) == -0.001598

def test_radar_jacobian_of_the_full_state_ignores_velocity():
    x = np.array([[300.0], [400.0], [20.0], [0.0]])
    H = ekf.H_radar(x)
    assert H.shape == (2, 4)
    np.testing.assert_allclose(H[:, 2:], 0.0)
    np.testing.assert_allclose(ekf.numeric_jacobian(ekf.h_radar, x), H, rtol=1e-6, atol=1e-10)

def test_section5_radar_update_by_hand():
    P = 100.0 * np.eye(2)
    R = ekf.radar_R(5.0, 0.02)
    np.testing.assert_allclose(R, np.diag([25.0, 0.0004]))
    z = np.array([[510.0], [0.927295 + 0.02]])                          # the essay's rounded bearing, 0.02 rad left
    u = ekf.update(P_XY, P, z, ekf.h_radar, ekf.H_radar, R, ekf.radar_residual)
    np.testing.assert_allclose(u.S, np.diag([125.0, 0.0008]), atol=1e-12)
    np.testing.assert_allclose(u.K, [[0.48, -200.0], [0.64, 150.0]], rtol=1e-9)
    np.testing.assert_allclose(u.x.ravel(), [300.8, 409.4], atol=1e-3)
    np.testing.assert_allclose(u.P, [[39.2, -14.4], [-14.4, 30.8]], rtol=1e-9)
    assert u.nis == pytest.approx(1.3, abs=1e-4)
    along = P_XY.ravel() / 500.0                                        # unit vector of the line of sight
    across = np.array([-along[1], along[0]])
    assert along @ u.P @ along == pytest.approx(20.0) and across @ u.P @ across == pytest.approx(50.0)
    d = sims2.radar_update()                                            # the same update in the simulation
    np.testing.assert_allclose(d["x_plus"], [300.8, 409.4], rtol=1e-12)
    assert (d["var_along"], d["var_across"], d["nis"]) == (pytest.approx(20.0), pytest.approx(50.0), pytest.approx(1.3))

def test_section6_heading_model_jacobian_by_hand_and_by_differences():
    x = np.array([[0.0], [0.0], [np.arctan2(4.0, 3.0)], [20.0]])        # heading 53.13 degrees, 20 m/s
    F = ekf.F_heading(x, 1.0)
    np.testing.assert_allclose(F, [[1, 0, -16, 0.6], [0, 1, 12, 0.8], [0, 0, 1, 0], [0, 0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose((ekf.f_heading(x, 1.0) - x).ravel(), [12.0, 16.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ekf.numeric_jacobian(lambda s: ekf.f_heading(s, 1.0), x), F, atol=1e-8)

def test_section7_constant_velocity_is_lesson_one_once_per_axis():
    for dt, q in ((1.0, 0.1), (0.5, 2.0)):
        F, Q = ekf.cv2d_model(dt, q)
        F1, _, Q1, _ = cv_model(dt, q, 1.0)
        for axis in (0, 1):
            np.testing.assert_allclose(F[np.ix_([axis, axis + 2], [axis, axis + 2])], F1)
            np.testing.assert_allclose(Q[np.ix_([axis, axis + 2], [axis, axis + 2])], Q1)
        np.testing.assert_allclose(F[np.ix_([0, 2], [1, 3])], 0.0)      # the two axes do not mix
        np.testing.assert_allclose(Q[np.ix_([0, 2], [1, 3])], 0.0)

def test_section8_wrap():
    assert round(ekf.wrap(-6.2545), 4) == 0.0287
    assert ekf.wrap(np.pi) == pytest.approx(-np.pi) and ekf.wrap(-np.pi) == pytest.approx(-np.pi)
    assert ekf.wrap(0.3) == pytest.approx(0.3) and ekf.wrap(7.0) == pytest.approx(7.0 - 2 * np.pi)
    y = ekf.radar_residual(np.array([[100.0], [-3.1]]), np.array([[50.0], [3.1]]))
    np.testing.assert_allclose(y.ravel(), [50.0, 2 * np.pi - 6.2])     # only the bearing is wrapped

def test_section9_first_estimate_from_one_report():
    p, P = ekf.radar_to_xy(np.array([[500.0], [0.927295]]), ekf.radar_R(5.0, 0.02))
    np.testing.assert_allclose(p.ravel(), [300.0, 400.0], atol=1e-3)
    np.testing.assert_allclose(P, [[73.0, -36.0], [-36.0, 52.0]], atol=1e-3)

def test_section9_vague_update_leaves_the_range_circle():
    d = sims2.radar_update(prior_sigma=300.0, dr=0.0, dtheta=0.4)
    np.testing.assert_allclose(d["x_plus"], [140.2, 519.9], atol=0.05)
    assert round(d["range_plus"], 1) == 538.4                          # the radar read 500 m
    assert round(d["var_along"], 1) == 25.0 and round(d["var_across"], 1) == 99.9
    e = export2.exact_radar_update((300.0, 400.0), 300.0, 0.0, 0.4, 5.0, 0.02)
    np.testing.assert_allclose(e["mean"], [120.7, 485.1], atol=0.05)
    assert round(e["range_of_mean"], 1) == 499.9                       # on the circle
