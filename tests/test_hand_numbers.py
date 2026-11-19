import numpy as np
import pytest
from fbh.kf import cv_model, predict, update, update_joseph, fuse

def test_section1_inverse_variance_fusion():
    m, v = fuse(20.0, 1.0, 30.0, 9.0)
    assert m == pytest.approx(21.0)
    assert v == pytest.approx(0.9)

def test_section3_one_d_loop_by_hand():
    # 1-D: x- = x+ + u, P- = P+ + Q ; K = P-/(P-+R)
    x, P, u, Q, R = 200.0, 4.0, 10.0, 5.0, 1.0
    xm, Pm = x + u, P + Q
    assert (xm, Pm) == (210.0, 9.0)
    K = Pm / (Pm + R)
    xp, Pp = xm + K * (216 - xm), (1 - K) * Pm
    assert K == pytest.approx(0.9) and xp == pytest.approx(215.4) and Pp == pytest.approx(0.9)
    assert (xp + u, Pp + Q) == (pytest.approx(225.4), pytest.approx(5.9))
    rows, x, P = [], xp + u, Pp + Q
    rows.append((x, P))
    for _ in range(2):
        x, P = x + u, P + Q
        rows.append((x, P))
    assert rows == [(pytest.approx(225.4), pytest.approx(5.9)), (pytest.approx(235.4), pytest.approx(10.9)), (pytest.approx(245.4), pytest.approx(15.9))]

def test_section8_covariance_from_trials():
    e = np.array([[-2, -1], [-1, 0], [1, 0], [2, 1]], float)
    P = e.T @ e / len(e)
    np.testing.assert_allclose(P, [[2.5, 1.0], [1.0, 0.5]])

def test_section9_propagation_two_ways():
    F, H, Q, R = cv_model(1.0, 0.0, 1.0)
    P = np.array([[2.5, 1.0], [1.0, 0.5]])
    _, Pm = predict(np.zeros((2, 1)), P, F, Q)
    np.testing.assert_allclose(Pm, [[5.0, 1.5], [1.5, 0.5]])
    e = np.array([[-2, -1], [-1, 0], [1, 0], [2, 1]], float) @ F.T
    np.testing.assert_allclose(e.T @ e / 4, Pm)
    np.testing.assert_allclose(F @ P, [[3.5, 1.5], [1.0, 0.5]])

def test_section10_update_by_hand():
    F, H, Q, R = cv_model(1.0, 0.0, 1.0)
    Pm = np.array([[5.0, 1.5], [1.5, 0.5]])
    xm = np.array([[225.4], [10.0]])
    u = update(xm, Pm, np.array([[229.0]]), H, R)
    assert u.S.item() == pytest.approx(6.0)
    np.testing.assert_allclose(u.K.ravel(), [5 / 6, 0.25])
    np.testing.assert_allclose(u.x.ravel(), [228.4, 10.9])
    np.testing.assert_allclose(u.P, [[5 / 6, 0.25], [0.25, 0.125]])
    assert u.y.item() == pytest.approx(3.6)
    assert u.nis == pytest.approx(3.6 ** 2 / 6)

def test_joseph_equals_short_form_at_optimal_gain():
    F, H, Q, R = cv_model(1.0, 0.0, 1.0)
    Pm = np.array([[5.0, 1.5], [1.5, 0.5]])
    xm = np.array([[225.4], [10.0]])
    a = update(xm, Pm, np.array([[229.0]]), H, R)
    b = update_joseph(xm, Pm, np.array([[229.0]]), H, R)
    np.testing.assert_allclose(a.P, b.P, atol=1e-12)
