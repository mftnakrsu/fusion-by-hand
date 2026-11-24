"""The unscented Kalman filter of lesson 3, first half: sigma points, the unscented transform and the update.

The unscaled form: 2n + 1 points, the mean and the mean plus and minus sqrt(n + kappa) times each column of a
square root L of P (L L^T = P), weighted kappa / (n + kappa) and 1 / (2 (n + kappa)). The lesson takes n + kappa = 3,
so kappa = 2 in one dimension and 1 in two.
"""
from dataclasses import dataclass

import numpy as np

from .kf import UpdateResult

def kappa(n: int) -> float:
    return 3.0 - n                                    # n + kappa = 3: the points match a Gaussian's 4th moment in 1-D

def sigma_points(x, P, k=None, sqrt=np.linalg.cholesky):
    """The sigma points of N(x, P) as the columns of X, and their weights w (they add up to 1).

    Column 0 is x, columns 1..n are x + sqrt(n + k) L_i and columns n+1..2n are x - sqrt(n + k) L_i, where L_i is
    column i of sqrt(P). Any L with L L^T = P gives a set with the same mean and covariance; Cholesky is the usual one.
    """
    n = x.shape[0]
    k = kappa(n) if k is None else k
    L = np.sqrt(n + k) * sqrt(P)
    X = np.hstack([x, x + L, x - L])
    w = np.full(2 * n + 1, 1 / (2 * (n + k)))
    w[0] = k / (n + k)
    return X, w

def unscented_transform(Y, w):
    """The weighted mean and covariance of the points in the columns of Y (the points after a function)."""
    mean = Y @ w[:, None]
    d = Y - mean
    return mean, (d * w) @ d.T

@dataclass
class UKFUpdate(UpdateResult):
    X: np.ndarray                                     # the sigma points of the prediction, one per column
    w: np.ndarray
    Z: np.ndarray                                     # the same points through h
    z_hat: np.ndarray
    spread: np.ndarray                                # the covariance of Z, before R is added
    P_xz: np.ndarray

def update(x, P, z, h, R, k=None) -> UKFUpdate:
    X, w = sigma_points(x, P, k)
    Z = np.hstack([h(X[:, [i]]) for i in range(X.shape[1])])   # each point through the curve itself: no Jacobian
    z_hat, spread = unscented_transform(Z, w)
    S = spread + R
    P_xz = ((X - x) * w) @ (Z - z_hat).T               # how state and measurement move together over the points
    K = P_xz @ np.linalg.inv(S)
    y = z - z_hat
    return UKFUpdate(x + K @ y, P - K @ S @ K.T, y, S, K, (y.T @ np.linalg.inv(S) @ y).item(),
                     X, w, Z, z_hat, spread, P_xz)
