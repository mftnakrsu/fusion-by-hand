"""The Kalman filter of lesson 1, line for line as in the essay's section 12."""
from dataclasses import dataclass
import numpy as np

def cv_model(dt: float, q: float, r: float):
    F = np.array([[1.0, dt], [0.0, 1.0]])            # constant velocity
    H = np.array([[1.0, 0.0]])                        # the camera sees position only
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])  # CWNA
    R = np.array([[float(r)]])
    return F, H, Q, R

def predict(x, P, F, Q):
    return F @ x, F @ P @ F.T + Q                     # x- = F x+ ; P- = F P+ F^T + Q

@dataclass
class UpdateResult:
    x: np.ndarray
    P: np.ndarray
    y: np.ndarray
    S: np.ndarray
    K: np.ndarray
    nis: float

def _gain(x, P, z, H, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    return y, S, K

def update(x, P, z, H, R) -> UpdateResult:
    y, S, K = _gain(x, P, z, H, R)
    I = np.eye(P.shape[0])
    return UpdateResult(x + K @ y, (I - K @ H) @ P, y, S, K, (y.T @ np.linalg.inv(S) @ y).item())

def update_joseph(x, P, z, H, R) -> UpdateResult:
    y, S, K = _gain(x, P, z, H, R)
    A = np.eye(P.shape[0]) - K @ H
    return UpdateResult(x + K @ y, A @ P @ A.T + K @ R @ K.T, y, S, K, (y.T @ np.linalg.inv(S) @ y).item())

def fuse(z1: float, var1: float, z2: float, var2: float):
    w1, w2 = 1 / var1, 1 / var2
    return (z1 * w1 + z2 * w2) / (w1 + w2), 1 / (w1 + w2)
