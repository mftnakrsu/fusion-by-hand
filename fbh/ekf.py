"""The extended Kalman filter of lesson 2, line for line as in the essay's section 7."""
import numpy as np

from .kf import UpdateResult

def predict(x, P, f, F_jac, Q):
    F = F_jac(x)                                      # slope of the motion at the estimate we hold
    return f(x), F @ P @ F.T + Q                      # x- = f(x+) ; P- = F P+ F^T + Q

def update(x, P, z, h, H_jac, R, residual=np.subtract) -> UpdateResult:
    H = H_jac(x)                                      # slope of the measurement at the prediction
    y = residual(z, h(x))                             # the curve itself, not H x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    I = np.eye(P.shape[0])
    return UpdateResult(x + K @ y, (I - K @ H) @ P, y, S, K, (y.T @ np.linalg.inv(S) @ y).item())

# --- the radar of the lesson: range and bearing of a target at (px, py) ---------------

def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi      # any angle into [-pi, pi)

def h_radar(x):
    px, py = x[0, 0], x[1, 0]
    return np.array([[np.hypot(px, py)], [np.arctan2(py, px)]])

def H_radar(x):
    px, py = x[0, 0], x[1, 0]
    r2 = px**2 + py**2
    r = np.sqrt(r2)
    H = np.zeros((2, x.shape[0]))                     # the radar does not see the velocity
    H[0, :2] = px / r, py / r                         # range: along the line of sight
    H[1, :2] = -py / r2, px / r2                      # bearing: across it, divided by r
    return H

def radar_residual(z, z_hat):
    y = z - z_hat
    y[1, 0] = wrap(y[1, 0])                           # 359 degrees off is 1 degree off
    return y

# --- models and helpers ---------------------------------------------------------------

def cv2d_model(dt: float, q: float):
    """Constant velocity in the plane, state [px, py, vx, vy]: the F and Q of lesson 1, once per axis."""
    I2, Z2 = np.eye(2), np.zeros((2, 2))
    F = np.block([[I2, dt * I2], [Z2, I2]])
    Q = q * np.block([[dt**3 / 3 * I2, dt**2 / 2 * I2], [dt**2 / 2 * I2, dt * I2]])
    return F, Q

def radar_R(sigma_r: float, sigma_theta: float):
    return np.diag([sigma_r**2, sigma_theta**2])

def h_range(x, offset: float):
    """Section 1: the range to a target at px on a line that passes the radar at `offset`."""
    return np.array([[np.hypot(x[0, 0], offset)]])

def H_range(x, offset: float):
    return np.array([[x[0, 0] / np.hypot(x[0, 0], offset)]])

def f_heading(x, dt: float):
    """Section 6: a vehicle that holds its heading and speed, state [px, py, heading, speed]."""
    px, py, psi, s = x[:, 0]
    return np.array([[px + s * np.cos(psi) * dt], [py + s * np.sin(psi) * dt], [psi], [s]])

def F_heading(x, dt: float):
    _, _, psi, s = x[:, 0]
    return np.array([[1.0, 0.0, -s * np.sin(psi) * dt, np.cos(psi) * dt],
                     [0.0, 1.0, s * np.cos(psi) * dt, np.sin(psi) * dt],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

def numeric_jacobian(fn, x, eps: float = 1e-6):
    """The Jacobian of fn at x by central differences: the check for a Jacobian written by hand."""
    cols = []
    for i in range(x.shape[0]):
        d = np.zeros_like(x, dtype=float)
        d[i, 0] = eps
        cols.append((fn(x + d) - fn(x - d)) / (2 * eps))
    return np.hstack(cols)

def radar_to_xy(z, R):
    """A first estimate from one radar report: the position, and its covariance J R J^T."""
    r, th = z[0, 0], z[1, 0]
    c, s = np.cos(th), np.sin(th)
    J = np.array([[c, -r * s], [s, r * c]])           # d(px, py) / d(r, theta)
    return np.array([[r * c], [r * s]]), J @ R @ J.T
