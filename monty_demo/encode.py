"""Impedance encoder.

``estimate_stiffness`` produces a kinematic-compliance proxy ``k_hat`` in
[0, 1]. NOT calibrated to N/m. Use F/T-instrumented data for real impedance;
this estimator exists for the common case where you only have commanded vs
observed kinematics (e.g. ALOHA, most teleop datasets).
"""

from __future__ import annotations

import numpy as np

from monty_demo._timing import timed


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential moving average. Loop is fine — T is small (~250)."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    one_minus = 1.0 - alpha
    for i in range(1, x.size):
        y[i] = alpha * x[i] + one_minus * y[i - 1]
    return y


@timed("monty_demo.estimate_stiffness")
def estimate_stiffness(
    positions: np.ndarray,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Kinematic compliance proxy.

    ``k_hat ≈ 1`` when commanded motion was resisted (high apparent stiffness).
    ``k_hat ≈ 0`` when motion was free.

    Vectorized; target < 5 ms for 250 frames × 14 DOF.
    """
    if positions.shape != actions.shape:
        raise ValueError(
            f"shape mismatch: positions={positions.shape} vs actions={actions.shape}"
        )
    T = positions.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float32)

    commanded_motion = np.linalg.norm(np.diff(actions, axis=0), axis=1)    # (T-1,)
    observed_motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)   # (T-1,)
    eps = 1e-4
    raw = (commanded_motion - observed_motion) / (commanded_motion + eps)  # (T-1,)
    raw = np.concatenate([[raw[0]], raw])                                   # (T,)
    smoothed = _ema(raw.astype(np.float32), alpha=0.3)
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)
