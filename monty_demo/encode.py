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
    effort: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate per-frame ``k_hat`` in [0, 1].

    Two signal sources, fused when both are available:

    * **Kinematic compliance proxy** (always computed): when commanded motion
      was resisted (commanded ≫ observed), apparent stiffness is high. Pure
      kinematics — works on any embodiment.

    * **Joint-torque proxy** (used when ``effort`` is provided — ALOHA via
      ``observation.effort``, e.g.): per-frame torque magnitude robustly
      normalized to the episode's 95th-percentile reference. Real physical
      signal, not a kinematic inference.

    When effort is provided, the final ``k_hat`` is a 60/40 weighted fusion
    (effort dominant; kinematic stays as a safety net for noisy effort
    readings). When absent, returns the kinematic proxy alone — same behavior
    as before this upgrade.

    Vectorized; target < 5 ms for 250 frames × 14 DOF either way.
    """
    if positions.shape != actions.shape:
        raise ValueError(
            f"shape mismatch: positions={positions.shape} vs actions={actions.shape}"
        )
    T = positions.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float32)

    def _smooth01(raw: np.ndarray) -> np.ndarray:
        return np.clip(_ema(raw.astype(np.float32), alpha=0.3), 0.0, 1.0)

    # --- Kinematic proxy (always) ---
    commanded_motion = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    observed_motion = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    raw = (commanded_motion - observed_motion) / (commanded_motion + 1e-4)
    k_kinematic = _smooth01(np.concatenate([[raw[0]], raw]))

    if effort is None or effort.size == 0 or effort.shape != positions.shape:
        return k_kinematic.astype(np.float32)

    # --- Effort proxy: robust 95th-percentile normalization (a single
    # current-sensing spike shouldn't compress the rest of the trace to 0..0.1) ---
    eff_mag = np.linalg.norm(effort, axis=1)
    eff_ref = float(np.percentile(eff_mag, 95)) + 1e-6
    k_effort = _smooth01(eff_mag / eff_ref)

    # --- Fusion: effort-dominant, kinematic as safety net ---
    return np.clip(0.6 * k_effort + 0.4 * k_kinematic, 0.0, 1.0).astype(np.float32)
