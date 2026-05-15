"""Encoder + segmenter unit tests on synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from monty_demo.encode import estimate_stiffness
from monty_demo.segment import segment_phases


# --- estimate_stiffness ---------------------------------------------------


def test_stiffness_bounds_in_unit_interval():
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((250, 14)).cumsum(axis=0).astype(np.float32)
    actions = positions + rng.standard_normal((250, 14)).astype(np.float32) * 0.1
    k = estimate_stiffness(positions, actions, dt=0.02)
    assert k.shape == (250,)
    assert k.dtype == np.float32
    assert k.min() >= 0.0
    assert k.max() <= 1.0


def test_stiffness_zero_when_perfect_tracking():
    """Identical commanded and observed motion → no resistance."""
    actions = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32) * np.ones((1, 7))
    positions = actions.copy()
    k = estimate_stiffness(positions, actions, dt=0.02)
    # All motion was free; values should be very near zero
    assert k.max() < 0.05


def test_stiffness_high_when_commanded_motion_is_resisted():
    """Commanded large motion, observed nothing → high apparent stiffness."""
    T, D = 100, 7
    actions = np.linspace(0, 1, T).reshape(-1, 1).astype(np.float32) * np.ones((1, D))
    positions = np.zeros((T, D), dtype=np.float32)
    k = estimate_stiffness(positions, actions, dt=0.02)
    # Commanded motion fully resisted → k_hat → 1.0 after EMA settles
    assert k[-1] > 0.9


def test_stiffness_monotonic_in_resistance():
    """As the commanded-vs-observed gap grows, mean k_hat should rise."""
    T, D = 200, 7
    actions = np.linspace(0, 1, T).reshape(-1, 1).astype(np.float32) * np.ones((1, D))
    means = []
    for fraction in (0.1, 0.5, 0.9):
        positions = actions * (1.0 - fraction)
        means.append(float(estimate_stiffness(positions, actions, dt=0.02).mean()))
    assert means[0] < means[1] < means[2]


def test_stiffness_handles_short_episode():
    positions = np.zeros((1, 7), dtype=np.float32)
    actions = positions.copy()
    k = estimate_stiffness(positions, actions, dt=0.02)
    assert k.shape == (1,)


def test_stiffness_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        estimate_stiffness(np.zeros((5, 7)), np.zeros((5, 6)), dt=0.02)


# --- segment_phases -------------------------------------------------------


def _synth_episode(T: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic episode with clear approach → contact → manipulate → retract.

    - frames 0..49:    approach    (high velocity, low error)
    - frames 50..89:   contact     (low velocity, high error)
    - frames 90..149:  manipulate  (moderate velocity, moderate error)
    - frames 150..199: retract     (high velocity, low error)
    """
    velocity = np.zeros(T, dtype=np.float32)
    error = np.zeros(T, dtype=np.float32)

    velocity[0:50] = 1.0   # approach
    velocity[50:90] = 0.05  # contact (very low velocity)
    velocity[90:150] = 0.5  # manipulate (moderate)
    velocity[150:200] = 1.0  # retract

    error[0:50] = 0.05   # approach has low tracking error
    error[50:90] = 0.9    # contact has high tracking error
    error[90:150] = 0.6   # manipulate has moderate-high error
    error[150:200] = 0.05  # retract returns to baseline

    return velocity, error


def test_segment_returns_ordered_spans():
    v, e = _synth_episode(200)
    segs = segment_phases(v, e, dt=0.02)
    assert len(segs) >= 3
    # Strictly increasing start frames; non-overlapping
    for a, b in zip(segs, segs[1:]):
        assert a.end_frame < b.start_frame
    # Cover the whole episode
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == 199


def test_segment_finds_approach_contact_retract():
    v, e = _synth_episode(200)
    segs = segment_phases(v, e, dt=0.02)
    names = [s.name for s in segs]
    assert "approach" in names
    assert "contact" in names
    assert "retract" in names


def test_segment_no_contact_returns_approach_retract_only():
    """Pure free motion (no error spike) → approach + retract, no contact."""
    T = 100
    v = np.linspace(1.0, 1.0, T).astype(np.float32)
    e = np.full(T, 0.01, dtype=np.float32)
    segs = segment_phases(v, e, dt=0.02)
    names = {s.name for s in segs}
    assert "contact" not in names
    assert names <= {"approach", "retract"}


def test_segment_short_episode_falls_back():
    v = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    e = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    segs = segment_phases(v, e, dt=0.02)
    assert len(segs) == 1
