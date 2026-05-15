"""Phase segmenter.

Vectorized rule-based state machine over per-frame velocity norm and tracking
error. Returns ``PhaseSegment`` *spans*, not per-frame labels — phases are
intervals; the KG and reasoner consume intervals.

Targets 3–6 segments per episode. < 2 ms for 250 frames.
"""

from __future__ import annotations

import numpy as np

from monty_demo._timing import timed
from monty_demo.schemas import PhaseName, PhaseSegment

_MIN_SEGMENT_FRAMES = 5       # absorb noisy sub-5-frame spans into neighbors
_CONTACT_ERR_FRACTION = 0.4   # relative error threshold for "in contact zone"
_LOW_VEL_FRACTION = 0.2       # relative velocity threshold for "near-stationary"
_SMOOTH_WINDOW = 9            # boxcar window for velocity/error smoothing
# Absolute floor: if the *peak* tracking error never exceeds this in raw units
# (joint-space, typically radians on ALOHA / Koch), there was no real contact —
# normalization would otherwise inflate uniformly-low noise into a fake contact
# zone.
_NO_CONTACT_ERR_FLOOR = 0.05


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Boxcar smoothing — knocks frame-level chatter out of the segmenter
    inputs without changing their overall shape. Vectorized."""
    if x.size == 0 or window <= 1:
        return x
    w = min(window, x.size)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _coalesce_same_name(segments: list[PhaseSegment]) -> list[PhaseSegment]:
    """Merge consecutive segments with the same name into one — real teleop
    alternates contact↔manipulate frequently and the reasoner cares about
    spans, not micro-oscillations."""
    if len(segments) <= 1:
        return segments
    out: list[PhaseSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if seg.name == prev.name and seg.start_frame == prev.end_frame + 1:
            out[-1] = prev.model_copy(update={
                "end_frame": seg.end_frame,
                "confidence": min(prev.confidence, seg.confidence),
            })
        else:
            out.append(seg)
    return out


def _merge_tiny(segments: list[PhaseSegment], min_len: int) -> list[PhaseSegment]:
    """Absorb segments shorter than ``min_len`` into the longer neighbor."""
    if len(segments) <= 1:
        return segments
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            if seg.end_frame - seg.start_frame + 1 >= min_len:
                continue
            left = segments[i - 1] if i > 0 else None
            right = segments[i + 1] if i < len(segments) - 1 else None
            if left and right:
                target = left if (left.end_frame - left.start_frame) >= (right.end_frame - right.start_frame) else right
            else:
                target = left or right
            assert target is not None  # len > 1 guarantees at least one neighbor
            if target is left:
                segments[i - 1] = target.model_copy(update={"end_frame": seg.end_frame})
            else:
                segments[i + 1] = target.model_copy(update={"start_frame": seg.start_frame})
            segments.pop(i)
            changed = True
            break
    return segments


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Run-length encode a bool array → list of (start, end_inclusive, value)."""
    if mask.size == 0:
        return []
    out = []
    cur = bool(mask[0])
    cur_start = 0
    for i in range(1, mask.size):
        v = bool(mask[i])
        if v != cur:
            out.append((cur_start, i - 1, cur))
            cur = v
            cur_start = i
    out.append((cur_start, mask.size - 1, cur))
    return out


@timed("monty_demo.segment_phases")
def segment_phases(
    velocity_norm: np.ndarray,
    tracking_error: np.ndarray,
    dt: float,
) -> tuple[PhaseSegment, ...]:
    """Segment an episode into ordered ``PhaseSegment`` spans.

    Strategy:
        1. Locate the *work zone*: contiguous frames whose tracking error
           exceeds 40% of the episode max — that's where something interactive
           happened. Frames before/after that zone are approach/retract.
        2. Inside the work zone, classify each frame as ``contact`` (low
           velocity, high error) or ``manipulate`` (moving + interacting).
        3. Run-length encode the work-zone classification, prepend approach,
           append retract, then absorb tiny segments into their longer neighbor.
    """
    if velocity_norm.shape != tracking_error.shape or velocity_norm.ndim != 1:
        raise ValueError("velocity_norm and tracking_error must be 1-D and same length")
    T = int(velocity_norm.shape[0])
    if T < 10:
        return (PhaseSegment(name="manipulate", start_frame=0, end_frame=max(0, T - 1), confidence=0.4),)

    # Smooth before normalization — real teleop chatters at frame level.
    velocity_norm = _smooth(velocity_norm, _SMOOTH_WINDOW)
    tracking_error = _smooth(tracking_error, _SMOOTH_WINDOW)

    v_max = float(velocity_norm.max()) + 1e-6
    e_max_raw = float(tracking_error.max())
    e_max = e_max_raw + 1e-6
    v = velocity_norm / v_max
    e = tracking_error / e_max

    # Below the absolute floor, treat as pure free motion regardless of how
    # "high" the relative error looks after normalization.
    if e_max_raw < _NO_CONTACT_ERR_FLOOR:
        contact_mask = np.zeros_like(e, dtype=bool)
    else:
        contact_mask = e > _CONTACT_ERR_FRACTION
    if not contact_mask.any():
        # Pure free motion — split the episode in half, approach then retract.
        mid = T // 2
        return (
            PhaseSegment(name="approach", start_frame=0, end_frame=mid - 1, confidence=0.3),
            PhaseSegment(name="retract", start_frame=mid, end_frame=T - 1, confidence=0.3),
        )

    contact_start = int(np.argmax(contact_mask))
    contact_end = T - 1 - int(np.argmax(contact_mask[::-1]))

    segments: list[PhaseSegment] = []
    if contact_start > 0:
        segments.append(
            PhaseSegment(name="approach", start_frame=0, end_frame=contact_start - 1)
        )

    work_v = v[contact_start : contact_end + 1]
    is_contact = work_v < _LOW_VEL_FRACTION
    for run_start, run_end, val in _runs(is_contact):
        name: PhaseName = "contact" if val else "manipulate"
        segments.append(
            PhaseSegment(
                name=name,
                start_frame=contact_start + run_start,
                end_frame=contact_start + run_end,
            )
        )

    if contact_end < T - 1:
        segments.append(
            PhaseSegment(name="retract", start_frame=contact_end + 1, end_frame=T - 1)
        )

    # Two-pass cleanup: absorb tiny noisy segments into neighbors, then
    # coalesce consecutive same-name spans (the absorption can produce them).
    segments = _merge_tiny(segments, _MIN_SEGMENT_FRAMES)
    segments = _coalesce_same_name(segments)
    return tuple(segments)
