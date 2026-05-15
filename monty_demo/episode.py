"""Episode — the unit the pipeline operates on.

Frozen dataclass with ``__slots__``. Each pipeline step returns a *new* Episode
via ``dataclasses.replace`` with extra fields populated; arrays are shared by
reference between immutable variants (no copy on chain step).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from monty_demo._io import load_lerobot_episode
from monty_demo._timing import timed
from monty_demo.schemas import EpisodeSource, Intent, PhaseSegment


@dataclass(frozen=True, slots=True)
class Episode:
    source: EpisodeSource
    n_frames: int
    dt: float

    # Hot-path arrays (float32, contiguous)
    joint_positions: np.ndarray            # (T, DOF) — observed
    joint_actions: np.ndarray              # (T, DOF) — commanded
    ee_velocity_norm: np.ndarray           # (T,) — magnitude of per-frame joint-velocity vector
    effort: Optional[np.ndarray] = None    # (T, DOF) — joint torque proxy if dataset provides it

    # Pipeline-filled (None until that step runs)
    k_hat: Optional[np.ndarray] = None     # (T,) — estimated stiffness
    phases: Optional[tuple[PhaseSegment, ...]] = None
    intent: Optional[Intent] = None

    # KG-relevant metadata
    objects: tuple[str, ...] = field(default_factory=tuple)
    operator_id: Optional[str] = None

    # Construction --------------------------------------------------------

    @classmethod
    @timed("monty_demo.Episode.from_lerobot")
    def from_lerobot(cls, repo_id: str, index: int) -> "Episode":
        raw = load_lerobot_episode(repo_id, index)
        n_frames = int(raw.joint_positions.shape[0])
        dt = 1.0 / raw.fps if raw.fps > 0 else 1.0 / 30.0

        # Per-frame joint-velocity norm — eagerly computed so the dataclass
        # stays immutable and downstream code never has to re-derive it.
        if n_frames >= 2:
            v = np.linalg.norm(np.diff(raw.joint_positions, axis=0), axis=1) / dt
            ee_velocity_norm = np.concatenate([[v[0]], v]).astype(np.float32)
        else:
            ee_velocity_norm = np.zeros(n_frames, dtype=np.float32)

        return cls(
            source=EpisodeSource(
                repo_id=raw.repo_id,
                index=raw.index,
                embodiment=raw.embodiment,
                fps=raw.fps,
            ),
            n_frames=n_frames,
            dt=dt,
            joint_positions=raw.joint_positions,
            joint_actions=raw.joint_actions,
            ee_velocity_norm=ee_velocity_norm,
            effort=raw.effort,
        )

    # Immutable updates ---------------------------------------------------

    def with_stiffness(self, k: np.ndarray) -> "Episode":
        return replace(self, k_hat=k)

    def with_phases(self, p: tuple[PhaseSegment, ...]) -> "Episode":
        return replace(self, phases=p)

    def with_intent(self, i: Intent) -> "Episode":
        return replace(self, intent=i)

    def with_objects(self, o: tuple[str, ...]) -> "Episode":
        return replace(self, objects=o)

    # Identity ------------------------------------------------------------

    @property
    def episode_id(self) -> str:
        return f"ep:{self.source.repo_id}/{self.source.index}"

    @property
    def tracking_error(self) -> np.ndarray:
        """Per-frame ``||action - position||`` — derived, cheap, used by segmenter."""
        return np.linalg.norm(self.joint_actions - self.joint_positions, axis=1).astype(np.float32)
