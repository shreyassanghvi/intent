"""Pydantic v2 contract surface. The ONLY module that other modules cross-import.

Speed-first stance: per-frame data lives as numpy arrays on the Episode dataclass,
not in pydantic models. Pydantic stays for metadata, KG refs, and the
reasoner's TaskBrief — the contract surface where type safety earns its weight.
"""

from typing import Literal

from pydantic import BaseModel, Field

PhaseName = Literal["approach", "contact", "manipulate", "retract", "recover"]


class EpisodeSource(BaseModel):
    repo_id: str
    index: int
    embodiment: str
    fps: float


class PhaseSegment(BaseModel):
    name: PhaseName
    start_frame: int
    end_frame: int
    confidence: float = 1.0
    source: Literal["heuristic", "ground_truth"] = "heuristic"
    k_lo: float = 0.0
    k_hi: float = 0.0


class Intent(BaseModel):
    name: str
    source: Literal["repo_metadata", "rule", "manual"] = "repo_metadata"
    confidence: float = 1.0


class PhasePlan(BaseModel):
    name: PhaseName
    expected_duration_s: float
    suggested_k_hat_band: tuple[float, float]
    n_supporting_episodes: int


class PhaseOutlier(BaseModel):
    episode_id: str
    phase: PhaseName
    metric: Literal["duration", "k_hat_peak", "k_hat_band_width"]
    z_score: float
    severity: Literal["info", "warning", "alert"]


class TaskBrief(BaseModel):
    query_intent: str
    matched_episodes: list[str]
    plan: list[PhasePlan]
    suggested_skills: list[str]
    objects_seen_before: list[str]
    transferable_skills_observed: list[str] = Field(default_factory=list)
    embodiment_diversity: int = 1
    confidence: float
    notes: list[str] = Field(default_factory=list)


class BriefDiff(BaseModel):
    matched_before: int
    matched_after: int
    confidence_delta: float
    tightened_phases: list[str] = Field(default_factory=list)
    new_skills: list[str] = Field(default_factory=list)
    new_objects: list[str] = Field(default_factory=list)
    outlier_phases: list[PhaseOutlier] = Field(default_factory=list)
    new_transferable_skills: list[str] = Field(default_factory=list)
