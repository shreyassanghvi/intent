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


class ObjectKnowledge(BaseModel):
    """Human prior on a single object — what the operator implicitly knew while
    handling it. Combines categorical estimates a human makes at a glance
    (fragility, mass class) with task-relevant safety context and an impedance
    regime hint capturing 'how cautiously did I treat this object'.

    In production these would come from a controlled vocabulary the operator
    picks at recording time, an object-classifier model, or a task spec.
    For the demo they're curated; the point isn't to *infer* the prior but to
    show that, *given* a prior, the reasoner produces materially better briefs.
    """

    name: str
    fragility: Literal["robust", "moderate", "fragile", "very_fragile"]
    mass_category: Literal["light", "medium", "heavy"]   # <0.2kg / 0.2-2kg / >2kg
    safety_context: list[str] = Field(default_factory=list)  # e.g. ["contains_liquid", "hot_surface"]
    suggested_impedance: Literal["gentle", "compliant", "firm", "stiff"]


# Numeric bands for each impedance regime. The reasoner intersects these with
# the data-driven k_hat band from prior episodes to produce the merged
# `recommended_impedance_regime` on a TaskBrief.
IMPEDANCE_BANDS: dict[str, tuple[float, float]] = {
    "gentle":    (0.00, 0.45),
    "compliant": (0.30, 0.65),
    "firm":      (0.50, 0.85),
    "stiff":     (0.70, 1.00),
}


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

    # --- human-priors layer (the "what the human knew" surface) ---
    object_knowledge: list[ObjectKnowledge] = Field(default_factory=list)
    recommended_impedance_regime: str = "compliant"   # merge of data-driven k_hat band + human priors
    safety_warnings: list[str] = Field(default_factory=list)


class BriefDiff(BaseModel):
    matched_before: int
    matched_after: int
    confidence_delta: float
    tightened_phases: list[str] = Field(default_factory=list)
    new_skills: list[str] = Field(default_factory=list)
    new_objects: list[str] = Field(default_factory=list)
    outlier_phases: list[PhaseOutlier] = Field(default_factory=list)
    new_transferable_skills: list[str] = Field(default_factory=list)
