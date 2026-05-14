"""monty_demo — reasoning layer for physical-AI teleop data.

Pitch artifact for usemonty.com. The headline call is ``reason()``; ``ingest()``
closes the self-improvement loop. Everything else is supporting machinery.
"""

from monty_demo.schemas import (
    BriefDiff,
    EpisodeSource,
    Intent,
    PhaseName,
    PhaseOutlier,
    PhasePlan,
    PhaseSegment,
    TaskBrief,
)
from monty_demo.episode import Episode
from monty_demo.encode import estimate_stiffness
from monty_demo.segment import segment_phases
from monty_demo.intent import REPO_METADATA, infer_intent, infer_objects, infer_skills
from monty_demo.kg import KnowledgeGraph
from monty_demo.reason import ingest, print_brief_diff, reason
from monty_demo._timing import timing_summary

__all__ = [
    "BriefDiff",
    "Episode",
    "EpisodeSource",
    "Intent",
    "KnowledgeGraph",
    "PhaseName",
    "PhaseOutlier",
    "PhasePlan",
    "PhaseSegment",
    "REPO_METADATA",
    "TaskBrief",
    "estimate_stiffness",
    "infer_intent",
    "infer_objects",
    "infer_skills",
    "ingest",
    "print_brief_diff",
    "reason",
    "segment_phases",
    "timing_summary",
]