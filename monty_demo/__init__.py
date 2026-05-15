"""monty_demo — reasoning layer for physical-AI teleop data.

Pitch artifact for usemonty.com. The headline call is ``reason()``; ``ingest()``
closes the self-improvement loop. Everything else is supporting machinery.
"""

from monty_demo.schemas import (
    IMPEDANCE_BANDS,
    BriefDiff,
    EpisodeSource,
    Intent,
    ObjectKnowledge,
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
from monty_demo._timing import timing_summary

# reason.py exports (ingest, print_brief_diff, reason) added in step 7

__all__ = [
    "BriefDiff",
    "Episode",
    "EpisodeSource",
    "IMPEDANCE_BANDS",
    "Intent",
    "KnowledgeGraph",
    "ObjectKnowledge",
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
    "segment_phases",
    "timing_summary",
]