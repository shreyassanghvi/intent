"""Curated metadata for every dataset the demo touches.

LeRobot episodes don't carry intent/skill/object metadata. We populate all of
it — including the human-prior ``ObjectKnowledge`` for every object the operator
handled — from this single table keyed by ``repo_id``. Provenance is explicit:
``Intent.source`` records whether a label came from repo metadata, a rule, or
a manual hand-label.

In production each of these would come from a controlled vocabulary picked at
recording time, an object-classifier model, or an operator-authored task spec.
For the demo it's a curated lookup so the reasoner's behavior is fully
deterministic and inspectable. The point of the demo isn't to *infer* the human
prior — it's to show that, *given* the prior, the reasoner produces materially
better briefs.

──────────────────────────────────────────────────────────────────────────────
Adding a new dataset — copy-paste template
──────────────────────────────────────────────────────────────────────────────

    "lerobot/<your_dataset>": RepoMetadata(
        intent=Intent(name="<task-name>", source="repo_metadata"),
        skills=("<skill-1>", "<skill-2>"),
        objects=(
            ObjectKnowledge(
                name="<object-name>",
                fragility="moderate",
                mass_category="light",
                safety_context=["<tag>"],
                suggested_impedance="gentle",
            ),
        ),
    ),

Valid values:
    Intent.source         "repo_metadata" | "rule" | "manual"
    fragility             "robust" | "moderate" | "fragile" | "very_fragile"
    mass_category         "light" (<0.2kg) | "medium" (0.2-2kg) | "heavy" (>2kg)
    suggested_impedance   "gentle" | "compliant" | "firm" | "stiff"
                          (each maps to a numeric band in schemas.IMPEDANCE_BANDS)
    safety_context        any list of strings; the four recognized tags get
                          human-readable explanations in TaskBrief.safety_warnings:
                            "contains_liquid"  → "spill risk during contact phase"
                            "hot_surface"      → "thermal-rated end-effector required"
                            "electrical"       → "de-energize before contact"
                            "sharp"            → "puncture risk; prefer pinch grasp"
                          Other tags fall back to "{tag} → caution".

After editing, ``Episode.from_lerobot("lerobot/<your_dataset>", index=N)``
followed by ``ingest(kg, ep)`` runs the full pipeline against the new entry
with NO other code changes — the KG, reasoner, and brief surface adapt
automatically because they read everything via the schemas.
"""

from __future__ import annotations

from dataclasses import dataclass

from monty_demo.episode import Episode
from monty_demo.schemas import Intent, ObjectKnowledge


@dataclass(frozen=True)
class RepoMetadata:
    intent: Intent
    skills: tuple[str, ...]
    objects: tuple[ObjectKnowledge, ...]


REPO_METADATA: dict[str, RepoMetadata] = {
    # --- Setup + outlier (same repo, different episode indices) --------------
    "lerobot/aloha_static_coffee": RepoMetadata(
        intent=Intent(name="brew-coffee", source="repo_metadata"),
        skills=("fine-bimanual-coordination", "place", "press-button"),
        objects=(
            ObjectKnowledge(
                name="mug",
                fragility="moderate",
                mass_category="light",
                safety_context=["contains_liquid"],
                suggested_impedance="gentle",
            ),
            ObjectKnowledge(
                name="coffee-machine",
                fragility="robust",
                mass_category="heavy",
                safety_context=["hot_surface", "electrical"],
                suggested_impedance="firm",
            ),
            ObjectKnowledge(
                name="filter-pod",
                fragility="fragile",
                mass_category="light",
                safety_context=[],
                suggested_impedance="gentle",
            ),
        ),
    ),
    # --- Attempt 1: cross-skill transfer (different intent, shared skill) ----
    "lerobot/aloha_static_thread_velcro": RepoMetadata(
        intent=Intent(name="thread-velcro", source="repo_metadata"),
        skills=("fine-bimanual-coordination", "thread", "pinch-grasp"),
        objects=(
            ObjectKnowledge(
                name="velcro-strap",
                fragility="robust",
                mass_category="light",
                safety_context=[],
                suggested_impedance="firm",
            ),
            ObjectKnowledge(
                name="cloth",
                fragility="moderate",
                mass_category="light",
                safety_context=[],
                suggested_impedance="compliant",
            ),
        ),
    ),
    # --- Attempt 2: cross-embodiment (single-arm, hand-labeled adjacent) -----
    "lerobot/koch_pick_place_5_lego": RepoMetadata(
        intent=Intent(name="brew-coffee", source="manual", confidence=0.4),
        skills=("place", "pick"),
        objects=(
            ObjectKnowledge(
                name="mug",
                fragility="moderate",
                mass_category="light",
                safety_context=["contains_liquid"],
                suggested_impedance="gentle",
            ),
        ),
    ),
}


def _md(ep: Episode) -> RepoMetadata | None:
    return REPO_METADATA.get(ep.source.repo_id)


def infer_intent(ep: Episode) -> Intent:
    md = _md(ep)
    if md is None:
        return Intent(name="unknown", source="rule", confidence=0.0)
    return md.intent


def infer_skills(ep: Episode) -> tuple[str, ...]:
    md = _md(ep)
    return md.skills if md else ()


def infer_object_knowledge(ep: Episode) -> tuple[ObjectKnowledge, ...]:
    md = _md(ep)
    return md.objects if md else ()


def infer_objects(ep: Episode) -> tuple[str, ...]:
    """Names only — convenience for the KG INVOLVES edge and Episode.objects."""
    return tuple(ok.name for ok in infer_object_knowledge(ep))
