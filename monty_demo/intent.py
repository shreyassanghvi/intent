"""Curated metadata for every dataset the demo touches.

LeRobot episodes don't carry intent/skill/object metadata. We populate all of
it — including the human-prior ``ObjectKnowledge`` for every object the operator
handled — from this single table keyed by ``(repo_id, episode_index)``.
Provenance is explicit: ``Intent.source`` records whether a label came from
repo metadata, a rule, or a manual hand-label.

The key shape ``(repo_id, episode_index | None)`` supports two patterns:

* ``(repo_id, None)`` — **per-repo default**: applies to every episode in the
  dataset unless overridden. Most entries are defaults.
* ``(repo_id, N)`` — **per-episode override**: applies to one specific episode
  only. Useful when a dataset contains multiple tasks (different episodes ran
  different intents), or when a particular recording's labels need correction.

Lookup order: per-episode key first, repo-level default second, ``unknown``
if neither matches.

In production each of these would come from a controlled vocabulary picked at
recording time, an object-classifier model, or an operator-authored task spec.
For the demo it's a curated lookup so the reasoner's behavior is fully
deterministic and inspectable. The point of the demo isn't to *infer* the human
prior — it's to show that, *given* the prior, the reasoner produces materially
better briefs.

──────────────────────────────────────────────────────────────────────────────
Adding a new dataset — copy-paste template
──────────────────────────────────────────────────────────────────────────────

    # Default for all episodes in this dataset
    ("lerobot/<your_dataset>", None): RepoMetadata(
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

    # Optional per-episode override (e.g. episode 7 is a different task)
    ("lerobot/<your_dataset>", 7): RepoMetadata(
        intent=Intent(name="<other-task>", source="manual", confidence=0.7),
        skills=(...),
        objects=(...),
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


REPO_METADATA: dict[tuple[str, int | None], RepoMetadata] = {
    # --- Setup + outlier (same repo, different episode indices) --------------
    ("lerobot/aloha_static_coffee", None): RepoMetadata(
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
    # Battery insertion shares fine-bimanual-coordination with brew-coffee, but
    # crucially adds 'precision-insert' and 'align-and-press' — exactly the
    # primitives needed for filter-pod insertion. The reasoner surfaces these
    # as transferable skills for future brew-coffee briefs.
    ("lerobot/aloha_static_battery", None): RepoMetadata(
        intent=Intent(name="insert-battery", source="repo_metadata"),
        skills=("fine-bimanual-coordination", "precision-insert", "align-and-press"),
        objects=(
            ObjectKnowledge(
                name="battery",
                fragility="moderate",
                mass_category="light",
                safety_context=["electrical"],
                suggested_impedance="gentle",
            ),
            ObjectKnowledge(
                name="battery-slot",
                fragility="robust",
                mass_category="medium",
                safety_context=["electrical"],
                suggested_impedance="firm",
            ),
        ),
    ),
    # Kept for tests (test_reason.py uses it for cross-skill cases on synthetic
    # episodes); not currently wired into the demo notebook.
    ("lerobot/aloha_static_thread_velcro", None): RepoMetadata(
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
    ("lerobot/koch_pick_place_5_lego", None): RepoMetadata(
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


def get_metadata(repo_id: str, episode_index: int | None = None) -> RepoMetadata | None:
    """Look up metadata for an episode.

    Per-episode override first (``(repo_id, episode_index)``), then per-repo
    default (``(repo_id, None)``). Returns ``None`` if neither key is present.
    """
    if episode_index is not None:
        specific = REPO_METADATA.get((repo_id, episode_index))
        if specific is not None:
            return specific
    return REPO_METADATA.get((repo_id, None))


def _md(ep: Episode) -> RepoMetadata | None:
    return get_metadata(ep.source.repo_id, ep.source.index)


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
