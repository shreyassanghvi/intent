"""Reasoner tests — covers reason() + ingest() + outlier + cross-embodiment +
human-priors merge."""

from __future__ import annotations

import numpy as np
import pytest

from monty_demo.episode import Episode
from monty_demo.kg import KnowledgeGraph
from monty_demo.reason import diff_briefs, ingest, reason
from monty_demo.schemas import EpisodeSource, Intent, PhaseSegment


def _make_episode(
    repo_id: str = "lerobot/aloha_static_coffee",
    index: int = 0,
    embodiment: str = "aloha-bimanual",
    intent_name: str = "brew-coffee",
    contact_k_band: tuple[float, float] = (0.4, 0.7),
    contact_duration_s: float = 1.0,
) -> Episode:
    T, D = 50, 14
    fps = 50.0
    dt = 1.0 / fps
    contact_frames = int(contact_duration_s * fps)
    contact_start = 5
    contact_end = min(T - 5, contact_start + contact_frames - 1)
    return Episode(
        source=EpisodeSource(repo_id=repo_id, index=index, embodiment=embodiment, fps=fps),
        n_frames=T,
        dt=dt,
        joint_positions=np.zeros((T, D), dtype=np.float32),
        joint_actions=np.zeros((T, D), dtype=np.float32),
        ee_velocity_norm=np.zeros(T, dtype=np.float32),
        intent=Intent(name=intent_name, source="repo_metadata"),
        phases=(
            PhaseSegment(name="approach", start_frame=0, end_frame=contact_start - 1,
                         k_lo=0.0, k_hi=0.1),
            PhaseSegment(name="contact", start_frame=contact_start, end_frame=contact_end,
                         k_lo=contact_k_band[0], k_hi=contact_k_band[1]),
            PhaseSegment(name="retract", start_frame=contact_end + 1, end_frame=T - 1,
                         k_lo=0.0, k_hi=0.1),
        ),
    )


# ---- Cold start ---------------------------------------------------------


def test_cold_start_returns_empty_brief_with_zero_confidence():
    kg = KnowledgeGraph()
    brief = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")
    assert brief.matched_episodes == []
    assert brief.confidence == 0.0
    assert "no prior data" in brief.notes[0]


def test_cold_start_does_not_emit_foreign_intent_plan():
    """When no episodes match the query intent, the plan / skills / objects
    must be empty — pulling them from the top-K scored episodes of a
    *different* intent would put a populated plan next to confidence=0.0,
    which is self-contradicting.
    """
    kg = KnowledgeGraph()
    # KG has only velcro episodes; query is for brew-coffee
    for i in range(2):
        kg.add(_make_episode(
            index=i,
            repo_id="lerobot/aloha_static_thread_velcro",
            intent_name="thread-velcro",
        ))
    brief = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")
    assert brief.matched_episodes == []
    assert brief.confidence == 0.0
    assert brief.plan == []
    assert brief.suggested_skills == []
    assert brief.objects_seen_before == []


# ---- Same-intent ingests tighten the brief ------------------------------


def test_more_same_intent_data_tightens_contact_band():
    kg = KnowledgeGraph()
    # Three episodes with progressively tighter k_hat bands around 0.5
    for i, band in enumerate([(0.30, 0.80), (0.40, 0.70), (0.45, 0.65)]):
        kg.add(_make_episode(index=i, contact_k_band=band))

    brief_before = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")
    assert len(brief_before.matched_episodes) == 3

    # Add a fourth right at the median → contact band should not widen
    kg.add(_make_episode(index=3, contact_k_band=(0.48, 0.62)))
    brief_after = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")

    diff = diff_briefs(brief_before, brief_after)
    assert diff.matched_after >= diff.matched_before
    # Contact band tightened (or stayed roughly the same, not widened)
    contact_before = next(p.suggested_k_hat_band for p in brief_before.plan if p.name == "contact")
    contact_after = next(p.suggested_k_hat_band for p in brief_after.plan if p.name == "contact")
    width_before = contact_before[1] - contact_before[0]
    width_after = contact_after[1] - contact_after[0]
    assert width_after <= width_before + 0.01


# ---- Cross-skill: transferable skills appear ----------------------------


def test_cross_skill_episode_surfaces_transferable_skills():
    kg = KnowledgeGraph()
    kg.add(_make_episode(index=0))
    kg.add(_make_episode(index=1))
    # Add a velcro episode — different intent, shared "fine-bimanual-coordination" skill
    kg.add(
        _make_episode(
            index=0,
            repo_id="lerobot/aloha_static_thread_velcro",
            intent_name="thread-velcro",
        )
    )
    brief = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")
    # transferable_skills_observed should include skills from the velcro episode
    # that aren't in the matched (brew-coffee) episodes' skill set
    assert "thread" in brief.transferable_skills_observed or "pinch-grasp" in brief.transferable_skills_observed


# ---- Cross-embodiment: confidence dips ----------------------------------


def test_cross_embodiment_drops_confidence_despite_more_matches():
    kg = KnowledgeGraph()
    for i in range(2):
        kg.add(_make_episode(index=i, contact_k_band=(0.45, 0.65)))
    brief_aloha_only = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")

    # Add a Koch episode (same intent, different embodiment, hand-labeled)
    kg.add(
        _make_episode(
            index=0,
            repo_id="lerobot/koch_pick_place_5_lego",
            embodiment="koch",
            intent_name="brew-coffee",
        )
    )
    brief_after = reason(kg, intent="brew-coffee", embodiment="aloha-bimanual")
    assert brief_after.embodiment_diversity == 2
    # Confidence should drop because of the cross-embodiment penalty,
    # even though n_matched would otherwise rise
    assert brief_after.confidence < brief_aloha_only.confidence


# ---- Outlier detection on ingest ----------------------------------------


def _make_real_episode_for_ingest(
    repo_id: str = "lerobot/aloha_static_coffee",
    index: int = 0,
    contact_extra_frames: int = 0,
) -> Episode:
    """Build an Episode whose joint signals will produce a clear contact phase
    after the real encode/segment pipeline. ``contact_extra_frames`` widens
    the contact phase so we can produce duration outliers."""
    T = 100 + contact_extra_frames
    D = 14
    fps = 50.0
    dt = 1.0 / fps
    actions = np.zeros((T, D), dtype=np.float32)
    positions = np.zeros((T, D), dtype=np.float32)

    # Approach: 0..29 — large commanded + observed motion (free, low error)
    for t in range(0, 30):
        actions[t] = t * 0.05
        positions[t] = t * 0.05
    # Contact: 30..(60+extra) — commanded keeps moving, position frozen → high error
    contact_end = 60 + contact_extra_frames
    actions[30:contact_end] = actions[29] + np.linspace(0, 1.0, contact_end - 30).reshape(-1, 1)
    positions[30:contact_end] = positions[29]                # held — high tracking error
    # Retract: contact_end..T — both move freely back
    for t in range(contact_end, T):
        actions[t] = actions[contact_end - 1] - (t - contact_end + 1) * 0.05
        positions[t] = actions[t]

    # ee_velocity_norm computed eagerly mimicking Episode.from_lerobot
    v = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    ee_v = np.concatenate([[v[0]], v]).astype(np.float32)

    return Episode(
        source=EpisodeSource(repo_id=repo_id, index=index, embodiment="aloha-bimanual", fps=fps),
        n_frames=T,
        dt=dt,
        joint_positions=positions,
        joint_actions=actions,
        ee_velocity_norm=ee_v,
    )


def test_ingest_pipeline_populates_episode_and_kg():
    kg = KnowledgeGraph()
    ep = _make_real_episode_for_ingest(index=0)
    populated = ingest(kg, ep)
    assert populated.k_hat is not None
    assert populated.phases is not None
    assert populated.intent is not None
    assert populated.intent.name == "brew-coffee"
    assert kg.has_episode(populated.episode_id)
    # Should have produced at least one contact phase given the synthetic profile
    phase_names = {p.name for p in populated.phases}
    assert "contact" in phase_names


def test_ingest_outlier_detection_flags_long_contact():
    kg = KnowledgeGraph()
    # Three baseline ingests with slight contact-length variance so std > 0
    # (otherwise outlier z-score is undefined and detection is skipped).
    for i in range(3):
        ingest(kg, _make_real_episode_for_ingest(index=i, contact_extra_frames=i * 4))
    # Fourth: contact phase dramatically longer
    ingest(kg, _make_real_episode_for_ingest(index=3, contact_extra_frames=120))
    outliers = getattr(kg, "_last_ingest_outliers", [])
    contact_outliers = [o for o in outliers if o.phase == "contact"]
    assert len(contact_outliers) >= 1
    assert contact_outliers[0].z_score >= 2.0
    assert contact_outliers[0].severity in ("warning", "alert")


def test_ingest_is_idempotent():
    kg = KnowledgeGraph()
    ep = _make_real_episode_for_ingest(index=0)
    ingest(kg, ep)
    s1 = kg.stats()
    ingest(kg, ep)
    s2 = kg.stats()
    assert s1 == s2


# ---- Human-priors merge -------------------------------------------------


def test_human_priors_tighten_recommended_impedance():
    kg = KnowledgeGraph()
    # Data says contact is "firm" (k_hat 0.5..0.85)
    for i in range(3):
        kg.add(_make_episode(index=i, contact_k_band=(0.55, 0.80)))
    # Reason with "mug" as target — REPO_METADATA says mug is "gentle"
    brief = reason(
        kg,
        intent="brew-coffee",
        target_objects=("mug",),
        embodiment="aloha-bimanual",
    )
    # Object knowledge should be populated
    assert any(ok.name == "mug" for ok in brief.object_knowledge)
    # Safety warnings should mention liquid
    assert any("liquid" in w.lower() for w in brief.safety_warnings)
    # Recommended regime should NOT be the data-driven "firm" — gentle prior wins
    assert brief.recommended_impedance_regime in ("gentle", "compliant")


def test_safety_warnings_aggregate_across_target_objects():
    kg = KnowledgeGraph()
    for i in range(2):
        kg.add(_make_episode(index=i))
    brief = reason(
        kg,
        intent="brew-coffee",
        target_objects=("mug", "coffee-machine"),
        embodiment="aloha-bimanual",
    )
    warnings_lower = " ".join(brief.safety_warnings).lower()
    assert "liquid" in warnings_lower
    assert "hot_surface" in warnings_lower or "thermal" in warnings_lower
    assert "electrical" in warnings_lower


def test_safety_warnings_pull_from_all_matched_episode_objects_not_just_targets():
    """Safety context belongs to the TASK, not just the operator's target_objects.

    A fixture (coffee-machine) the operator doesn't grasp but the task
    involves still contributes its hazard tags to the brief.
    """
    kg = KnowledgeGraph()
    for i in range(2):
        kg.add(_make_episode(index=i))   # ingests mug + coffee-machine + filter-pod

    # target_objects deliberately excludes coffee-machine (it's a fixture)
    brief = reason(
        kg,
        intent="brew-coffee",
        target_objects=("mug",),
        embodiment="aloha-bimanual",
    )

    # ObjectKnowledge stays target-scoped: only mug in object_knowledge
    assert [ok.name for ok in brief.object_knowledge] == ["mug"]

    # Safety surface is task-scoped: contains_liquid (mug), hot_surface +
    # electrical (coffee-machine, NOT a target) all show up
    warnings_lower = " ".join(brief.safety_warnings).lower()
    assert "liquid" in warnings_lower
    assert ("hot_surface" in warnings_lower) or ("thermal" in warnings_lower)
    assert "electrical" in warnings_lower


def test_merge_impedance_does_not_falsely_conflict_due_to_earlier_intersections():
    """Regression: _merge_impedance used to test each prior against the
    *running* (already-narrowed) band, so a third prior could appear to
    "conflict" with the data when it actually fit fine — it just didn't
    fit inside the band that earlier priors had already squeezed.

    Set up a case: data band is wide, two priors narrow it from both sides,
    a third prior overlaps the data but not the post-2 narrowing. The third
    prior must NOT trigger the data-conflict fallback note.
    """
    from monty_demo.reason import _merge_impedance
    from monty_demo.schemas import ObjectKnowledge

    data_band = (0.20, 0.80)
    # gentle (0.00, 0.45) narrows data → (0.20, 0.45)
    # compliant (0.30, 0.65) narrows running → (0.30, 0.45)
    # firm (0.50, 0.85) does NOT fit (0.30, 0.45) but DOES intersect data (0.50, 0.80)
    priors = [
        ObjectKnowledge(name="a", fragility="moderate", mass_category="light",
                        safety_context=[], suggested_impedance="gentle"),
        ObjectKnowledge(name="b", fragility="moderate", mass_category="light",
                        safety_context=[], suggested_impedance="compliant"),
        ObjectKnowledge(name="c", fragility="robust", mass_category="light",
                        safety_context=[], suggested_impedance="firm"),
    ]
    regime, conflict_note = _merge_impedance(data_band, priors)

    # No data conflict — every prior overlaps data_band individually
    assert conflict_note is None, f"Expected no conflict note, got: {conflict_note}"
    # Regime should be something sensible (gentlest containing band wins)
    assert regime in ("gentle", "compliant"), regime


def test_unknown_target_object_does_not_crash():
    kg = KnowledgeGraph()
    kg.add(_make_episode())
    brief = reason(
        kg,
        intent="brew-coffee",
        target_objects=("unknown-thing",),
        embodiment="aloha-bimanual",
    )
    # No knowledge for unknown object → not in object_knowledge, no safety warnings from it
    assert all(ok.name != "unknown-thing" for ok in brief.object_knowledge)
