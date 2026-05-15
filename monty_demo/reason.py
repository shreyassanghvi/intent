"""Reasoning layer — the headline of the demo.

``reason()`` is the front door: given a new task spec, retrieve top-K
prior episodes, aggregate them into a phase plan + suggested skills +
recommended impedance regime, and merge with the human's prior knowledge
of the target objects (fragility, mass class, safety context). The output
is a ``TaskBrief`` — actionable bootstrap for the next attempt.

``ingest()`` closes the self-improvement loop: encode → segment → aggregate
per-phase k_hat → infer intent/objects → check for outliers → add to KG.

``print_brief_diff()`` is the money-shot helper: side-by-side comparison
of two briefs with the outlier notes from the most recent ingest.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np

from monty_demo._timing import timed
from monty_demo.encode import estimate_stiffness
from monty_demo.episode import Episode
from monty_demo.intent import infer_intent, infer_object_knowledge, infer_objects
from monty_demo.kg import KnowledgeGraph
from monty_demo.schemas import (
    IMPEDANCE_BANDS,
    BriefDiff,
    ObjectKnowledge,
    PhaseOutlier,
    PhasePlan,
    PhaseSegment,
    TaskBrief,
)
from monty_demo.segment import segment_phases

# Tunables
_MATCH_THRESHOLD = 0.5
_CROSS_EMB_PENALTY = 0.6           # multiplier on score when embodiment differs
_CROSS_EMB_CONF_DIP = 0.85         # multiplier on confidence when ≥2 embodiments
_OUTLIER_Z_INFO = 1.5
_OUTLIER_Z_WARNING = 2.0
_OUTLIER_Z_ALERT = 3.0
_TIGHTENED_BAND_DELTA = 0.01       # bands are "tightened" if width drops by this much
_DEFAULT_CONTACT_BAND = (0.3, 0.7)


# ---------------------------------------------------------------- Ingest


def _attach_k_hat_to_phases(
    phases: tuple[PhaseSegment, ...], k_hat: np.ndarray
) -> tuple[PhaseSegment, ...]:
    """Compute per-phase k_lo / k_hi from the per-frame k_hat trace."""
    out: list[PhaseSegment] = []
    for p in phases:
        slc = k_hat[p.start_frame : p.end_frame + 1]
        if slc.size == 0:
            out.append(p)
            continue
        out.append(
            p.model_copy(
                update={
                    "k_lo": float(np.percentile(slc, 25)),
                    "k_hi": float(np.percentile(slc, 75)),
                }
            )
        )
    return tuple(out)


def _detect_phase_outliers(kg: KnowledgeGraph, ep: Episode) -> list[PhaseOutlier]:
    """For each new phase, compute z-score of duration vs same-named phases
    already in the KG. Flag z >= _OUTLIER_Z_WARNING."""
    if ep.phases is None:
        return []
    out: list[PhaseOutlier] = []
    for ph in ep.phases:
        existing_durations = [
            attrs.get("duration_s", 0.0) for attrs in kg.iter_phase_nodes_by_name(ph.name)
        ]
        if len(existing_durations) < 2:
            continue
        mean = float(np.mean(existing_durations))
        std = float(np.std(existing_durations))
        if std < 1e-6:
            continue
        new_dur = (ph.end_frame - ph.start_frame + 1) * ep.dt
        z = abs(new_dur - mean) / std
        if z < _OUTLIER_Z_INFO:
            continue
        severity = (
            "alert" if z >= _OUTLIER_Z_ALERT
            else "warning" if z >= _OUTLIER_Z_WARNING
            else "info"
        )
        out.append(
            PhaseOutlier(
                episode_id=ep.episode_id,
                phase=ph.name,
                metric="duration",
                z_score=round(z, 2),
                severity=severity,
            )
        )
    return out


@timed("monty_demo.ingest")
def ingest(kg: KnowledgeGraph, ep: Episode) -> Episode:
    """Encode → segment → infer → outlier-check → KG.add. Returns the
    populated Episode. Idempotent on (repo_id, index): re-ingest is a no-op
    that returns the original Episode unchanged."""
    if kg.has_episode(ep.episode_id):
        kg._last_ingest_outliers = []     # type: ignore[attr-defined]
        return ep

    if ep.k_hat is None:
        ep = ep.with_stiffness(estimate_stiffness(ep.joint_positions, ep.joint_actions, ep.dt))
    if ep.phases is None:
        raw_phases = segment_phases(ep.ee_velocity_norm, ep.tracking_error, ep.dt)
        ep = ep.with_phases(_attach_k_hat_to_phases(raw_phases, ep.k_hat))
    if ep.intent is None:
        ep = ep.with_intent(infer_intent(ep))
    if not ep.objects:
        ep = ep.with_objects(infer_objects(ep))

    outliers = _detect_phase_outliers(kg, ep)
    kg._last_ingest_outliers = outliers   # type: ignore[attr-defined]
    kg.add(ep)
    return ep


# ---------------------------------------------------------------- Reason


def _score_episode(
    kg: KnowledgeGraph,
    ep_id: str,
    *,
    intent: str,
    target_objects: tuple[str, ...],
    embodiment: str | None,
) -> tuple[float, bool]:
    """Returns (score, was_cross_embodiment)."""
    ep_intent = kg.intent_of(ep_id)
    ep_emb = kg.embodiment_of(ep_id)
    ep_objects = set(kg.objects_of(ep_id))

    s_intent = 1.0 if ep_intent == intent else 0.0
    if target_objects:
        union = set(target_objects) | ep_objects
        s_obj = len(set(target_objects) & ep_objects) / len(union) if union else 0.0
    else:
        s_obj = 0.0
    s_emb = 1.0 if (embodiment is None or ep_emb == embodiment) else 0.0

    raw = 1.0 * s_intent + 0.5 * s_obj + 0.7 * s_emb
    cross_emb = embodiment is not None and ep_emb != embodiment and ep_emb is not None
    if cross_emb:
        raw *= _CROSS_EMB_PENALTY
    return raw, cross_emb


def _aggregate_phase_plan(
    kg: KnowledgeGraph, top_ep_ids: list[str]
) -> tuple[list[PhasePlan], list[float]]:
    """Returns (plan, band_widths_for_confidence)."""
    plan: list[PhasePlan] = []
    band_widths: list[float] = []
    threshold = max(1, (len(top_ep_ids) + 1) // 2)
    for phase_name in ("approach", "contact", "manipulate", "retract", "recover"):
        durations: list[float] = []
        k_los: list[float] = []
        k_his: list[float] = []
        for ep_id in top_ep_ids:
            for p in kg.phases_of(ep_id):
                if p.get("name") == phase_name:
                    durations.append(p.get("duration_s", 0.0))
                    k_los.append(p.get("k_lo", 0.0))
                    k_his.append(p.get("k_hi", 0.0))
                    break  # one per episode
        n_supporting = len(durations)
        if n_supporting < threshold:
            continue
        med = float(np.median(durations))
        lo = float(np.percentile(k_los, 25)) if k_los else 0.0
        hi = float(np.percentile(k_his, 75)) if k_his else 0.0
        plan.append(
            PhasePlan(
                name=phase_name,                                  # type: ignore[arg-type]
                expected_duration_s=round(med, 3),
                suggested_k_hat_band=(round(lo, 3), round(hi, 3)),
                n_supporting_episodes=n_supporting,
            )
        )
        band_widths.append(max(0.0, hi - lo))
    return plan, band_widths


def _band_midpoint_to_regime(band: tuple[float, float]) -> str:
    mid = 0.5 * (band[0] + band[1])
    best, best_dist = "compliant", 1e9
    for regime, (lo, hi) in IMPEDANCE_BANDS.items():
        rmid = 0.5 * (lo + hi)
        d = abs(rmid - mid)
        if d < best_dist:
            best, best_dist = regime, d
    return best


def _intersect_bands(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return (lo, hi) if hi >= lo else None


_REGIME_ORDER = ["gentle", "compliant", "firm", "stiff"]  # cautious → assertive


def _most_cautious_regime(regimes: list[str]) -> str:
    """Of a set of regime labels, return the gentlest (lowest in the order)."""
    return min(regimes, key=_REGIME_ORDER.index)


def _merge_impedance(
    data_band: tuple[float, float], object_knowledge: list[ObjectKnowledge]
) -> tuple[str, str | None]:
    """Merge a data-driven k_hat band with per-object human priors.

    Strategy: try to find a band that respects *all* priors AND the data
    (successive intersection). If any pair of priors / data has no overlap,
    fall back to the **most cautious** prior across all target objects —
    the safety-biased default that's order-independent and predictable.

    Returns (regime_label, conflict_note_or_None).
    """
    if not object_knowledge:
        return _band_midpoint_to_regime(data_band), None

    band = data_band
    conflicts: list[str] = []
    for ok in object_knowledge:
        prior = IMPEDANCE_BANDS.get(ok.suggested_impedance, _DEFAULT_CONTACT_BAND)
        intersection = _intersect_bands(band, prior)
        if intersection is None:
            conflicts.append(ok.name)
            continue
        band = intersection

    if conflicts:
        cautious = _most_cautious_regime([ok.suggested_impedance for ok in object_knowledge])
        return cautious, (
            f"data-driven k_hat {tuple(round(x, 2) for x in data_band)} conflicts with "
            f"priors on {conflicts} — defaulting to most cautious regime '{cautious}'"
        )
    return _band_midpoint_to_regime(band), None


_SAFETY_PHRASE = {
    "contains_liquid": "contains_liquid → spill risk during contact phase",
    "hot_surface":     "hot_surface → thermal-rated end-effector required",
    "electrical":      "electrical → de-energize before contact",
    "sharp":           "sharp → puncture risk; prefer pinch grasp away from edge",
}


def _humanize_safety(tag: str) -> str:
    return _SAFETY_PHRASE.get(tag, f"{tag} → caution")


@timed("monty_demo.reason")
def reason(
    kg: KnowledgeGraph,
    *,
    intent: str,
    target_objects: tuple[str, ...] = (),
    embodiment: str | None = None,
    k: int = 5,
) -> TaskBrief:
    """Build a TaskBrief from prior episodes for the requested task spec."""
    # 1. Score every episode
    scored: list[tuple[float, str, bool]] = []
    for ep_id in kg.episode_ids():
        s, cross = _score_episode(
            kg, ep_id, intent=intent, target_objects=target_objects, embodiment=embodiment
        )
        scored.append((s, ep_id, cross))
    scored.sort(key=lambda t: -t[0])
    top = scored[:k]
    # A "match" requires intent equality, not just a high score. An episode
    # with the right embodiment but wrong intent shouldn't count as a match
    # — it's still a useful neighbor, but it doesn't *bootstrap* this task.
    matched_episodes = [
        ep_id
        for s, ep_id, _ in top
        if s >= _MATCH_THRESHOLD and kg.intent_of(ep_id) == intent
    ]
    n_matched = len(matched_episodes)
    # Strict matches contribute confidence boost; cross-embodiment matches do
    # not (they're still in matched_episodes but the n_matched signal counts
    # only same-embodiment evidence — confirmed by the *0.85 dip below).
    n_strict = sum(
        1
        for s, ep_id, cross in top
        if s >= _MATCH_THRESHOLD and not cross and kg.intent_of(ep_id) == intent
    )
    top_ids = [ep_id for _, ep_id, _ in top]

    # 2. Phase plan
    plan, band_widths = _aggregate_phase_plan(kg, matched_episodes or top_ids)

    # 3. Skills
    skill_counts: dict[str, int] = {}
    for ep_id in matched_episodes or top_ids:
        for sk in kg.skills_of(ep_id):
            skill_counts[sk] = skill_counts.get(sk, 0) + 1
    suggested_skills = [s for s, _ in sorted(skill_counts.items(), key=lambda x: -x[1])]

    # 4. Adjacency — transferable skills from non-matching episodes
    top_skill_set = set(suggested_skills)
    matched_set = set(matched_episodes)
    transferable: list[str] = []
    for ep_id in kg.episode_ids():
        if ep_id in matched_set:
            continue
        ep_skills = set(kg.skills_of(ep_id))
        if ep_skills & top_skill_set:
            for sk in ep_skills - top_skill_set:
                if sk not in transferable:
                    transferable.append(sk)

    # 5. Objects seen before
    objects_seen_before: list[str] = []
    target_set = set(target_objects)
    for ep_id in matched_episodes or top_ids:
        for obj in kg.objects_of(ep_id):
            if obj in target_set and obj not in objects_seen_before:
                objects_seen_before.append(obj)

    # 6. Embodiment diversity (over matched episodes)
    embs = {kg.embodiment_of(ep_id) for ep_id in (matched_episodes or top_ids)}
    embs.discard(None)
    embodiment_diversity = max(1, len(embs))

    # 7. Confidence — gated on having any matches at all
    if n_matched == 0:
        confidence = 0.0
    else:
        mean_band = sum(band_widths) / len(band_widths) if band_widths else 0.5
        # Use n_strict (same-embodiment intent matches) for the boost. Cross-em
        # matches don't earn the boost; they only trigger the dip below.
        confidence = min(1.0, 0.2 * n_strict + 0.5 * (1.0 - mean_band))
        if embodiment is not None and embodiment_diversity > 1:
            confidence *= _CROSS_EMB_CONF_DIP

    # 8. Apply human priors
    object_knowledge: list[ObjectKnowledge] = []
    safety_set: set[str] = set()
    for obj_name in target_objects:
        attrs = kg.object_attrs(obj_name)
        if attrs is None:
            continue
        ok = ObjectKnowledge(
            name=obj_name,
            fragility=attrs["fragility"],
            mass_category=attrs["mass_category"],
            safety_context=kg.safety_tags_for_object(obj_name),
            suggested_impedance=attrs["suggested_impedance"],
        )
        object_knowledge.append(ok)
        safety_set.update(ok.safety_context)
    safety_warnings = [_humanize_safety(t) for t in sorted(safety_set)]

    # Merge with the data-driven contact band (or default if no data)
    contact_band = next(
        (p.suggested_k_hat_band for p in plan if p.name == "contact"),
        _DEFAULT_CONTACT_BAND,
    )
    data_regime = _band_midpoint_to_regime(contact_band)
    recommended_regime, conflict_note = _merge_impedance(contact_band, object_knowledge)

    # 9. Notes
    notes: list[str] = []
    if n_matched == 0:
        notes.append("no prior data — cold start")
    if embodiment is not None and embodiment_diversity > 1:
        notes.append(
            f"cross-embodiment evidence ({embodiment_diversity} embodiments) — "
            "confidence reduced for transfer uncertainty"
        )
    if transferable:
        notes.append(
            f"observed transferable skills from adjacent tasks: {', '.join(transferable)}"
        )
    if object_knowledge and recommended_regime != data_regime and conflict_note is None:
        notes.append(
            f"human priors tightened impedance regime from data-driven '{data_regime}' "
            f"to '{recommended_regime}'"
        )
    if conflict_note:
        notes.append(conflict_note)

    return TaskBrief(
        query_intent=intent,
        matched_episodes=matched_episodes,
        plan=plan,
        suggested_skills=suggested_skills,
        objects_seen_before=objects_seen_before,
        transferable_skills_observed=transferable,
        embodiment_diversity=embodiment_diversity,
        confidence=round(confidence, 3),
        notes=notes,
        object_knowledge=object_knowledge,
        recommended_impedance_regime=recommended_regime,
        safety_warnings=safety_warnings,
    )


# ---------------------------------------------------------------- Diff


def diff_briefs(
    before: TaskBrief, after: TaskBrief, recent_outliers: list[PhaseOutlier] | None = None
) -> BriefDiff:
    """Build a BriefDiff between two TaskBriefs (typically reason() before
    and after one or more ingest() calls)."""
    before_phases = {p.name: p.suggested_k_hat_band for p in before.plan}
    tightened: list[str] = []
    for p in after.plan:
        before_band = before_phases.get(p.name)
        if before_band is None:
            continue
        before_w = before_band[1] - before_band[0]
        after_w = p.suggested_k_hat_band[1] - p.suggested_k_hat_band[0]
        if (before_w - after_w) > _TIGHTENED_BAND_DELTA:
            tightened.append(p.name)

    new_skills = sorted(set(after.suggested_skills) - set(before.suggested_skills))
    new_objects = sorted(set(after.objects_seen_before) - set(before.objects_seen_before))
    new_transferable = sorted(
        set(after.transferable_skills_observed) - set(before.transferable_skills_observed)
    )
    return BriefDiff(
        matched_before=len(before.matched_episodes),
        matched_after=len(after.matched_episodes),
        confidence_delta=round(after.confidence - before.confidence, 3),
        tightened_phases=tightened,
        new_skills=new_skills,
        new_objects=new_objects,
        outlier_phases=list(recent_outliers or ()),
        new_transferable_skills=new_transferable,
    )


def print_brief_diff(
    before: TaskBrief,
    after: TaskBrief,
    kg: KnowledgeGraph | None = None,
) -> str:
    """Format a BriefDiff for printing. If ``kg`` is supplied, pulls the
    most recent ingest's outliers from it."""
    recent_outliers: list[PhaseOutlier] = []
    if kg is not None:
        recent_outliers = list(getattr(kg, "_last_ingest_outliers", []))
    d = diff_briefs(before, after, recent_outliers=recent_outliers)
    lines: list[str] = []
    lines.append(f"matched_episodes:        {d.matched_before} → {d.matched_after}")
    for name in d.tightened_phases:
        before_b = next(p.suggested_k_hat_band for p in before.plan if p.name == name)
        after_b = next(p.suggested_k_hat_band for p in after.plan if p.name == name)
        lines.append(
            f"{name:<8} stiffness band:  {before_b} → {after_b}   [tightened]"
        )
    delta_arrow = "↑" if d.confidence_delta > 0 else ("↓" if d.confidence_delta < 0 else "·")
    lines.append(
        f"confidence:              {before.confidence} → {after.confidence}   "
        f"({delta_arrow} {d.confidence_delta:+.3f})"
    )
    if d.new_transferable_skills:
        lines.append(f"new transferable skills: {d.new_transferable_skills}")
    if d.new_skills:
        lines.append(f"new skills:              {d.new_skills}")
    if d.new_objects:
        lines.append(f"new objects:             {d.new_objects}")
    if d.outlier_phases:
        for o in d.outlier_phases:
            lines.append(
                f"outlier phase:           {o.phase} (z={o.z_score}) on {o.episode_id} [{o.severity}]"
            )
    if before.recommended_impedance_regime != after.recommended_impedance_regime:
        lines.append(
            f"recommended_impedance:   {before.recommended_impedance_regime} → "
            f"{after.recommended_impedance_regime}"
        )
    if set(after.safety_warnings) - set(before.safety_warnings):
        new_safety = sorted(set(after.safety_warnings) - set(before.safety_warnings))
        for s in new_safety:
            lines.append(f"new safety warning:      {s}")
    out = "\n".join(lines)
    print(out)
    return out
