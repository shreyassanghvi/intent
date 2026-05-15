"""KnowledgeGraph unit tests on synthetic Episodes (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from monty_demo.episode import Episode
from monty_demo.kg import KnowledgeGraph
from monty_demo.schemas import EpisodeSource, Intent, PhaseSegment


def _make_episode(
    repo_id: str = "lerobot/aloha_static_coffee",
    index: int = 0,
    embodiment: str = "aloha-bimanual",
    intent_name: str = "brew-coffee",
    phases: tuple[PhaseSegment, ...] = (),
    operator_id: str | None = None,
) -> Episode:
    T, D = 50, 14
    return Episode(
        source=EpisodeSource(repo_id=repo_id, index=index, embodiment=embodiment, fps=50.0),
        n_frames=T,
        dt=1.0 / 50.0,
        joint_positions=np.zeros((T, D), dtype=np.float32),
        joint_actions=np.zeros((T, D), dtype=np.float32),
        ee_velocity_norm=np.zeros(T, dtype=np.float32),
        intent=Intent(name=intent_name, source="repo_metadata"),
        phases=phases,
        operator_id=operator_id,
    )


# --- Add + stats ----------------------------------------------------------


def test_add_episode_creates_expected_nodes_and_edges():
    kg = KnowledgeGraph()
    ep = _make_episode(
        phases=(
            PhaseSegment(name="approach", start_frame=0, end_frame=9, k_lo=0.0, k_hi=0.1),
            PhaseSegment(name="contact", start_frame=10, end_frame=29, k_lo=0.4, k_hi=0.7),
            PhaseSegment(name="retract", start_frame=30, end_frame=49, k_lo=0.0, k_hi=0.1),
        ),
    )
    kg.add(ep)
    s = kg.stats()
    # Episode + Intent + Embodiment + 3 Phases + 3 Skills + 3 Objects + 3 SafetyTags
    assert s["nodes"]["Episode"] == 1
    assert s["nodes"]["Intent"] == 1
    assert s["nodes"]["Embodiment"] == 1
    assert s["nodes"]["Phase"] == 3
    assert s["nodes"]["Skill"] == 4      # fine-bimanual-coordination, place, press-button, insert-pod
    assert s["nodes"]["Object"] == 3     # mug, coffee-machine, filter-pod
    # Coffee REPO_METADATA: contains_liquid + hot_surface + electrical = 3 distinct safety tags
    assert s["nodes"]["SafetyTag"] == 3
    # Edges: 1 INTENT + 1 EMBODIMENT + 3 HAS_PHASE + 3 USES_SKILL + 3 INVOLVES + 3 HAS_SAFETY_CONTEXT
    assert s["edges"]["HAS_INTENT"] == 1
    assert s["edges"]["ON_EMBODIMENT"] == 1
    assert s["edges"]["HAS_PHASE"] == 3
    assert s["edges"]["USES_SKILL"] == 4
    assert s["edges"]["INVOLVES"] == 3
    assert s["edges"]["HAS_SAFETY_CONTEXT"] == 3


def test_add_is_idempotent_on_repeated_episode():
    kg = KnowledgeGraph()
    ep = _make_episode()
    kg.add(ep)
    s1 = kg.stats()
    kg.add(ep)
    s2 = kg.stats()
    assert s1 == s2


def test_add_multiple_episodes_dedupes_shared_nodes():
    kg = KnowledgeGraph()
    kg.add(_make_episode(index=0))
    kg.add(_make_episode(index=1))
    kg.add(_make_episode(index=2))
    s = kg.stats()
    # 3 episodes, but only one of each shared node
    assert s["nodes"]["Episode"] == 3
    assert s["nodes"]["Intent"] == 1
    assert s["nodes"]["Embodiment"] == 1
    assert s["nodes"]["Object"] == 3      # mug, coffee-machine, filter-pod (shared)
    assert s["nodes"]["Skill"] == 4       # shared too: fine-bimanual + place + press-button + insert-pod
    assert s["nodes"]["SafetyTag"] == 3


def test_safety_tag_edge_dedup():
    """Object→SafetyTag edges should not duplicate when multiple episodes
    touch the same object."""
    kg = KnowledgeGraph()
    kg.add(_make_episode(index=0))
    kg.add(_make_episode(index=1))
    s = kg.stats()
    # Still 3 HAS_SAFETY_CONTEXT edges from the 3 objects, not 6
    assert s["edges"]["HAS_SAFETY_CONTEXT"] == 3


# --- Query ----------------------------------------------------------------


def test_query_by_intent():
    kg = KnowledgeGraph()
    ep_a = _make_episode(index=0, intent_name="brew-coffee")
    ep_b = _make_episode(
        index=0,
        repo_id="lerobot/aloha_static_thread_velcro",
        intent_name="thread-velcro",
    )
    kg.add(ep_a)
    kg.add(ep_b)
    coffee = kg.query(intent="brew-coffee")
    velcro = kg.query(intent="thread-velcro")
    assert ep_a.episode_id in coffee and ep_b.episode_id not in coffee
    assert ep_b.episode_id in velcro and ep_a.episode_id not in velcro


def test_query_by_phase_and_k_hat_band():
    kg = KnowledgeGraph()
    high_contact = _make_episode(
        index=0,
        phases=(PhaseSegment(name="contact", start_frame=0, end_frame=49, k_lo=0.6, k_hi=0.9),),
    )
    low_contact = _make_episode(
        index=1,
        phases=(PhaseSegment(name="contact", start_frame=0, end_frame=49, k_lo=0.0, k_hi=0.2),),
    )
    kg.add(high_contact)
    kg.add(low_contact)
    high = kg.query(phase="contact", min_k_hat=0.5)
    low = kg.query(phase="contact", max_k_hat=0.3)
    assert high_contact.episode_id in high and low_contact.episode_id not in high
    assert low_contact.episode_id in low and high_contact.episode_id not in low


def test_query_by_embodiment():
    kg = KnowledgeGraph()
    aloha = _make_episode(index=0)
    koch = _make_episode(
        index=0,
        repo_id="lerobot/koch_pick_place_lego",
        embodiment="koch",
        intent_name="brew-coffee",
    )
    kg.add(aloha)
    kg.add(koch)
    aloha_only = kg.query(embodiment="aloha-bimanual")
    koch_only = kg.query(embodiment="koch")
    assert aloha.episode_id in aloha_only and koch.episode_id not in aloha_only
    assert koch.episode_id in koch_only and aloha.episode_id not in koch_only


# --- Read helpers ---------------------------------------------------------


def test_object_attrs_carry_human_priors():
    kg = KnowledgeGraph()
    kg.add(_make_episode())
    mug = kg.object_attrs("mug")
    assert mug is not None
    assert mug["fragility"] == "moderate"
    assert mug["mass_category"] == "light"
    assert mug["suggested_impedance"] == "gentle"


def test_safety_tags_for_object():
    kg = KnowledgeGraph()
    kg.add(_make_episode())
    tags = set(kg.safety_tags_for_object("mug"))
    assert tags == {"contains_liquid"}
    machine_tags = set(kg.safety_tags_for_object("coffee-machine"))
    assert machine_tags == {"hot_surface", "electrical"}


def test_phases_of_returns_attrs():
    kg = KnowledgeGraph()
    ep = _make_episode(
        phases=(PhaseSegment(name="contact", start_frame=10, end_frame=29, k_lo=0.4, k_hi=0.7),),
    )
    kg.add(ep)
    phases = kg.phases_of(ep.episode_id)
    assert len(phases) == 1
    assert phases[0]["name"] == "contact"
    assert phases[0]["k_lo"] == pytest.approx(0.4)
    assert phases[0]["k_hi"] == pytest.approx(0.7)


# --- Cypher export --------------------------------------------------------


def test_to_cypher_emits_match_return_shape():
    kg = KnowledgeGraph()
    cypher = kg.to_cypher({"intent": "brew-coffee", "phase": "contact", "min_k_hat": 0.5})
    assert cypher.startswith("MATCH ")
    assert "(e:Episode)" in cypher
    assert "[:HAS_INTENT]" in cypher
    assert "[:HAS_PHASE]" in cypher
    assert "WHERE " in cypher
    assert "p.k_hi >= 0.5" in cypher
    assert cypher.rstrip().endswith("RETURN e")


def test_to_cypher_no_filters():
    kg = KnowledgeGraph()
    cypher = kg.to_cypher({})
    assert cypher.startswith("MATCH (e:Episode)")
    assert cypher.rstrip().endswith("RETURN e")
