"""End-to-end integration test on the checked-in offline fixture.

Loads the slim ALOHA-coffee parquet (~8 KB) from ``fixtures/``, constructs
an Episode without touching the network, runs the full ingest pipeline,
and asserts every layer behaved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monty_demo.episode import Episode
from monty_demo.kg import KnowledgeGraph
from monty_demo.reason import ingest, reason
from monty_demo.schemas import EpisodeSource

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "episode_aloha_coffee_004.parquet"


@pytest.fixture
def offline_episode() -> Episode:
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE} — run scripts/build_fixture.py")

    df = pd.read_parquet(FIXTURE)
    positions = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
    actions = np.asarray(df["action"].tolist(), dtype=np.float32)
    T = positions.shape[0]
    fps = 50.0
    dt = 1.0 / fps

    v = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    ee_v = np.concatenate([[v[0]], v]).astype(np.float32) if T >= 2 else np.zeros(T, dtype=np.float32)

    return Episode(
        source=EpisodeSource(
            repo_id="lerobot/aloha_static_coffee",
            index=0,
            embodiment="aloha-bimanual",
            fps=fps,
        ),
        n_frames=T,
        dt=dt,
        joint_positions=positions,
        joint_actions=actions,
        ee_velocity_norm=ee_v,
    )


def test_full_ingest_populates_episode_and_kg(offline_episode):
    kg = KnowledgeGraph()
    populated = ingest(kg, offline_episode)

    # Every layer of the pipeline ran
    assert populated.k_hat is not None
    assert populated.k_hat.shape == (offline_episode.n_frames,)
    assert populated.phases is not None and len(populated.phases) >= 1
    assert populated.intent is not None
    assert populated.intent.name == "brew-coffee"
    assert populated.objects == ("mug", "coffee-machine", "filter-pod")

    # KG reflects it
    assert kg.has_episode(populated.episode_id)
    s = kg.stats()
    assert s["nodes"]["Episode"] == 1
    assert s["nodes"]["Object"] == 3
    assert s["nodes"]["SafetyTag"] == 3   # contains_liquid + hot_surface + electrical
    assert s["edges"]["HAS_SAFETY_CONTEXT"] == 3


def test_reason_returns_non_empty_brief_after_ingest(offline_episode):
    kg = KnowledgeGraph()
    ingest(kg, offline_episode)
    brief = reason(
        kg,
        intent="brew-coffee",
        target_objects=("mug",),
        embodiment="aloha-bimanual",
    )
    assert brief.query_intent == "brew-coffee"
    assert brief.matched_episodes == [offline_episode.episode_id]
    # Human priors should populate object_knowledge and safety_warnings
    assert any(ok.name == "mug" for ok in brief.object_knowledge)
    assert any("liquid" in w.lower() for w in brief.safety_warnings)
    # Mug's gentle prior should dominate the recommended regime
    assert brief.recommended_impedance_regime in ("gentle", "compliant")
    # Confidence > 0 since we have one strict same-embodiment match
    assert brief.confidence > 0.0
