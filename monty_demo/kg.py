"""KnowledgeGraph — NetworkX MultiDiGraph, planning-time only.

This is the substrate the reasoner queries between attempts. Explicitly NOT
in any control loop. Latency budget is generous (10–100 ms is fine) compared
to the encoder/segmenter hot path (single-digit ms).

Nodes are typed via the ``kind`` attribute. Edges are typed via the ``kind``
attribute on each edge. ``Object`` nodes carry the human-prior ``ObjectKnowledge``
attrs (fragility, mass_category, suggested_impedance); per-object ``safety_context``
tags become first-class ``SafetyTag`` nodes connected by ``HAS_SAFETY_CONTEXT``
edges so the reasoner can hop from a target object to all its safety tags in
one query.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import networkx as nx

from monty_demo._timing import timed
from monty_demo.episode import Episode
from monty_demo.intent import infer_object_knowledge, infer_skills

# Node-id prefixes — keyed lookup is O(1) and keeps the graph human-readable
# when printed (`ep:lerobot/aloha_static_coffee/4`, `intent:brew-coffee`, ...).
_EP = "ep"
_INT = "intent"
_PHASE = "phase"
_SKILL = "skill"
_OBJ = "obj"
_SAFETY = "safety"
_OP = "op"
_EMB = "emb"


class KnowledgeGraph:
    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # --- Construction ----------------------------------------------------

    def _ensure(self, node_id: str, kind: str, **attrs: Any) -> None:
        if node_id not in self._g.nodes:
            self._g.add_node(node_id, kind=kind, **attrs)

    def _ensure_unique_edge(self, src: str, dst: str, kind: str) -> None:
        """For edges that semantically should be unique (e.g. HAS_SAFETY_CONTEXT
        between an object and a tag) — MultiDiGraph allows duplicates so we
        check first."""
        for _, v, attrs in self._g.out_edges(src, data=True):
            if v == dst and attrs.get("kind") == kind:
                return
        self._g.add_edge(src, dst, kind=kind)

    @timed("monty_demo.KnowledgeGraph.add")
    def add(self, ep: Episode) -> None:
        """Add an Episode and all its derived nodes/edges. Idempotent on
        ``(repo_id, index)`` — re-adding the same episode is a no-op."""
        ep_id = ep.episode_id
        if ep_id in self._g.nodes:
            return  # idempotent

        self._g.add_node(
            ep_id,
            kind="Episode",
            n_frames=ep.n_frames,
            fps=ep.source.fps,
            ingested_at=time.time(),
        )

        # Embodiment — always present
        emb_id = f"{_EMB}:{ep.source.embodiment}"
        self._ensure(emb_id, "Embodiment", name=ep.source.embodiment)
        self._g.add_edge(ep_id, emb_id, kind="ON_EMBODIMENT")

        # Intent (set by pipeline before .add())
        if ep.intent is not None:
            int_id = f"{_INT}:{ep.intent.name}"
            self._ensure(int_id, "Intent", name=ep.intent.name)
            self._g.add_edge(ep_id, int_id, kind="HAS_INTENT")

        # Phases (set by pipeline; k_lo/k_hi already aggregated by ingest())
        if ep.phases is not None:
            for p in ep.phases:
                phase_id = f"{_PHASE}:{ep_id}/{p.name}/{p.start_frame}"
                self._g.add_node(
                    phase_id,
                    kind="Phase",
                    name=p.name,
                    start_frame=p.start_frame,
                    end_frame=p.end_frame,
                    duration_s=(p.end_frame - p.start_frame + 1) * ep.dt,
                    k_lo=p.k_lo,
                    k_hi=p.k_hi,
                    confidence=p.confidence,
                )
                self._g.add_edge(ep_id, phase_id, kind="HAS_PHASE")

        # Skills — sourced from REPO_METADATA
        for skill in infer_skills(ep):
            sk_id = f"{_SKILL}:{skill}"
            self._ensure(sk_id, "Skill", name=skill)
            self._g.add_edge(ep_id, sk_id, kind="USES_SKILL")

        # Objects + safety tags — sourced from REPO_METADATA (full ObjectKnowledge)
        for ok in infer_object_knowledge(ep):
            obj_id = f"{_OBJ}:{ok.name}"
            self._ensure(
                obj_id,
                "Object",
                name=ok.name,
                fragility=ok.fragility,
                mass_category=ok.mass_category,
                suggested_impedance=ok.suggested_impedance,
            )
            self._g.add_edge(ep_id, obj_id, kind="INVOLVES")
            for tag in ok.safety_context:
                tag_id = f"{_SAFETY}:{tag}"
                self._ensure(tag_id, "SafetyTag", tag=tag)
                self._ensure_unique_edge(obj_id, tag_id, "HAS_SAFETY_CONTEXT")

        # Operator (optional)
        if ep.operator_id is not None:
            op_id = f"{_OP}:{ep.operator_id}"
            self._ensure(op_id, "Operator", id=ep.operator_id)
            self._g.add_edge(ep_id, op_id, kind="DEMONSTRATED_BY")

    # --- Read-side surface used by the reasoner --------------------------

    def episode_ids(self) -> list[str]:
        return [n for n, a in self._g.nodes(data=True) if a.get("kind") == "Episode"]

    def has_episode(self, ep_id: str) -> bool:
        return ep_id in self._g.nodes and self._g.nodes[ep_id].get("kind") == "Episode"

    def episode_attrs(self, ep_id: str) -> dict:
        return dict(self._g.nodes[ep_id])

    def _neighbors_of_kind(self, ep_id: str, edge_kind: str) -> list[str]:
        return [
            v
            for _, v, attrs in self._g.out_edges(ep_id, data=True)
            if attrs.get("kind") == edge_kind
        ]

    def _neighbor_attrs(self, ep_id: str, edge_kind: str, attr: str) -> list[str]:
        return [self._g.nodes[n].get(attr) for n in self._neighbors_of_kind(ep_id, edge_kind)]

    def intent_of(self, ep_id: str) -> str | None:
        names = self._neighbor_attrs(ep_id, "HAS_INTENT", "name")
        return names[0] if names else None

    def embodiment_of(self, ep_id: str) -> str | None:
        names = self._neighbor_attrs(ep_id, "ON_EMBODIMENT", "name")
        return names[0] if names else None

    def skills_of(self, ep_id: str) -> list[str]:
        return self._neighbor_attrs(ep_id, "USES_SKILL", "name")

    def objects_of(self, ep_id: str) -> list[str]:
        return self._neighbor_attrs(ep_id, "INVOLVES", "name")

    def phases_of(self, ep_id: str) -> list[dict]:
        """Return phase-node attribute dicts for an episode."""
        return [dict(self._g.nodes[n]) for n in self._neighbors_of_kind(ep_id, "HAS_PHASE")]

    def safety_tags_for_object(self, object_name: str) -> list[str]:
        obj_id = f"{_OBJ}:{object_name}"
        if obj_id not in self._g.nodes:
            return []
        return [
            self._g.nodes[v].get("tag")
            for _, v, attrs in self._g.out_edges(obj_id, data=True)
            if attrs.get("kind") == "HAS_SAFETY_CONTEXT"
        ]

    def object_attrs(self, object_name: str) -> dict | None:
        obj_id = f"{_OBJ}:{object_name}"
        if obj_id not in self._g.nodes:
            return None
        return dict(self._g.nodes[obj_id])

    # --- Query API -------------------------------------------------------

    @timed("monty_demo.KnowledgeGraph.query")
    def query(
        self,
        *,
        intent: str | None = None,
        phase: str | None = None,
        embodiment: str | None = None,
        min_k_hat: float | None = None,
        max_k_hat: float | None = None,
    ) -> list[str]:
        """Filter Episode IDs by intent / phase-with-k-band / embodiment.

        ``min_k_hat`` / ``max_k_hat`` only take effect when ``phase`` is given,
        and they bound the phase's *band* (k_hi >= min, k_lo <= max).
        """
        results: list[str] = []
        for ep_id in self.episode_ids():
            if intent is not None and self.intent_of(ep_id) != intent:
                continue
            if embodiment is not None and self.embodiment_of(ep_id) != embodiment:
                continue
            if phase is not None:
                phases = [p for p in self.phases_of(ep_id) if p.get("name") == phase]
                if not phases:
                    continue
                if min_k_hat is not None:
                    phases = [p for p in phases if p.get("k_hi", 0.0) >= min_k_hat]
                if max_k_hat is not None:
                    phases = [p for p in phases if p.get("k_lo", 0.0) <= max_k_hat]
                if not phases:
                    continue
            results.append(ep_id)
        return results

    # --- Cypher export ---------------------------------------------------

    def to_cypher(self, query_kwargs: dict) -> str:
        """Emit the Cypher equivalent of a ``query()`` call.

        Signals "this maps to a real graph DB (Kuzu, Neo4j) when you outgrow
        in-memory NetworkX." Costs ~30 LOC; real value is in the pitch.
        """
        matches: list[str] = ["(e:Episode)"]
        wheres: list[str] = []

        if query_kwargs.get("intent") is not None:
            matches.append(f"(e)-[:HAS_INTENT]->(i:Intent {{name: '{query_kwargs['intent']}'}})")
        if query_kwargs.get("embodiment") is not None:
            matches.append(
                f"(e)-[:ON_EMBODIMENT]->(em:Embodiment {{name: '{query_kwargs['embodiment']}'}})"
            )
        if query_kwargs.get("phase") is not None:
            matches.append(f"(e)-[:HAS_PHASE]->(p:Phase {{name: '{query_kwargs['phase']}'}})")
            if query_kwargs.get("min_k_hat") is not None:
                wheres.append(f"p.k_hi >= {query_kwargs['min_k_hat']}")
            if query_kwargs.get("max_k_hat") is not None:
                wheres.append(f"p.k_lo <= {query_kwargs['max_k_hat']}")

        cypher = "MATCH " + ", ".join(matches)
        if wheres:
            cypher += "\nWHERE " + " AND ".join(wheres)
        cypher += "\nRETURN e"
        return cypher

    # --- Stats -----------------------------------------------------------

    def stats(self) -> dict:
        node_counts: dict[str, int] = {}
        for _, attrs in self._g.nodes(data=True):
            kind = attrs.get("kind", "?")
            node_counts[kind] = node_counts.get(kind, 0) + 1
        edge_counts: dict[str, int] = {}
        for _, _, attrs in self._g.edges(data=True):
            kind = attrs.get("kind", "?")
            edge_counts[kind] = edge_counts.get(kind, 0) + 1
        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": self._g.number_of_nodes(),
            "total_edges": self._g.number_of_edges(),
        }

    # --- Iterators (used by reasoner adjacency search) -------------------

    def iter_phase_nodes_by_name(self, phase_name: str) -> Iterable[dict]:
        for n, attrs in self._g.nodes(data=True):
            if attrs.get("kind") == "Phase" and attrs.get("name") == phase_name:
                yield dict(attrs)
