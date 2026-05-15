# Monty Demo — Reasoning Layer for Physical-AI Teleop Data

## Context

This is a pitch artifact for **usemonty.com** — a company building "real-world data infrastructure for physical AI." The demo's purpose is to show that Monty's data layer can be more than passive storage: with the right encoding, schema, and graph structure, past teleop becomes a **reasoning substrate** that bootstraps every new task attempt with structured priors.

The single-sentence pitch the demo earns the right to make:

> *"Teach physical AI from human motion — the trajectories AND the world knowledge the human was implicitly using. Past teleop becomes data + priors; every new task starts with both, and the system gets sharper with every attempt."*

The deliverable is a **Python SDK** (no GUI) exercised by a **single Jupyter notebook** that runs end-to-end on real LeRobot ALOHA episodes in under 90 seconds on a laptop. The notebook is the demo vehicle; the SDK is the actual artifact under evaluation.

**Scope:** ~6–8 hours of focused build (one-day demo).

---

## Goals

1. **Ship a small, opinionated SDK** whose API surface is tight enough that every public symbol is intentional and explainable.
2. **Run the full pipeline on real public data** (LeRobot ALOHA bimanual) — not synthetic — so credibility is immediate.
3. **Make the reasoning layer the headline.** `monty.reason(...) -> TaskBrief` is the call Monty stares at; everything else is supporting machinery.
4. **Demonstrate the self-improvement loop** with three qualitatively different "new attempts" (transfer, calibration, critical sense) so the system visibly gets both more capable and more discerning over time.
5. **Make speed visible.** Every public op self-times; the notebook prints actual ms per call. Speed is a claim — exposing the numbers turns it into evidence.
6. **Capture human priors alongside motion data.** Each object the human handled carries the human's implicit knowledge — fragility, mass class, safety context, suggested impedance regime. The reasoner *merges* this human prior with the data-driven `k_hat` band so briefs reflect both what the data showed AND what the human knew. This is the layer that makes the system *teach* physical AI rather than just *describe* trajectories.

## Non-goals

- No GUI. No dashboards, widgets, or graph viz.
- No learned models. All encoders, segmenters, and reasoners are heuristic — and that's the right call for a 1-day pitch.
- No real graph DB. NetworkX is the storage; a `kg.to_cypher()` exporter signals "we know how this maps to Kuzu/Neo4j" without paying the dep cost.
- No production-grade tests beyond unit + one end-to-end on a checked-in fixture.
- No cross-dataset normalization beyond what the demo specifically needs.

---

## Architecture

**Package: `monty_demo`** (avoids namespace collision with anything Monty might own).

```
monty_demo/
  __init__.py         # exports the public surface
  schemas.py          # pydantic v2 models — the contract surface
  episode.py          # Episode (frozen dataclass) + .from_lerobot(repo_id, index)
  encode.py           # estimate_stiffness() — kinematic-compliance impedance proxy
  segment.py          # segment_phases() — vectorized rule-based segmenter
  intent.py           # infer_intent() — repo-keyed lookup with override hook
  kg.py               # KnowledgeGraph (NetworkX-backed) + query() + to_cypher()
  reason.py           # reason() -> TaskBrief; ingest(); print_brief_diff()
  _io.py              # internal: HF-Hub parquet loading helpers
  _timing.py          # @timed decorator + accumulator; format_timing_table / reset_timings are public
notebooks/
  demo.ipynb          # the narrative — exercises the SDK on real episodes
tests/
  test_encode.py
  test_segment.py
  test_kg.py
  test_reason.py
  test_pipeline.py    # end-to-end on the checked-in fixture
fixtures/
  episode_aloha_coffee_004.parquet   # ~4 KB slice for offline tests
pyproject.toml
README.md
```

### Two design rules the layout enforces

1. **`schemas.py` is the only file that other modules cross-import from each other.** Encoders, segmenters, and KG never import each other directly — they only know about the schemas. Swapping NetworkX for Kuzu later is a one-file change in `kg.py`.
2. **`Episode` is immutable.** Each pipeline step returns a *new* `Episode` with extra fields populated via `dataclasses.replace`. Arrays are shared by reference between immutable variants — no copy on chain step. The fluent chain is safe and any cell can re-run independently.

### Dependencies

- `huggingface_hub` (parquet shard download — *not* the heavy `lerobot` lib)
- `numpy`
- `pandas`
- `networkx`
- `pydantic >= 2`
- `matplotlib` *(notebook only)*
- `pytest` *(tests only)*

Pinned in `pyproject.toml`. No torch, no opencv, no lerobot — install completes in seconds, not minutes.

---

## Data model (`schemas.py` + `episode.py`)

**Speed-first stance:** per-frame data lives as `numpy.ndarray` (float32). Pydantic is for **metadata and KG refs only** — where it earns its weight via type safety on the contract surface, not in the hot loop.

```python
# schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

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
    # k_hat aggregates over this span — populated by encoder/KG layer
    k_lo: float = 0.0
    k_hi: float = 0.0

class Intent(BaseModel):
    name: str
    source: Literal["repo_metadata", "rule", "manual"] = "repo_metadata"
    confidence: float = 1.0

class ObjectKnowledge(BaseModel):
    """Human prior on a single object — what the operator implicitly knew while
    handling it. Combines categorical estimates a human would make at a glance
    (fragility, mass class) with task-relevant safety context and an impedance
    regime hint that captures 'how cautiously did I treat this object'."""
    name: str
    fragility: Literal["robust", "moderate", "fragile", "very_fragile"]
    mass_category: Literal["light", "medium", "heavy"]   # <0.2kg / 0.2-2kg / >2kg
    safety_context: list[str] = Field(default_factory=list)   # e.g. ["contains_liquid", "hot_surface", "electrical", "sharp"]
    suggested_impedance: Literal["gentle", "compliant", "firm", "stiff"]

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
    transferable_skills_observed: list[str] = []   # from adjacent intents
    embodiment_diversity: int = 1
    confidence: float
    notes: list[str]

    # --- human-priors layer (the "what the human knew" surface) ---
    object_knowledge: list[ObjectKnowledge] = []         # priors on every target object
    recommended_impedance_regime: str = "compliant"      # merge of data-driven k_hat band + human priors
    safety_warnings: list[str] = []                      # surfaced from object safety_context tags

class BriefDiff(BaseModel):
    matched_before: int
    matched_after: int
    confidence_delta: float                         # signed — can drop
    tightened_phases: list[str]
    new_skills: list[str]
    new_objects: list[str]
    outlier_phases: list[PhaseOutlier] = []
    new_transferable_skills: list[str] = []
```

```python
# episode.py
from dataclasses import dataclass, field, replace
import numpy as np

@dataclass(frozen=True, slots=True)
class Episode:
    source: EpisodeSource
    n_frames: int
    dt: float                                        # 1/fps, cached

    # Hot-path arrays — float32, contiguous
    joint_positions: np.ndarray                      # (T, DOF) — observed
    joint_actions:   np.ndarray                      # (T, DOF) — commanded
    ee_velocity_norm: np.ndarray | None = None      # (T,) — computed lazily

    # Pipeline-filled (None until that step runs)
    k_hat:   np.ndarray | None = None                # (T,) — estimated stiffness
    phases:  tuple[PhaseSegment, ...] | None = None
    intent:  Intent | None = None

    # KG metadata
    objects:     tuple[str, ...] = ()
    operator_id: str | None = None

    @classmethod
    def from_lerobot(cls, repo_id: str, index: int) -> "Episode": ...

    def with_stiffness(self, k: np.ndarray) -> "Episode":
        return replace(self, k_hat=k)
    def with_phases(self, p: tuple[PhaseSegment, ...]) -> "Episode":
        return replace(self, phases=p)
    def with_intent(self, i: Intent) -> "Episode":
        return replace(self, intent=i)
```

**Critical naming choice:** stiffness is `k_hat` (with hat) everywhere — in code, in plots, in TaskBrief field names. This is a **kinematic compliance proxy**, not measured impedance, and the naming makes that lie impossible to tell.

---

## Pipeline components

### Episode loader (`episode.py`)

`Episode.from_lerobot(repo_id, index)` uses `huggingface_hub.snapshot_download` (cached) to pull the dataset's parquet shard, then materializes one episode by index into float32 numpy arrays. Skips the `lerobot` library — saves a heavy install (torch + opencv) at the cost of ~30 LOC of parquet-parsing per dataset family.

### Impedance encoder (`encode.py`)

```python
def estimate_stiffness(positions, actions, dt) -> np.ndarray:
    """
    Kinematic compliance proxy. k_hat ≈ 1 when commanded motion was resisted,
    k_hat ≈ 0 when motion was free. NOT calibrated to N/m. Use F/T-instrumented
    data for real impedance; this estimator exists for the common case where you
    only have commanded-vs-observed kinematics.
    """
    commanded_motion = np.linalg.norm(np.diff(actions,   axis=0), axis=1)
    observed_motion  = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    eps = 1e-4
    raw = (commanded_motion - observed_motion) / (commanded_motion + eps)
    raw = np.concatenate([[raw[0]], raw])
    return _ema(raw, alpha=0.3).clip(0.0, 1.0).astype(np.float32)
```

Vectorized; in practice < 1 ms for a 250-frame × 14-DOF episode.

### Phase segmenter (`segment.py`)

Vectorized rule-based state machine over `velocity_norm` and `tracking_error = ||actions - positions||`:

- `approach`: high velocity, low tracking error
- `contact`: velocity drops below `v_low`, tracking error > `e_high`
- `manipulate`: moderate velocity, sustained moderate tracking error
- `retract`: velocity rises again, tracking error returns to baseline
- `recover`: optional, on clear correction signal

Returns 3–6 `PhaseSegment` spans per episode. **Output is spans, not per-frame labels** — phases are intervals; downstream code (KG, queries) wants intervals. Target < 2 ms per episode.

### Intent / skill / object-knowledge labeler (`intent.py`)

LeRobot episodes don't carry skill or object metadata, so the demo populates all of it — including the **human-prior `ObjectKnowledge`** for every object the operator handled — from a single curated table keyed by `repo_id`. Same honest-framing as before — `source="repo_metadata"` makes the provenance explicit at every call site, and the per-object `ObjectKnowledge` is exactly the kind of common-sense annotation an operator would jot down without thinking.

```python
@dataclass(frozen=True)
class RepoMetadata:
    intent: Intent
    skills:  tuple[str, ...]
    objects: tuple[ObjectKnowledge, ...]

REPO_METADATA: dict[str, RepoMetadata] = {
    "lerobot/aloha_static_coffee": RepoMetadata(
        intent=Intent(name="brew-coffee", source="repo_metadata"),
        skills=("fine-bimanual-coordination", "place", "press-button"),
        objects=(
            ObjectKnowledge(name="mug", fragility="moderate", mass_category="light",
                            safety_context=["contains_liquid"], suggested_impedance="gentle"),
            ObjectKnowledge(name="coffee-machine", fragility="robust", mass_category="heavy",
                            safety_context=["hot_surface", "electrical"], suggested_impedance="firm"),
            ObjectKnowledge(name="filter-pod", fragility="fragile", mass_category="light",
                            safety_context=[], suggested_impedance="gentle"),
        ),
    ),
    "lerobot/aloha_static_thread_velcro": RepoMetadata(
        intent=Intent(name="thread-velcro", source="repo_metadata"),
        skills=("fine-bimanual-coordination", "thread", "pinch-grasp"),
        objects=(
            ObjectKnowledge(name="velcro-strap", fragility="robust", mass_category="light",
                            safety_context=[], suggested_impedance="firm"),
            ObjectKnowledge(name="cloth", fragility="moderate", mass_category="light",
                            safety_context=[], suggested_impedance="compliant"),
        ),
    ),
    "lerobot/koch_pick_place_5_lego": RepoMetadata(   # cross-embodiment hand-label
        intent=Intent(name="brew-coffee", source="manual", confidence=0.4),
        skills=("place", "pick"),
        objects=(
            ObjectKnowledge(name="mug", fragility="moderate", mass_category="light",
                            safety_context=["contains_liquid"], suggested_impedance="gentle"),
        ),
    ),
    # one more entry for attempt 3's outlier episode (same repo as setup)
}

def infer_intent(ep: Episode) -> Intent: ...
def infer_skills(ep: Episode) -> tuple[str, ...]: ...
def infer_object_knowledge(ep: Episode) -> tuple[ObjectKnowledge, ...]: ...
def infer_objects(ep: Episode) -> tuple[str, ...]:
    """Names only — convenience for the KG INVOLVES edge."""
    return tuple(ok.name for ok in infer_object_knowledge(ep))
```

`Episode.objects` is populated at ingest time via `infer_objects(ep)`. The full `ObjectKnowledge` records are written into the KG as `Object` node attrs (fragility, mass_category, suggested_impedance) and as `SafetyTag` nodes connected via `HAS_SAFETY_CONTEXT` edges so the reasoner can do "find episodes touching liquid-bearing objects" in a single graph hop.

Honest framing in docstring: in production each of intent/skills/object-knowledge would come from a model (text-from-prompt, video classifier, operator-tagged) or a controlled vocabulary the operator picks from at recording time; for the demo it's curated lookup so the reasoner's behavior is fully deterministic and inspectable. The point of the demo isn't to *infer* the human prior — it's to show that, *given* the prior, the reasoner produces materially better briefs.

---

## KG (`kg.py`) — planning-time, not runtime

NetworkX-backed in-memory `MultiDiGraph`. Explicitly framed in the README and notebook as **planning-time** — for curriculum design, dataset assembly, retrieving similar episodes — *not* in any inner control loop.

### Nodes

| Type | id format | attrs |
|---|---|---|
| Episode | `ep:{repo_id}/{index}` | `n_frames, fps, ingested_at` |
| Intent | `intent:{name}` | `name` |
| Phase | `phase:{episode_id}/{name}/{start_frame}` | `name, start_frame, end_frame, k_lo, k_hi, confidence` |
| Skill | `skill:{name}` | `name` |
| Object | `obj:{name}` | `name, fragility, mass_category, suggested_impedance` |
| SafetyTag | `safety:{tag}` | `tag` (e.g. `"contains_liquid"`, `"hot_surface"`) |
| Operator | `op:{id}` | `id` |
| Embodiment | `emb:{name}` | `name` |

### Edges

Typed via `kind` attribute on the edge: `HAS_INTENT`, `HAS_PHASE`, `USES_SKILL`, `INVOLVES`, `DEMONSTRATED_BY`, `ON_EMBODIMENT` (all originate at `Episode` nodes), plus `HAS_SAFETY_CONTEXT` (`Object → SafetyTag`) so the reasoner can hop from a target object to all safety tags in one query.

Phase nodes are first-class (not edge attrs) because the reasoner aggregates over them by name across episodes — needs `Intent → Episode → Phase{name=contact}` traversal.

### Public surface

```python
class KnowledgeGraph:
    def add(self, ep: Episode) -> None: ...                  # idempotent on (repo_id, index)
    def query(self, *, intent=None, phase=None, embodiment=None,
              min_k_hat=None, max_k_hat=None) -> list[str]:  # episode IDs
        ...
    def to_cypher(self, query_kwargs: dict) -> str:          # equivalent Cypher string
        ...
    def stats(self) -> dict:                                 # node/edge counts by type
        ...
```

`to_cypher()` is the "we-know-this-maps-to-a-real-graph-DB" hook. Costs ~30 LOC, signals fluency.

---

## Reasoning layer (`reason.py`) — the headline

### `reason()` — the front door

```python
def reason(
    kg: KnowledgeGraph,
    *,
    intent: str,
    target_objects: tuple[str, ...] = (),
    embodiment: str | None = None,
    k: int = 5,
) -> TaskBrief:
    ...
```

**Pipeline inside `reason()`:**

1. **Retrieve.** Score every Episode node by
   `1.0·intent_match + 0.5·object_jaccard(target_objects, ep.objects) + 0.7·embodiment_match`,
   with a **0.6× cross-embodiment penalty** when embodiment differs from query's. Top-K. Single NetworkX traversal pass — O(|episodes|), target < 50 ms.

2. **Adjacency.** Also collect episodes that share ≥1 phase pattern or ≥1 skill with the top-K's intent — these don't count as matches but feed `transferable_skills_observed`.

3. **Plan.** For each `PhaseName` appearing in ≥ ⌈K/2⌉ of top-K: emit a `PhasePlan` with median duration and 25th/75th percentile k_hat band (drawn from Phase nodes' `k_lo`/`k_hi` attrs).

4. **Skills.** Union of skills across top-K, ranked by frequency.

5. **Objects-seen-before.** Intersection of `target_objects` with objects across top-K.

6. **Confidence.** `min(1.0, 0.2·n_strict + 0.5·(1 - mean(stiffness_band_width)))`, where `n_strict` is the count of *same-embodiment* same-intent matches (cross-embodiment matches still contribute to `matched_episodes` but do *not* earn the confidence boost). Cross-embodiment matches additionally trigger a `*0.85` dip on the final confidence. Drops when fewer strict matches survive or when stiffness bands are wider/inconsistent. **Confidence can decrease when more data is added** — see calibration attempt below.

7. **Apply human priors.** For every object in `target_objects`, look up its `ObjectKnowledge` from the KG (or `REPO_METADATA` directly). Then:
   - **Merge impedance.** Map each `suggested_impedance` to a numeric band — `gentle: (0.0, 0.45)`, `compliant: (0.3, 0.65)`, `firm: (0.5, 0.85)`, `stiff: (0.7, 1.0)`. Take the *intersection* of these per-object prior bands with the data-driven `k_hat` band derived in step 3. The intersection is the `recommended_impedance_regime`; emit it as one of the four labels by which band it falls in. If the intersection is empty (data and human disagree), keep the prior and emit a note: `"data-driven k_hat (X, Y) exceeds human-prior 'gentle' band — investigate; safety priors win for now"`.
   - **Surface safety.** Union of all `safety_context` tags across target objects → `safety_warnings`. Each tag becomes a human-readable line ("contains_liquid → spill risk during contact phase").
   - **Attach knowledge.** Copy the per-object `ObjectKnowledge` records into `TaskBrief.object_knowledge` so the brief is self-contained.

8. **Notes.** Human-readable strings derived from above (priors merge, outliers, transferable skills, cross-embodiment hedging).

### `ingest()` — the closing call of the loop

```python
def ingest(kg: KnowledgeGraph, ep: Episode) -> Episode:
    """Encode → segment → infer-intent → outlier-check → KG.add. Returns populated Episode."""
```

**Idempotent** on `(repo_id, index)`; re-ingest is a no-op with a logged warning.

Outlier check on ingest: for each new phase, compute z-scores against existing phases of the same name across the KG. Flag z ≥ 2.0. Cheap — single pass per phase.

### `print_brief_diff()` — the money shot helper

Side-by-side print of `BriefDiff`:
```
matched_episodes:        3 → 4
contact stiffness band:  (0.41, 0.78) → (0.48, 0.74)   [tightened]
confidence:              0.52 → 0.61
new transferable skills: ['fine-bimanual-coordination']
outlier phases:          contact (z=2.4) on ep:.../coffee_009 [warning]
```

---

## The self-improvement loop — three attempts with teeth

Setup: 3 prior ingests from `lerobot/aloha_static_coffee` → KG has brew-coffee episodes on aloha-bimanual with mug + coffee-machine.

### Attempt 1 — cross-skill transfer
Ingest one episode from `lerobot/aloha_static_battery`. Different objects (battery, battery-slot), different intent label (`insert-battery`), but shares `fine-bimanual-coordination` skill structure with brew-coffee — and crucially brings two new transferable skills (`precision-insert`, `align-and-press`) that map onto the existing `insert-pod` step in the brew-coffee plan.

**Expected reasoner behavior:**
- Does *not* increase `matched_episodes` for `intent="brew-coffee"`.
- Adds `precision-insert` and `align-and-press` to `transferable_skills_observed` in the brief.
- The two new transferable skills are candidate refinements to the existing `insert-pod` step.

**Capability proven:** the reasoner does more than label-matching.

### Attempt 2 — cross-embodiment calibration
Ingest one episode from a Koch single-arm dataset (e.g., `lerobot/koch_pick_place_5_lego`), manually labeled with a coffee-adjacent intent.

**Expected reasoner behavior:**
- Increases `matched_episodes` for brew-coffee but with reduced per-match weight (0.6× cross-embodiment penalty).
- Widens `embodiment_diversity` to 2.
- **Confidence dips slightly** — counter-intuitive but correct. Cross-embodiment evidence is less reliable; a reasoner that blindly grows confidence with more data is broken.

**Capability proven:** the reasoner has calibration.

### Attempt 3 — outlier detection
Ingest one more `aloha_static_coffee` episode whose contact phase is >2σ longer than the existing median (a fumbled or partially-failed attempt).

**Expected reasoner behavior:**
- BriefDiff contains `outlier_phases: [PhaseOutlier(..., metric="duration", z_score=2.4, severity="warning")]`.
- Note: `"newly ingested episode has anomalous contact-phase duration — investigate before training on it"`.

**Capability proven:** the reasoner has critical sense — doesn't trust new data blindly.

Together: **transfer + calibration + critical sense.** Hard to dismiss as "just averaging."

---

## Notebook narrative (`notebooks/demo.ipynb`)

Single notebook, ~12 cells, runs end-to-end in < 90 s on a laptop after first cache.

1. **Title + 4-line pitch** (markdown)
2. **Install + imports** — the whole API surface visible in one cell
3. **Ingest 3 brew-coffee episodes** — single `monty.ingest()` per loop iteration
4. **Inspect KG state** — `kg.stats()` print + a small subgraph render via networkx
5. **`brief_before = monty.reason(intent="brew-coffee", embodiment="aloha-bimanual")`** — print the TaskBrief
6. **Attempt 1 (cross-skill):** ingest battery-insertion episode, re-reason, show new `transferable_skills_observed` field
7. **Attempt 2 (cross-embodiment):** ingest Koch episode, re-reason, show confidence *drop* with explanation
8. **Attempt 3 (outlier):** ingest weird-contact coffee episode, show `outlier_phases` warning
9. **Cumulative diff** — `print_brief_diff(brief_before, brief_final)` — the money shot
10. **Human priors in action** — query a brief whose target object is `mug` with `safety_context=["contains_liquid"]`. Print the data-driven `k_hat` band, the human-prior `suggested_impedance` band, and the merged `recommended_impedance_regime`. Show the `safety_warnings`. The point: human knowledge tightens the brief beyond what pure data could.
11. **Cypher export** — `print(kg.to_cypher({"intent": "brew-coffee"}))` to signal "this maps to a real graph DB"
12. **Timing table** — actual ms per public op on this run
13. **"What this enables"** — 3 bullets, no fluff:
    - Every new task starts with retrieved priors instead of cold.
    - Cross-task transfer is structural, not just label-based.
    - Bad new data is flagged before it pollutes downstream training.

---

## Performance budgets (visible in the timing cell)

| Operation | Target | Notes |
|---|---|---|
| `Episode.from_lerobot` | < 200 ms | After HF cache populated |
| `estimate_stiffness` | < 5 ms | 250 frames × 14 DOF |
| `segment_phases` | < 2 ms | Vectorized state machine |
| `infer_intent` | < 0.1 ms | Dict lookup |
| `kg.add(episode)` | < 1 ms | Single-episode insert |
| `kg.query(...)` | < 50 ms | Over ~100 episodes |
| `reason(...)` | < 100 ms | Including retrieve + plan + adjacency |
| `ingest(...)` end-to-end | < 10 ms | Excluding HF download |

A `_timing.py` decorator accumulates per-op timings; the notebook's timing cell prints them as a table.

---

## Error handling

- `Episode.from_lerobot(repo_id, index)` raises `EpisodeNotFoundError(repo_id, index)` on bad index. Missing expected columns → `EpisodeDataError("expected column X, got Y")`. **No silent fallbacks.**
- `reason(...)` with zero matches returns a `TaskBrief` with `matched_episodes=[]`, `confidence=0.0`, and a note `"no prior data — cold start"`. *Empty brief is a valid answer, not an error.*
- `ingest(...)` is idempotent on `(repo_id, index)`; re-ingest is a no-op with a logged warning.
- All schema validation errors propagate naturally from pydantic — no custom wrapping.

---

## Testing

All tests run in < 5 s total via `pytest`. Four files keep coverage tight without ceremony.

- **`test_encode_segment.py`** — stiffness bounds + monotonicity on synthetic arrays; segmenter returns expected segment count and ordering on synthetic velocity/error traces.
- **`test_kg.py`** — add 3 synthetic episodes (with metadata); query each edge type; verify counts; verify idempotency on re-add; verify `to_cypher()` emits a syntactically-shaped `MATCH ... RETURN ...` string.
- **`test_reason.py`** — covers reasoner + outlier + priors together: (a) ingest 2 same-intent synthetic episodes then a 3rd; verify confidence rises and `BriefDiff.tightened_phases` contains `"contact"`. (b) Ingest a 4th whose contact phase is 2.5σ longer than baseline; verify `BriefDiff.outlier_phases` contains a `PhaseOutlier` with `severity="warning"`. (c) Ingest a cross-embodiment episode; verify confidence drops despite `n_matched` rising. (d) Reason about a task with a fragile/liquid-bearing object; verify `recommended_impedance_regime` is tightened compared to pure data-driven, `safety_warnings` includes `"contains_liquid"`-derived note, and `object_knowledge` is populated.
- **`test_pipeline.py`** — full `ingest()` on `fixtures/episode_aloha_coffee_004.parquet` (checked-in ~4 KB slice); assert `Episode` is fully populated, KG has expected node/edge counts, and `reason(intent="brew-coffee")` returns a non-empty TaskBrief.

The fixture is generated once by a separate `scripts/build_fixture.py` (not part of the test path) so tests are fully offline.

---

## Verification (how to confirm the demo works end-to-end)

After implementation:

1. **Install:** `pip install -e .` from the project root. Should complete in seconds (no torch).
2. **Tests:** `pytest -q` — all green in < 5 s.
3. **Notebook smoke:** `jupyter nbconvert --to notebook --execute notebooks/demo.ipynb --output demo.executed.ipynb` — completes without errors in < 90 s after first HF cache.
4. **Manual review of the executed notebook:**
   - Cell 5 prints a TaskBrief with ≥ 3 matched episodes and confidence in `(0.4, 0.7)`.
   - Cell 6's brief shows a non-empty `transferable_skills_observed`.
   - Cell 7's brief shows `confidence` *lower* than cell 6's (calibration check).
   - Cell 8's BriefDiff contains a `PhaseOutlier` with `z_score >= 2.0`.
   - Cell 11's timing table shows every op within budget.
5. **Cypher export sanity check:** the string emitted by `kg.to_cypher(...)` is syntactically valid Cypher (eyeball: `MATCH ... RETURN ...` shape with proper variable bindings).

---

## Critical files to create

| File | Purpose | Approx LOC |
|---|---|---|
| `monty_demo/schemas.py` | All pydantic models | ~80 |
| `monty_demo/episode.py` | Episode dataclass + loader | ~120 |
| `monty_demo/encode.py` | `estimate_stiffness` + EMA helper | ~40 |
| `monty_demo/segment.py` | Vectorized phase segmenter | ~80 |
| `monty_demo/intent.py` | Lookup table (intent + skills + ObjectKnowledge) + `infer_*` | ~110 |
| `monty_demo/kg.py` | KnowledgeGraph + query + to_cypher (incl. SafetyTag nodes) | ~170 |
| `monty_demo/reason.py` | reason + ingest + diff helpers + outlier check + human-prior merge | ~220 |
| `monty_demo/_io.py` | HF parquet loading | ~50 |
| `monty_demo/_timing.py` | @timed decorator + accumulator | ~30 |
| `monty_demo/__init__.py` | Re-exports | ~20 |
| `notebooks/demo.ipynb` | The demo vehicle | ~12 cells |
| `tests/*.py` | 4 test files | ~200 total |
| `scripts/build_fixture.py` | One-off fixture generator | ~40 |
| `pyproject.toml` | Pinned deps | — |
| `README.md` | The pitch + how to run | ~80 |

Total: ~1,300 LOC of source + ~220 of tests + 1 notebook. Comfortable in a 7–9 hour build (+~1.5 hrs vs prior estimate for the human-priors layer).

---

## Build sequence (recommended order)

1. `pyproject.toml`, `__init__.py`, `schemas.py` — establish the contract surface first.
2. `_io.py` + `episode.py` — load one real ALOHA episode end-to-end; eyeball numpy shapes.
3. `_timing.py` — add the decorator; bake timing into every public function from the start.
4. `encode.py` + `segment.py` + `test_encode_segment.py` — k_hat and phases on the loaded episode.
5. `intent.py` — populate the `REPO_METADATA` table with the 4 episodes the demo uses.
6. `kg.py` + `test_kg.py` — add nodes, query, `to_cypher` last.
7. `reason.py` — `reason()` first, then `ingest()` wrapping the pipeline, then outlier detection.
8. `test_reason.py` — covers reason + outlier + cross-embodiment together.
9. `notebooks/demo.ipynb` — last; the narrative is easier to write once everything works.
10. `README.md` — the pitch sentence + how-to-run.
11. `scripts/build_fixture.py` + `test_pipeline.py` — final integration test on fixture.

---

## Risks & known limitations (called out honestly)

- **Cross-embodiment intent labeling is hand-authored** for the Koch episode. We're choosing one and asserting "this is brew-coffee adjacent." Documented in the README — not hidden.
- **`k_hat` is not real impedance.** Stated in every docstring, plot label, and brief field name. The honest framing is part of the pitch, not a weakness.
- **Outlier detection is z-score on duration only**, not on k_hat profile shape or phase ordering. Could be extended; for 1-day scope this is enough to demonstrate the capability.
- **3-episode initial KG is small** for statistical claims. The demo says "this is a sketch" up front; production would have hundreds.
- **Human priors are hand-curated** in `REPO_METADATA`. In production they'd come from a controlled vocabulary picked by the operator at recording time, an object-classifier model, or a task-spec the operator wrote. The demo is *not* about inferring priors — it's about showing the value of *having* them once they exist. Documented in the README.

---

## Post-plan-mode followups (queued)

1. Copy this plan to project root: `C:\Users\shrey\PycharmProjects\intent\` (per user rule).
2. Update `CLAUDE.md` with the rule "always write plans to project root."
3. Save memories: (a) speed is a first-class constraint in robotics work, (b) KG belongs at planning-time not runtime, (c) the system is a reasoning layer that learns from past data to help next tasks.
