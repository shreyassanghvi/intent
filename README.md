# monty_demo — reasoning layer for physical-AI teleop data

A small, opinionated Python SDK that demonstrates a **reasoning substrate** sitting on top of teleop data. Past episodes go in (encoded with impedance estimates, segmented into intent + phase, landed in a small knowledge graph). New tasks come out with **structured priors that bootstrap the next attempt** — and the system gets sharper, better-calibrated, and more discerning every time another attempt lands.

> *"Teach physical AI from human motion — the trajectories AND the world knowledge the human was implicitly using. Past teleop becomes data + priors; every new task starts with both, and the system gets sharper with every attempt."*

This is a pitch artifact for **[usemonty.com](https://usemonty.com)** — *real-world data infrastructure for physical AI*. The full design spec is in [`plan.md`](./plan.md).

---

## What's the headline?

```python
from monty_demo import Episode, KnowledgeGraph, ingest, reason

kg = KnowledgeGraph()

# Past data — three coffee-brewing demonstrations
for i in range(3):
    ingest(kg, Episode.from_lerobot("lerobot/aloha_static_coffee", index=i))

# New task — what should the next attempt look like?
brief = reason(
    kg,
    intent="brew-coffee",
    target_objects=("mug", "coffee-machine"),
    embodiment="aloha-bimanual",
)

print(brief.plan)                          # phase-by-phase suggested plan
print(brief.recommended_impedance_regime)  # merged from data + human priors
print(brief.safety_warnings)               # surfaced from object knowledge
print(brief.confidence)                    # calibrated, can drop with cross-em data
```

Feed one more attempt back via `ingest()` and re-run `reason()` — the brief tightens, gains transferable-skill notes from adjacent tasks, and flags duration outliers when something looks wrong.

---

## Three layers stacked into one pipeline

**1. Impedance encoding (`encode.py`)**
A kinematic-compliance proxy `k_hat ∈ [0, 1]` derived from commanded-vs-observed motion. Vectorized; runs in single-digit milliseconds for a 250-frame × 14-DOF episode. Honest naming: this is *not* calibrated impedance — it's the right signal to extract when you only have kinematics, and the `k_hat` (with hat) name makes the assumption visible at every call site.

**2. Intent + phase + human-priors schema (`intent.py`, `segment.py`)**
Every episode gets a structured intent label, a span-based phase decomposition (approach → contact → manipulate → retract), and — critically — `ObjectKnowledge` priors for every object the operator handled: fragility, mass class, safety context, suggested impedance regime. These are the *human-curated semantic priors* that turn raw motion into something a physical-AI model can learn from.

**3. Knowledge graph + reasoner (`kg.py`, `reason.py`)**
A NetworkX MultiDiGraph (planning-time only — *not* in any control loop) holds Episode / Intent / Phase / Skill / Object / SafetyTag / Embodiment / Operator nodes. The `reason()` function scores past episodes for relevance to a new task spec, aggregates them into a phase plan with stiffness bands, merges those bands with the human's prior knowledge of the target objects, surfaces safety warnings, and emits a `TaskBrief`. A `to_cypher()` exporter signals the migration path to a real graph DB (Kuzu, Neo4j) without paying its dep cost.

---

## Quickstart

```bash
# Setup (Python 3.12+)
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e .[notebook,test]

# Run the tests (offline, ~0.7 s)
pytest -q

# Run the demo notebook (live LeRobot data; ~90 s on first run while HF caches)
jupyter notebook notebooks/demo.ipynb
```

---

## Project layout

```
monty_demo/
  schemas.py    # pydantic v2 contract surface (the only cross-import target)
  episode.py    # frozen dataclass + .from_lerobot(repo_id, index)
  encode.py     # estimate_stiffness() — kinematic-compliance proxy (+ effort fusion when present)
  segment.py    # segment_phases() — vectorized rule-based phase spans
  intent.py     # REPO_METADATA + infer_{intent,skills,objects,object_knowledge}
  kg.py         # KnowledgeGraph + query() + to_cypher() + stats()
  reason.py     # reason() / ingest() / diff_briefs() / print_brief_diff()
  _io.py        # LeRobot v3.0 parquet loader (no lerobot dep)
  _timing.py    # @timed decorator + per-op accumulator
notebooks/
  demo.ipynb    # end-to-end narrative on real ALOHA episodes
tests/
  test_encode_segment.py
  test_kg.py
  test_reason.py
  test_pipeline.py
plan.md         # full design spec
```

---

## Extending the demo to a new LeRobot dataset

Adding a new dataset is a **single edit to `monty_demo/intent.py`** — no other code changes. The schema is the contract; everything else (KG ingestion, reasoning, brief generation, safety surface) reads from it transparently.

```python
# monty_demo/intent.py
REPO_METADATA: dict[tuple[str, int | None], RepoMetadata] = {
    # ... existing entries ...

    # Per-repo default (applies to every episode unless overridden)
    ("lerobot/<your_dataset>", None): RepoMetadata(
        intent=Intent(name="<task-name>", source="repo_metadata"),
        skills=("<skill-1>", "<skill-2>", ...),
        objects=(
            ObjectKnowledge(
                name="<object-name>",
                fragility="moderate",            # see valid values below
                mass_category="light",
                safety_context=["<tag>", ...],   # see recognized tags below
                suggested_impedance="gentle",
            ),
            # ... one ObjectKnowledge per object the operator handled ...
        ),
    ),

    # Optional per-episode override (uses the same shape, applies only to one episode)
    ("lerobot/<your_dataset>", 7): RepoMetadata(
        intent=Intent(name="<different-task>", source="manual", confidence=0.7),
        skills=(...),
        objects=(...),
    ),
}
```

Lookup order is per-episode-specific first, repo-level default second — so adding a per-episode override never breaks the dataset's default behavior.

Then anywhere in your code (or the notebook):
```python
ep = Episode.from_lerobot("lerobot/<your_dataset>", index=0)
ingest(kg, ep)                                     # full pipeline runs
brief = reason(kg, intent="<task-name>", ...)      # brief reflects the new entry
```

### Valid values

| Field | Allowed values |
|---|---|
| `Intent.source` | `"repo_metadata"` (curated), `"rule"` (heuristic), `"manual"` (operator hand-label) |
| `fragility` | `"robust"`, `"moderate"`, `"fragile"`, `"very_fragile"` |
| `mass_category` | `"light"` (< 0.2 kg), `"medium"` (0.2–2 kg), `"heavy"` (> 2 kg) |
| `suggested_impedance` | `"gentle"`, `"compliant"`, `"firm"`, `"stiff"` (each maps to a numeric `(lo, hi)` band in `IMPEDANCE_BANDS`) |
| `safety_context` (recognized — others fall back to generic `"{tag} → caution"`) | `"contains_liquid"`, `"hot_surface"`, `"electrical"`, `"sharp"` |

### Cross-embodiment / hand-labeled entries

If you're tagging a dataset whose intent doesn't actually match the rest of your KG (the way `lerobot/koch_pick_place_5_lego` is hand-labeled `brew-coffee` for the cross-embodiment demo), use:

```python
intent=Intent(name="brew-coffee", source="manual", confidence=0.4),
```

The reasoner doesn't act on `confidence` directly today, but it's there for when the consuming layer wants to weight by provenance. The cross-embodiment penalty (`*0.6` on the score, `*0.85` on the final confidence) handles the "less reliable" property automatically.

---

## Honest framing

A few things this demo *does not* claim, and the README states them up front because the pitch lives or dies on this honesty:

- **`k_hat` is not real impedance.** It's a relative compliance proxy from kinematics. F/T-instrumented data would give actual N/m; this estimator handles the common case where you don't have it.
- **Object knowledge is hand-curated** in `REPO_METADATA`. In production these priors would come from a controlled vocabulary the operator picks at recording time, an object-classifier model, or a task-spec author. The demo's point isn't to *infer* the prior — it's to show that, *given* the prior, the reasoner produces materially better briefs.
- **Cross-embodiment intent labeling** for the Koch dataset entry is also hand-authored. We assert "this is brew-coffee adjacent" and the reasoner appropriately *down-weights* it (cross-embodiment evidence dips confidence, doesn't boost it). That calibration is the point.
- **The KG is NetworkX in-memory.** Production scale wants Kuzu / Neo4j. `kg.to_cypher()` shows the migration path.
- **Three episodes is small** for statistical claims. The demo says "this is a sketch" up front; production would have hundreds.

---

## Performance budgets (visible in the notebook's timing cell)

| Operation | Typical | Notes |
|---|---|---|
| `Episode.from_lerobot` | median ~250 ms, p95 ~520 ms | After HF cache populated; dominated by parquet read + meta-shard walk. Wide tail because the meta walk hits one extra HTTP call when the cache is partial. |
| `estimate_stiffness` | < 1 ms | 250 frames × 14 DOF; effort-fusion path adds a single percentile pass |
| `segment_phases` | < 1 ms | Vectorized state machine over smoothed velocity / tracking-error |
| `kg.add(episode)` | < 1 ms | Single-episode insert into the in-memory MultiDiGraph |
| `reason(...)` | < 5 ms | Over the demo's ~5-episode KG; scales O(\|episodes\|) |
| `ingest(...)` | < 5 ms | End-to-end pipeline (excluding HF download) |

Speed is a claim — the notebook prints actual ms per public op so you can verify. Numbers above are from a fresh `nbconvert --execute` on a Windows laptop.

---

## License

Demo / pitch artifact — please don't redistribute without asking.
