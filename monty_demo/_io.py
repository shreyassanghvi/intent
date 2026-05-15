"""HuggingFace Hub parquet loading for LeRobot datasets.

We deliberately skip the ``lerobot`` library to avoid pulling torch + opencv +
a heavy transitive tree. LeRobot dataset layout (v2) is well-defined:

- ``meta/info.json``: contains ``fps``, ``chunks_size`` (default 1000), and
  ``data_path`` template (e.g. ``"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"``).
- ``data/chunk-XXX/episode_NNNNNN.parquet``: per-episode rows with columns
  ``observation.state`` (joint positions) and ``action`` (commanded positions).

Both as object columns of fixed-length lists/arrays. We materialize them as
contiguous float32 numpy arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from monty_demo._timing import timed


class EpisodeNotFoundError(LookupError):
    def __init__(self, repo_id: str, index: int):
        super().__init__(f"episode index {index} not found in {repo_id!r}")
        self.repo_id = repo_id
        self.index = index


class EpisodeDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _RawEpisode:
    """Raw per-episode payload returned by the loader. Internal — callers see
    the public ``Episode`` from ``episode.py`` instead."""

    repo_id: str
    index: int
    fps: float
    embodiment: str
    joint_positions: np.ndarray  # (T, DOF) float32, observed
    joint_actions: np.ndarray    # (T, DOF) float32, commanded


_DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def _detect_embodiment(repo_id: str) -> str:
    name = repo_id.lower()
    if "aloha" in name:
        return "aloha-bimanual"
    if "koch" in name:
        return "koch"
    if "so100" in name or "so_100" in name:
        return "so100"
    if "moss" in name:
        return "moss"
    if "xarm" in name:
        return "xarm"
    return "unknown"


def _load_info(repo_id: str) -> dict:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
    except Exception as e:
        raise EpisodeDataError(f"could not fetch meta/info.json from {repo_id!r}: {e}") from e
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _column_to_array(series: pd.Series, name: str) -> np.ndarray:
    """Object column of fixed-length list/array → contiguous (T, D) float32."""
    if len(series) == 0:
        raise EpisodeDataError(f"column {name!r} is empty")
    try:
        arr = np.asarray(series.tolist(), dtype=np.float32)
    except Exception as e:
        raise EpisodeDataError(f"column {name!r} could not be coerced to float32 array: {e}") from e
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise EpisodeDataError(f"column {name!r} produced ndim={arr.ndim}, expected 2")
    return np.ascontiguousarray(arr)


@timed("monty_demo._io.load_lerobot_episode")
def load_lerobot_episode(repo_id: str, index: int) -> _RawEpisode:
    """Download (cached) and parse one LeRobot episode shard."""
    info = _load_info(repo_id)
    fps = float(info.get("fps", 30.0))
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_template = info.get("data_path", _DEFAULT_DATA_PATH)

    if index < 0:
        raise EpisodeNotFoundError(repo_id, index)
    total = info.get("total_episodes")
    if isinstance(total, int) and index >= total:
        raise EpisodeNotFoundError(repo_id, index)

    chunk = index // chunks_size
    rel_path = data_path_template.format(episode_chunk=chunk, episode_index=index)
    try:
        parquet_path = hf_hub_download(repo_id=repo_id, filename=rel_path, repo_type="dataset")
    except Exception as e:
        raise EpisodeNotFoundError(repo_id, index) from e

    df = pd.read_parquet(parquet_path)
    if "observation.state" not in df.columns:
        raise EpisodeDataError(
            f"expected column 'observation.state' in {rel_path}, got {list(df.columns)[:6]}..."
        )
    if "action" not in df.columns:
        raise EpisodeDataError(
            f"expected column 'action' in {rel_path}, got {list(df.columns)[:6]}..."
        )

    positions = _column_to_array(df["observation.state"], "observation.state")
    actions = _column_to_array(df["action"], "action")
    if positions.shape != actions.shape:
        raise EpisodeDataError(
            f"shape mismatch: observation.state={positions.shape} vs action={actions.shape}"
        )

    return _RawEpisode(
        repo_id=repo_id,
        index=index,
        fps=fps,
        embodiment=_detect_embodiment(repo_id),
        joint_positions=positions,
        joint_actions=actions,
    )
