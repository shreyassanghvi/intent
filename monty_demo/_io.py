"""HuggingFace Hub parquet loading for LeRobot datasets (v3.0 layout).

We deliberately skip the ``lerobot`` library to avoid pulling torch + opencv +
a heavy transitive tree. LeRobot v3.0 layout is two-tier:

- ``meta/info.json`` carries ``fps``, ``chunks_size``, and a ``data_path``
  template that uses ``{chunk_index}`` / ``{file_index}`` (e.g.
  ``"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"``).
- ``meta/episodes/chunk-XXX/file-XXX.parquet`` is an *episode index* — one row
  per episode with columns ``data/chunk_index``, ``data/file_index``,
  ``dataset_from_index``, ``dataset_to_index``, ``length``.
- ``data/chunk-XXX/file-XXX.parquet`` is the actual frame data, multiple
  episodes concatenated; each episode lives in rows
  ``[dataset_from_index : dataset_to_index]``.

Frame columns of interest: ``observation.state`` (joint positions),
``action`` (commanded positions), and on some embodiments ``observation.effort``
(joint torque — a real impedance signal we don't yet exploit but worth noting).
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


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_META_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


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


def _load_episode_meta(repo_id: str, index: int, chunks_size: int) -> dict:
    """Locate the per-episode meta row, walking meta/episodes shards. For
    typical demo datasets (≤ chunks_size episodes total) this hits one file."""
    meta_chunk = 0
    meta_file = 0
    while True:
        rel = _META_EPISODES_PATH.format(chunk_index=meta_chunk, file_index=meta_file)
        try:
            path = hf_hub_download(repo_id=repo_id, filename=rel, repo_type="dataset")
        except Exception as e:
            raise EpisodeNotFoundError(repo_id, index) from e
        df = pd.read_parquet(path)
        match = df[df["episode_index"] == index]
        if not match.empty:
            row = match.iloc[0]
            return {
                "data_chunk_index": int(row["data/chunk_index"]),
                "data_file_index": int(row["data/file_index"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "length": int(row["length"]),
            }
        # Walk to the next meta shard if this dataset has more
        if df["episode_index"].max() < index:
            meta_file += 1
            if meta_file >= chunks_size:
                meta_file = 0
                meta_chunk += 1
                if meta_chunk > 100:   # sanity bound for demo
                    raise EpisodeNotFoundError(repo_id, index)
            continue
        raise EpisodeNotFoundError(repo_id, index)


@timed("monty_demo._io.load_lerobot_episode")
def load_lerobot_episode(repo_id: str, index: int) -> _RawEpisode:
    """Download (cached) and parse one LeRobot v3.0 episode."""
    info = _load_info(repo_id)
    fps = float(info.get("fps", 30.0))
    chunks_size = int(info.get("chunks_size", 1000))
    data_path_template = info.get("data_path", _DEFAULT_DATA_PATH)

    if index < 0:
        raise EpisodeNotFoundError(repo_id, index)
    total = info.get("total_episodes")
    if isinstance(total, int) and index >= total:
        raise EpisodeNotFoundError(repo_id, index)

    ep_meta = _load_episode_meta(repo_id, index, chunks_size)
    rel_path = data_path_template.format(
        chunk_index=ep_meta["data_chunk_index"],
        file_index=ep_meta["data_file_index"],
    )
    try:
        parquet_path = hf_hub_download(repo_id=repo_id, filename=rel_path, repo_type="dataset")
    except Exception as e:
        raise EpisodeNotFoundError(repo_id, index) from e

    df = pd.read_parquet(parquet_path).iloc[
        ep_meta["dataset_from_index"] : ep_meta["dataset_to_index"]
    ]
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
