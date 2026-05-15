"""One-off script to build the offline test fixture.

Pulls one short ALOHA episode from HuggingFace, slices it down to ~1 s of
frames, and saves it as a checked-in parquet that ``test_pipeline.py`` can
read without any network access.

Run once whenever the LeRobot v3.0 schema changes:
    python scripts/build_fixture.py
"""

from __future__ import annotations

from pathlib import Path

from monty_demo._io import load_lerobot_episode

REPO_ID = "lerobot/aloha_static_coffee"
EPISODE_INDEX = 0
SLICE_FRAMES = 50            # ~1 s at 50 fps
OUT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "episode_aloha_coffee_004.parquet"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = load_lerobot_episode(REPO_ID, EPISODE_INDEX)
    n = min(SLICE_FRAMES, raw.joint_positions.shape[0])

    import pandas as pd

    df = pd.DataFrame(
        {
            "observation.state": [raw.joint_positions[i].tolist() for i in range(n)],
            "action": [raw.joint_actions[i].tolist() for i in range(n)],
            "frame_index": list(range(n)),
        }
    )
    df.attrs["repo_id"] = raw.repo_id
    df.attrs["episode_index"] = raw.index
    df.attrs["fps"] = raw.fps
    df.attrs["embodiment"] = raw.embodiment
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {n} frames to {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
