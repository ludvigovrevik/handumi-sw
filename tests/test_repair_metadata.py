from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from handumi.dataset.raw import HANDUMI_RAW_STATE_NAMES
from handumi.scripts.repair_metadata import rebuild_episode_metadata


def _broken_dataset(root: Path) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "handumi_raw",
        "total_episodes": 2,
        "total_frames": 5,
        "fps": 30,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": list(HANDUMI_RAW_STATE_NAMES),
            },
            "observation.valid": {"dtype": "int64", "shape": [8], "names": None},
            "observation.tracking.workspace_from_device_pose": {
                "dtype": "float32",
                "shape": [7],
                "names": None,
            },
            "observation.sync.target_time_ns": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "observation.sync.record_time_ns": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text("{}")
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index(["move toy"], name="task")
    ).to_parquet(root / "meta" / "tasks.parquet")
    pd.DataFrame(
        {
            "observation.state": [
                np.arange(value, value + 16, dtype=np.float32) for value in range(5)
            ],
            "observation.valid": [np.ones(8, dtype=np.int64) for _ in range(5)],
            "observation.tracking.workspace_from_device_pose": [
                np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32) for _ in range(5)
            ],
            "observation.sync.target_time_ns": np.arange(5, dtype=np.int64),
            "observation.sync.record_time_ns": np.arange(5, dtype=np.int64),
            "timestamp": [0.0, 1 / 30, 2 / 30, 0.0, 1 / 30],
            "frame_index": [0, 1, 2, 0, 1],
            "episode_index": [0, 0, 0, 1, 1],
            "index": [0, 1, 2, 3, 4],
            "task_index": [0, 0, 0, 0, 0],
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)
    broken = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    broken.write_bytes(b"PAR1truncated")
    return broken


def test_rebuild_episode_metadata_from_frame_parquet(tmp_path: Path) -> None:
    broken = _broken_dataset(tmp_path)

    output = rebuild_episode_metadata(tmp_path)

    assert output == broken
    assert broken.with_suffix(".parquet.corrupt").read_bytes() == b"PAR1truncated"
    table = pq.read_table(output).to_pandas()
    assert table["episode_index"].tolist() == [0, 1]
    assert table["length"].tolist() == [3, 2]
    assert table["dataset_from_index"].tolist() == [0, 3]
    assert table["dataset_to_index"].tolist() == [3, 5]
    assert table["tasks"].apply(list).tolist() == [["move toy"], ["move toy"]]
    assert "stats/observation.state/mean" in table
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["handumi"]["tracking_schema"] == "controller_raw_compact"
    assert info["handumi"]["sources"]["feetech"]["enabled"] is False
    assert (tmp_path / "meta" / "info.json.incomplete").is_file()


def test_rebuild_refuses_valid_metadata_without_force(tmp_path: Path) -> None:
    output = _broken_dataset(tmp_path)
    rebuild_episode_metadata(tmp_path)

    with pytest.raises(RuntimeError, match="already valid"):
        rebuild_episode_metadata(tmp_path)

    assert pq.read_metadata(output).num_rows == 2
