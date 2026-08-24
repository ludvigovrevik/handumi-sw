"""Rebuild LeRobot v3 episode metadata from intact frame Parquet files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_CHUNK_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.(?:parquet|mp4)$")


def _chunk_file(path: Path) -> tuple[int, int]:
    match = _CHUNK_FILE_RE.search(path.as_posix())
    if match is None:
        raise RuntimeError(f"Unexpected LeRobot chunk path: {path}")
    return int(match.group(1)), int(match.group(2))


def _video_frame_count(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def _video_locations(
    root: Path, video_key: str, episode_lengths: dict[int, int], fps: float
) -> dict[int, dict[str, int | float]]:
    paths = sorted((root / "videos" / video_key).glob("chunk-*/*.mp4"))
    if not paths:
        raise RuntimeError(f"Missing video files for {video_key!r}.")

    locations: dict[int, dict[str, int | float]] = {}
    episode_indices = iter(sorted(episode_lengths))
    try:
        episode_index = next(episode_indices)
    except StopIteration:
        return locations

    for path in paths:
        chunk_index, file_index = _chunk_file(path)
        file_frames = _video_frame_count(path)
        frame_offset = 0
        while frame_offset < file_frames:
            length = episode_lengths[episode_index]
            if frame_offset + length > file_frames:
                raise RuntimeError(
                    f"Episode {episode_index} crosses the boundary of video file {path}."
                )
            prefix = f"videos/{video_key}"
            locations[episode_index] = {
                f"{prefix}/chunk_index": chunk_index,
                f"{prefix}/file_index": file_index,
                f"{prefix}/from_timestamp": frame_offset / fps,
                f"{prefix}/to_timestamp": (frame_offset + length) / fps,
            }
            frame_offset += length
            try:
                episode_index = next(episode_indices)
            except StopIteration:
                if frame_offset != file_frames:
                    raise RuntimeError(f"Video {path} contains unreferenced frames.")
                return locations

        if frame_offset != file_frames:
            raise RuntimeError(
                f"Video frame count does not match episode boundaries: {path}"
            )

    raise RuntimeError(f"Videos for {video_key!r} end before all episodes are present.")


def _task_names(root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    if "task_index" not in tasks.columns:
        raise RuntimeError("meta/tasks.parquet has no task_index column.")
    return {int(row.task_index): str(index) for index, row in tasks.iterrows()}


def _global_video_stats(
    root: Path, video_keys: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads((root / "meta" / "stats.json").read_text())
    return {
        key: {name: np.asarray(value) for name, value in raw[key].items()}
        for key in video_keys
        if key in raw
    }


def _episode_stats(
    frame_df: pd.DataFrame,
    features: dict[str, dict[str, Any]],
    global_video_stats: dict[str, dict[str, np.ndarray]],
    total_frames: int,
) -> dict[str, dict[str, np.ndarray]]:
    from lerobot.datasets.compute_stats import compute_episode_stats

    numeric_features = {
        key: feature
        for key, feature in features.items()
        if feature.get("dtype") not in {"video", "image", "string"} and key in frame_df
    }
    episode_data: dict[str, np.ndarray] = {}
    for key in numeric_features:
        values = frame_df[key].to_numpy()
        episode_data[key] = (
            np.stack(values)
            if len(values) and hasattr(values[0], "__len__")
            else values
        )
    stats = compute_episode_stats(episode_data, numeric_features)

    # The intact global image statistics keep training normalization correct.  A
    # corrupt episode Parquet no longer contains the original per-episode image
    # samples, so apportion their count by episode length.  All other values are
    # copied; this is conservative and allows LeRobot editing tools to aggregate
    # the repaired metadata without losing image-stat columns.
    for key, feature_stats in global_video_stats.items():
        copied = {name: value.copy() for name, value in feature_stats.items()}
        global_count = int(np.asarray(copied["count"]).reshape(-1)[0])
        count = round(global_count * len(frame_df) / total_frames)
        copied["count"] = np.asarray([count], dtype=np.int64)
        stats[key] = copied
    return stats


def _flatten_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        f"stats/{feature}/{name}": value
        for feature, feature_stats in stats.items()
        for name, value in feature_stats.items()
    }


def _read_frames(root: Path) -> tuple[pd.DataFrame, dict[int, tuple[int, int]]]:
    frames: list[pd.DataFrame] = []
    data_locations: dict[int, tuple[int, int]] = {}
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        chunk_file = _chunk_file(path)
        frame_df = pd.read_parquet(path)
        frames.append(frame_df)
        for episode_index in frame_df["episode_index"].unique():
            index = int(episode_index)
            previous = data_locations.setdefault(index, chunk_file)
            if previous != chunk_file:
                raise RuntimeError(f"Episode {index} crosses data Parquet files.")
    if not frames:
        raise RuntimeError("Dataset contains no data Parquet files.")
    return pd.concat(frames, ignore_index=True), data_locations


def restore_handumi_layout_metadata(
    root: str | Path, info: dict[str, Any], all_frames: pd.DataFrame
) -> bool:
    """Restore only HandUMI fields that are provable from the captured schema."""
    if info.get("robot_type") != "handumi_raw":
        return False

    from handumi.dataset.raw import (
        HANDUMI_CAPTURE_SCHEMA,
        HANDUMI_RAW_STATE_NAMES,
        HANDUMI_STATE_SEMANTICS,
        HANDUMI_TRACKING_SCHEMA,
    )

    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError("Cannot restore HandUMI metadata without feature metadata.")
    required = {
        "observation.state",
        "observation.valid",
        "observation.tracking.workspace_from_device_pose",
        "observation.sync.target_time_ns",
        "observation.sync.record_time_ns",
    }
    missing = sorted(required - features.keys())
    if missing:
        raise RuntimeError(
            "Dataset declares handumi_raw but lacks required capture features: "
            + ", ".join(missing)
        )
    state_feature = features["observation.state"]
    if (
        not isinstance(state_feature, dict)
        or tuple(state_feature.get("shape", ())) != (len(HANDUMI_RAW_STATE_NAMES),)
        or tuple(state_feature.get("names") or ()) != HANDUMI_RAW_STATE_NAMES
    ):
        raise RuntimeError(
            "Dataset declares handumi_raw but observation.state does not match "
            "the current compact raw layout."
        )

    current = info.get("handumi")
    handumi = dict(current) if isinstance(current, dict) else {}
    before = json.dumps(handumi, sort_keys=True)
    handumi.setdefault("tracking_schema", HANDUMI_TRACKING_SCHEMA)
    handumi.setdefault("capture_schema", HANDUMI_CAPTURE_SCHEMA)
    handumi.setdefault("state_semantics", HANDUMI_STATE_SEMANTICS)

    camera_names = sorted(
        key.removeprefix("observation.images.")
        for key, feature in features.items()
        if key.startswith("observation.images.")
        and isinstance(feature, dict)
        and feature.get("dtype") in {"video", "image"}
    )
    feetech_healthy = all_frames.get("observation.feetech.healthy")
    feetech_enabled = bool(
        feetech_healthy is not None
        and any(np.asarray(value).astype(bool).any() for value in feetech_healthy)
    )
    inferred_sources = {
        "tracking": {"enabled": True},
        "feetech": {"enabled": feetech_enabled},
        "audio": {"enabled": any((Path(root) / "audio").glob("chunk-*/*.wav"))},
        "cameras": {name: {"enabled": True} for name in camera_names},
    }
    sources = handumi.get("sources")
    if not isinstance(sources, dict):
        handumi["sources"] = inferred_sources
    else:
        for name, value in inferred_sources.items():
            sources.setdefault(name, value)

    if json.dumps(handumi, sort_keys=True) == before:
        return False
    info["handumi"] = handumi
    root = Path(root)
    info_path = root / "meta" / "info.json"
    backup = info_path.with_suffix(".json.incomplete")
    if not backup.exists():
        shutil.copy2(info_path, backup)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=info_path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(info, handle, indent=4, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        json.loads(temporary.read_text())
        temporary.replace(info_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def rebuild_episode_metadata(root: str | Path, *, force: bool = False) -> Path:
    """Rebuild ``meta/episodes`` atomically and retain the damaged file."""
    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Missing {info_path}.")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError("Metadata repair currently supports LeRobot v3.0 only.")

    output = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if output.is_file() and not force:
        try:
            pq.read_metadata(output)
        except (OSError, pa.ArrowInvalid):
            pass
        else:
            raise RuntimeError(
                f"Episode metadata is already valid: {output}. Use --force to rebuild it."
            )

    all_frames, data_locations = _read_frames(root)
    expected_episodes = list(range(int(info["total_episodes"])))
    actual_episodes = sorted(
        int(value) for value in all_frames["episode_index"].unique()
    )
    if actual_episodes != expected_episodes:
        raise RuntimeError(
            "Frame episodes do not match info.json: "
            f"expected {expected_episodes}, found {actual_episodes}."
        )
    if len(all_frames) != int(info["total_frames"]):
        raise RuntimeError(
            f"Frame count mismatch: info.json={info['total_frames']}, data={len(all_frames)}."
        )

    task_names = _task_names(root)
    episode_frames = {
        index: all_frames[all_frames["episode_index"] == index].sort_values(
            "frame_index"
        )
        for index in expected_episodes
    }
    episode_lengths = {
        index: len(frame_df) for index, frame_df in episode_frames.items()
    }
    video_keys = [
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    ]
    video_locations = {
        key: _video_locations(root, key, episode_lengths, float(info["fps"]))
        for key in video_keys
    }
    video_stats = _global_video_stats(root, video_keys)

    rows: list[dict[str, Any]] = []
    for episode_index, frame_df in episode_frames.items():
        indices = frame_df["index"].astype(int).to_numpy()
        if not np.array_equal(
            indices, np.arange(indices[0], indices[0] + len(indices))
        ):
            raise RuntimeError(
                f"Episode {episode_index} has non-contiguous global frame indices."
            )
        task_indices = sorted(int(value) for value in frame_df["task_index"].unique())
        try:
            tasks = [task_names[index] for index in task_indices]
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown task index in episode {episode_index}: {exc.args[0]}"
            ) from exc
        data_chunk, data_file = data_locations[episode_index]
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": tasks,
            "length": len(frame_df),
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
            "dataset_from_index": int(indices[0]),
            "dataset_to_index": int(indices[-1]) + 1,
        }
        for key in video_keys:
            row.update(video_locations[key][episode_index])
        row.update(
            _flatten_stats(
                _episode_stats(frame_df, info["features"], video_stats, len(all_frames))
            )
        )
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        rows.append(row)

    columns = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(
                value.tolist() if isinstance(value, np.ndarray) else value
            )
    table = pa.Table.from_pydict(columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = output.with_suffix(output.suffix + ".corrupt")
        if not backup.exists():
            shutil.copy2(output, backup)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".parquet", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        pq.read_metadata(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    restore_handumi_layout_metadata(root, info, all_frames)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a missing or truncated LeRobot v3 episode metadata Parquet."
    )
    parser.add_argument("dataset", type=Path, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild metadata even when it is readable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = rebuild_episode_metadata(args.dataset, force=args.force)
    print(f"Rebuilt episode metadata: {output}")
    backup = output.with_suffix(output.suffix + ".corrupt")
    if backup.exists():
        print(f"Original retained at: {backup}")


if __name__ == "__main__":
    main()
