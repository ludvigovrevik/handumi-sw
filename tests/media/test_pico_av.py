from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from handumi.media.pico_av import (
    AUDIO_SAMPLE_INDEX_KEY,
    EPISODE_TRANSCRIPT_KEY,
    PICO_LEFT_VIDEO_KEY,
    PICO_RIGHT_VIDEO_KEY,
    TRANSCRIPT_KEY,
    PicoAvCapture,
    PicoAvController,
    augment_lerobot_dataset_with_pico_av,
)


def _run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True)


def test_augment_lerobot_dataset_adds_loadable_synchronized_media(
    tmp_path: Path,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/pico-av-test",
        root=root,
        fps=10,
        robot_type="test",
        features={
            "observation.tracking.device_time_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }
        },
        use_videos=False,
    )
    video_start_ns = 1_700_000_000_000_000_000
    device_times_ns = video_start_ns + np.arange(5, dtype=np.int64) * 100_000_000
    for timestamp_ns in device_times_ns:
        dataset.add_frame(
            {
                "observation.tracking.device_time_ns": np.asarray(
                    [timestamp_ns], dtype=np.int64
                ),
                "task": "test",
            }
        )
    dataset.save_episode()
    dataset.finalize()

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    video_path = capture_dir / "pico_stereo.mp4"
    audio_path = capture_dir / "pico_microphone.m4a"
    _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x120:rate=10:duration=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    )
    _run(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:a",
        "aac",
        str(audio_path),
    )
    manifest = {
        "schema_version": 1,
        "episode_id": "episode_000000_test",
        "state": "complete",
        "video_file": video_path.name,
        "audio_file": audio_path.name,
        "video_start_time_ns": video_start_ns,
        "video_stop_time_ns": video_start_ns + 1_000_000_000,
        "audio_start_time_ns": video_start_ns,
        "audio_stop_time_ns": video_start_ns + 1_000_000_000,
        "audio_sample_rate_hz": 48000,
        "audio_requested": True,
        "audio_available": True,
    }
    (capture_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    capture = PicoAvCapture(
        episode_index=0,
        episode_id="episode_000000_test",
        local_dir=capture_dir,
        manifest=manifest,
        frame_count=5,
    )

    info = augment_lerobot_dataset_with_pico_av(root, [capture], transcribe=False)

    assert info["features"][PICO_LEFT_VIDEO_KEY]["shape"] == [120, 160, 3]
    assert info["features"][PICO_RIGHT_VIDEO_KEY]["shape"] == [120, 160, 3]
    assert (root / "audio/pico_microphone/chunk-000/file-000.m4a").is_file()
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    assert stats[PICO_LEFT_VIDEO_KEY]["count"] == [5]
    assert stats[PICO_RIGHT_VIDEO_KEY]["count"] == [5]
    assert stats[AUDIO_SAMPLE_INDEX_KEY]["count"] == [5]
    data = pq.read_table(next((root / "data").glob("chunk-*/*.parquet")))
    assert data[TRANSCRIPT_KEY].to_pylist() == [""] * 5
    assert data[EPISODE_TRANSCRIPT_KEY].to_pylist() == [""] * 5
    assert data[AUDIO_SAMPLE_INDEX_KEY].to_pylist() == [0, 4800, 9600, 14400, 19200]
    for key in ("pico_head_left", "pico_head_right"):
        assert data[f"observation.camera.{key}.healthy"].to_pylist() == [1] * 5

    loaded = LeRobotDataset(repo_id="local/pico-av-test", root=root)
    assert len(loaded) == 5
    assert loaded.meta.video_keys == [PICO_LEFT_VIDEO_KEY, PICO_RIGHT_VIDEO_KEY]
    frame = loaded[2]
    assert tuple(frame[PICO_LEFT_VIDEO_KEY].shape) == (3, 120, 160)
    assert tuple(frame[PICO_RIGHT_VIDEO_KEY].shape) == (3, 120, 160)


def test_capture_fixture_requires_ffmpeg() -> None:
    assert shutil.which("ffmpeg")
    assert shutil.which("ffprobe")


def test_controller_waits_for_recording_ack_before_returning(tmp_path: Path) -> None:
    sent: list[dict] = []

    class FakeXrt:
        def device_control_json(self, device_id: str, payload: str) -> None:
            assert device_id == "PICO-SN"
            sent.append(json.loads(payload))

    def runner(command: list[str], **_: object) -> SimpleNamespace:
        assert command[:3] == ["adb", "shell", "cat"]
        episode_id = sent[0]["value"]["episodeId"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"state": "recording", "episode_id": episode_id}),
            stderr="",
        )

    controller = PicoAvController(
        xrt=FakeXrt(),
        device_id="PICO-SN",
        output_root=tmp_path / "new-dataset",
        runner=runner,
    )
    pending = controller.start_episode(3)

    assert pending.episode_index == 3
    assert pending.episode_id.startswith("episode_000003_")
    assert sent[0]["functionName"] == "CameraRecord"
    assert sent[0]["value"]["recordAudio"] is True
    # Controller construction/start must not create the LeRobot root before
    # LeRobotDataset.create() gets a chance to initialize it.
    assert not (tmp_path / "new-dataset").exists()
