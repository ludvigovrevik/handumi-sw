"""Synchronize XRoboToolkit PICO video/audio with a HandUMI LeRobot dataset.

The modified XRoboToolkit APK records one transactional directory per HandUMI
episode.  This module controls that recorder through ``device_control_json``,
pulls the completed directory over ADB, aligns it against the PICO device clock
already stored in every HandUMI row, and adds two headset camera streams plus
audio/transcript fields to the finalized LeRobot v3 dataset.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

log = logging.getLogger("handumi.pico_av")

PICO_REMOTE_ROOT = "/sdcard/Download/HandUMI"
PICO_LEFT_VIDEO_KEY = "observation.images.pico_head_left"
PICO_RIGHT_VIDEO_KEY = "observation.images.pico_head_right"
TRANSCRIPT_KEY = "observation.language.pico_transcript"
EPISODE_TRANSCRIPT_KEY = "observation.language.pico_episode_transcript"
AUDIO_ACTIVE_KEY = "observation.audio.pico.active"
AUDIO_SAMPLE_INDEX_KEY = "observation.audio.pico.sample_index"
AUDIO_SAMPLE_TIME_KEY = "observation.audio.pico.sample_time_ns"
AUDIO_SYNC_ERROR_KEY = "observation.audio.pico.sync_error_ns"


@dataclass(frozen=True)
class PicoAvCapture:
    episode_index: int
    episode_id: str
    local_dir: Path
    manifest: dict[str, Any]
    frame_count: int

    @property
    def video_path(self) -> Path:
        return self.local_dir / str(self.manifest["video_file"])

    @property
    def audio_path(self) -> Path | None:
        if not bool(self.manifest.get("audio_available", False)):
            return None
        path = self.local_dir / str(self.manifest.get("audio_file", ""))
        return path if path.is_file() and path.stat().st_size > 0 else None


@dataclass(frozen=True)
class _PendingCapture:
    episode_index: int
    episode_id: str
    remote_dir: str


class PicoAvController:
    """Episode-level control of the HandUMI XRoboToolkit APK."""

    def __init__(
        self,
        *,
        xrt: Any,
        device_id: str,
        output_root: Path,
        adb_serial: str | None = None,
        width: int = 2160,
        height: int = 810,
        fps: int = 30,
        bitrate: int = 20 * 1024 * 1024,
        record_audio: bool = True,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.xrt = xrt
        self.device_id = device_id
        self.output_root = Path(output_root)
        self.adb_serial = adb_serial
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.record_audio = record_audio
        self.runner = runner
        self.raw_root = self.output_root / ".pico_av_raw"

    def start_episode(
        self, episode_index: int, *, timeout_s: float = 15.0
    ) -> _PendingCapture:
        episode_id = f"episode_{episode_index:06d}_{uuid.uuid4().hex[:8]}"
        value = {
            "on": 1,
            "episodeId": episode_id,
            "videoWidth": self.width,
            "videoHeight": self.height,
            "videoFps": self.fps,
            "bitrate": self.bitrate,
            "captureRenderMode": 2,
            "recordAudio": self.record_audio,
            "recordTracking": True,
        }
        pending = _PendingCapture(
            episode_index=episode_index,
            episode_id=episode_id,
            remote_dir=f"{PICO_REMOTE_ROOT}/{episode_id}",
        )
        self._send_camera_record(value)
        log.info("PICO audiovisual capture requested: %s", episode_id)
        try:
            manifest = self._wait_for_remote_json(
                f"{pending.remote_dir}/manifest.inprogress.json",
                expected_state="recording",
                timeout_s=timeout_s,
            )
        except Exception:
            # A failed start must not leave the headset camera or microphone open.
            try:
                self._send_camera_record({"on": 0, "episodeId": pending.episode_id})
            except Exception:
                log.exception("Could not stop PICO after failed audiovisual start")
            raise
        if manifest.get("episode_id") != episode_id:
            self._send_camera_record({"on": 0, "episodeId": pending.episode_id})
            raise RuntimeError(
                f"PICO acknowledged the wrong episode: {manifest.get('episode_id')!r}"
            )
        log.info("PICO audiovisual capture ready: %s", episode_id)
        return pending

    def stop_episode(self, pending: _PendingCapture) -> None:
        self._send_camera_record({"on": 0, "episodeId": pending.episode_id})
        log.info("PICO audiovisual stop requested: %s", pending.episode_id)

    def wait_episode_stopped(
        self, pending: _PendingCapture, *, timeout_s: float = 20.0
    ) -> None:
        self._wait_for_remote_json(
            f"{pending.remote_dir}/manifest.json",
            expected_state="complete",
            timeout_s=timeout_s,
        )

    def collect_episode(
        self,
        pending: _PendingCapture,
        *,
        frame_count: int,
        timeout_s: float = 20.0,
    ) -> PicoAvCapture:
        manifest_remote = f"{pending.remote_dir}/manifest.json"
        remote_manifest = self._wait_for_remote_json(
            manifest_remote,
            expected_state="complete",
            timeout_s=timeout_s,
        )
        self._wait_for_remote_file_stable(
            f"{pending.remote_dir}/{remote_manifest.get('video_file', 'pico_stereo.mp4')}",
            timeout_s=timeout_s,
        )

        self.raw_root.mkdir(parents=True, exist_ok=True)
        local_dir = self.raw_root / pending.episode_id
        if local_dir.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing PICO capture: {local_dir}"
            )
        pull = self._adb(
            ["pull", pending.remote_dir, str(local_dir)],
            timeout=max(30.0, timeout_s),
            check=False,
        )
        if pull.returncode != 0:
            raise RuntimeError(
                f"adb pull failed for {pending.remote_dir}: {pull.stderr.strip()}"
            )

        manifest_path = local_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid PICO AV manifest {manifest_path}: {exc}"
            ) from exc
        if manifest.get("state") != "complete":
            raise RuntimeError(
                f"PICO AV manifest is not complete: {manifest.get('state')!r}"
            )
        if manifest.get("episode_id") != pending.episode_id:
            raise RuntimeError(
                f"PICO episode mismatch: {manifest.get('episode_id')!r} != {pending.episode_id!r}"
            )
        if bool(manifest.get("audio_requested")) and not bool(
            manifest.get("audio_available")
        ):
            raise RuntimeError(
                "PICO microphone recording failed: "
                f"{manifest.get('audio_error') or 'no audio file was produced'}"
            )
        video_path = local_dir / str(manifest.get("video_file", ""))
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"PICO video is missing or empty: {video_path}")
        _probe_video_size(video_path)
        if bool(manifest.get("audio_requested")):
            audio_path = local_dir / str(manifest.get("audio_file", ""))
            if not audio_path.is_file() or audio_path.stat().st_size <= 0:
                raise RuntimeError(f"PICO audio is missing or empty: {audio_path}")
            sample_rate_hz, channels = _probe_audio_format(audio_path)
            if sample_rate_hz != int(manifest.get("audio_sample_rate_hz") or 48000):
                raise RuntimeError(
                    f"PICO audio sample rate mismatch: {sample_rate_hz} Hz"
                )
            if channels != int(manifest.get("audio_channels") or 1):
                raise RuntimeError(f"PICO audio channel mismatch: {channels}")
        return PicoAvCapture(
            episode_index=pending.episode_index,
            episode_id=pending.episode_id,
            local_dir=local_dir,
            manifest=manifest,
            frame_count=frame_count,
        )

    def _wait_for_remote_json(
        self,
        remote_path: str,
        *,
        expected_state: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            result = self._adb(["shell", "cat", remote_path], timeout=5.0, check=False)
            if result.returncode == 0:
                try:
                    payload = json.loads(result.stdout)
                except json.JSONDecodeError as exc:
                    last_error = str(exc)
                else:
                    if payload.get("state") == expected_state:
                        return payload
                    if payload.get("state") == "error":
                        raise RuntimeError(
                            f"PICO recorder rejected the episode: {payload.get('error') or payload}"
                        )
                    last_error = f"state={payload.get('state')!r}"
            else:
                last_error = (result.stderr or result.stdout or "").strip()
            time.sleep(0.1)
        raise RuntimeError(
            f"PICO did not reach {expected_state!r} in {remote_path} within {timeout_s:.0f}s"
            + (f": {last_error}" if last_error else "")
        )

    def _wait_for_remote_file_stable(
        self, remote_path: str, *, timeout_s: float
    ) -> None:
        """Wait for camera-service buffers to be flushed after the final manifest."""
        deadline = time.monotonic() + timeout_s
        stable_observations = 0
        previous_size = -1
        last_error = ""
        while time.monotonic() < deadline:
            result = self._adb(
                ["shell", "stat", "-c", "%s", remote_path], timeout=5.0, check=False
            )
            try:
                size = int(result.stdout.strip()) if result.returncode == 0 else -1
            except ValueError:
                size = -1
            if size > 0 and size == previous_size:
                stable_observations += 1
                if stable_observations >= 3:
                    return
            else:
                stable_observations = 0
            previous_size = size
            last_error = (result.stderr or result.stdout or "").strip()
            time.sleep(0.25)
        raise RuntimeError(
            f"PICO media file did not become stable: {remote_path}"
            + (f" ({last_error})" if last_error else "")
        )

    def _send_camera_record(self, value: dict[str, Any]) -> None:
        command = {
            "functionName": "CameraRecord",
            "value": value,
            "timestamp_ns": time.time_ns(),
        }
        try:
            self.xrt.device_control_json(self.device_id, json.dumps(command))
        except AttributeError as exc:
            raise RuntimeError(
                "Installed xrobotoolkit_sdk has no device_control_json API; "
                "install the cloned XRoboToolkit-PC-Service-Pybind version."
            ) from exc

    def _adb(
        self,
        args: list[str],
        *,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess:
        command = ["adb"]
        if self.adb_serial:
            command.extend(["-s", self.adb_serial])
        command.extend(args)
        return self.runner(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )


@dataclass
class _EpisodeAugmentation:
    capture: PicoAvCapture
    device_times_ns: np.ndarray
    camera_times_ns: np.ndarray
    camera_errors_ns: np.ndarray
    camera_healthy: np.ndarray
    transcript: list[str]
    episode_transcript: str
    audio_sample_indices: np.ndarray
    audio_times_ns: np.ndarray
    audio_errors_ns: np.ndarray
    audio_active: np.ndarray
    duration_s: float
    video_width: int
    video_height: int
    left_video: Path
    right_video: Path
    audio_copy: Path | None


def augment_lerobot_dataset_with_pico_av(
    root: Path,
    captures: Sequence[PicoAvCapture],
    *,
    transcribe: bool = True,
    transcription_model: str = "small",
    language: str | None = None,
    max_sync_skew_s: float = 0.060,
) -> dict[str, Any]:
    """Add aligned PICO stereo video, audio and per-row text to LeRobot v3."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            "PICO AV augmentation requires a finalized LeRobot v3.0 dataset."
        )
    if not captures:
        return info
    if any(
        key in info["features"] for key in (PICO_LEFT_VIDEO_KEY, PICO_RIGHT_VIDEO_KEY)
    ):
        raise RuntimeError("Dataset already contains PICO headset video features.")

    fps = int(info["fps"])
    capture_by_episode = {capture.episode_index: capture for capture in captures}
    if len(capture_by_episode) != len(captures):
        raise RuntimeError("Duplicate PICO captures were supplied for an episode.")
    expected = set(range(int(info["total_episodes"])))
    if set(capture_by_episode) != expected:
        raise RuntimeError(
            "Every saved episode needs exactly one PICO capture; "
            f"expected {sorted(expected)}, got {sorted(capture_by_episode)}."
        )

    rows_by_episode = _load_episode_device_times(root)
    whisper = _load_whisper(transcription_model) if transcribe else None
    augmentations: dict[int, _EpisodeAugmentation] = {}
    for episode_index in sorted(capture_by_episode):
        capture = capture_by_episode[episode_index]
        device_times = rows_by_episode.get(episode_index)
        if device_times is None or len(device_times) != capture.frame_count:
            raise RuntimeError(
                f"Episode {episode_index} row mismatch: dataset={0 if device_times is None else len(device_times)} "
                f"capture={capture.frame_count}."
            )
        augmentation = _prepare_episode_media(
            root=root,
            capture=capture,
            device_times_ns=device_times,
            fps=fps,
            whisper=whisper,
            language=language,
            max_sync_skew_s=max_sync_skew_s,
        )
        augmentations[episode_index] = augmentation

    _append_frame_columns(root, augmentations)
    _append_episode_video_metadata(root, augmentations)
    _update_dataset_stats(root, augmentations)

    sample = augmentations[min(augmentations)]
    if not info.get("video_path"):
        info["video_path"] = (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
    video_info = {
        "video.height": sample.video_height,
        "video.width": sample.video_width,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    for key in (PICO_LEFT_VIDEO_KEY, PICO_RIGHT_VIDEO_KEY):
        info["features"][key] = {
            "dtype": "video",
            "shape": [sample.video_height, sample.video_width, 3],
            "names": ["height", "width", "channel"],
            "info": dict(video_info),
        }
    info["features"].update(
        {
            TRANSCRIPT_KEY: {"dtype": "string", "shape": [1], "names": None},
            EPISODE_TRANSCRIPT_KEY: {"dtype": "string", "shape": [1], "names": None},
            AUDIO_ACTIVE_KEY: {"dtype": "int64", "shape": [1], "names": None},
            AUDIO_SAMPLE_INDEX_KEY: {"dtype": "int64", "shape": [1], "names": None},
            AUDIO_SAMPLE_TIME_KEY: {"dtype": "int64", "shape": [1], "names": None},
            AUDIO_SYNC_ERROR_KEY: {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    for name in ("pico_head_left", "pico_head_right"):
        prefix = f"observation.camera.{name}"
        for suffix in ("sample_time_ns", "sequence", "healthy"):
            info["features"][f"{prefix}.{suffix}"] = {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            }

    handumi = info.setdefault("handumi", {})
    handumi["pico_av"] = {
        "schema_version": 1,
        "clock": "pico_unix_epoch_ns",
        "alignment": "nearest_constant_rate_frame_from_device_clock",
        "max_sync_skew_s": max_sync_skew_s,
        "stereo_video_keys": [PICO_LEFT_VIDEO_KEY, PICO_RIGHT_VIDEO_KEY],
        "audio_path": "audio/pico_microphone/chunk-{chunk_index:03d}/file-{file_index:03d}.m4a",
        "audio_sample_index_feature": AUDIO_SAMPLE_INDEX_KEY,
        "audio_sample_rate_hz": int(
            sample.capture.manifest.get("audio_sample_rate_hz") or 48000
        ),
        "transcript_feature": TRANSCRIPT_KEY,
        "episode_transcript_feature": EPISODE_TRANSCRIPT_KEY,
        "transcription_model": transcription_model if transcribe else None,
        "transcription_language": language,
        "raw_capture_path": ".pico_av_raw/",
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    return info


def _load_episode_device_times(root: Path) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    parts: dict[int, list[tuple[int, int]]] = {}
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "observation.tracking.device_time_ns",
            ],
        )
        episodes = table["episode_index"].to_numpy()
        frames = table["frame_index"].to_numpy()
        times = table["observation.tracking.device_time_ns"].to_numpy()
        for episode, frame, timestamp in zip(episodes, frames, times, strict=True):
            parts.setdefault(int(episode), []).append((int(frame), int(timestamp)))
    output: dict[int, np.ndarray] = {}
    for episode, values in parts.items():
        ordered = sorted(values)
        output[episode] = np.asarray(
            [timestamp for _, timestamp in ordered], dtype=np.int64
        )
    return output


def _prepare_episode_media(
    *,
    root: Path,
    capture: PicoAvCapture,
    device_times_ns: np.ndarray,
    fps: int,
    whisper: Any | None,
    language: str | None,
    max_sync_skew_s: float,
) -> _EpisodeAugmentation:
    manifest = capture.manifest
    video_start_ns = int(manifest["video_start_time_ns"])
    first_device_ns = (
        int(device_times_ns[0]) if int(device_times_ns[0]) > 0 else video_start_ns
    )
    input_offset_s = max(0.0, (first_device_ns - video_start_ns) / 1e9)
    leading_pad_s = max(0.0, (video_start_ns - first_device_ns) / 1e9)
    width, height = _probe_video_size(capture.video_path)
    if width % 2:
        raise RuntimeError(f"PICO stereo video width must be even, got {width}.")
    half_width = width // 2
    chunk_index = capture.episode_index // 1000
    file_index = capture.episode_index % 1000
    left_path = root / (
        f"videos/{PICO_LEFT_VIDEO_KEY}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    right_path = root / (
        f"videos/{PICO_RIGHT_VIDEO_KEY}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    _encode_aligned_eye(
        capture.video_path,
        left_path,
        crop_x=0,
        width=half_width,
        height=height,
        fps=fps,
        frames=capture.frame_count,
        input_offset_s=input_offset_s,
        leading_pad_s=leading_pad_s,
    )
    _encode_aligned_eye(
        capture.video_path,
        right_path,
        crop_x=half_width,
        width=half_width,
        height=height,
        fps=fps,
        frames=capture.frame_count,
        input_offset_s=input_offset_s,
        leading_pad_s=leading_pad_s,
    )

    source_first_ns = max(video_start_ns, first_device_ns)
    camera_times = source_first_ns + (
        np.arange(capture.frame_count, dtype=np.int64) * int(round(1e9 / fps))
    )
    if leading_pad_s > 0:
        leading = int(round(leading_pad_s * fps))
        camera_times[: min(leading, len(camera_times))] = video_start_ns
    camera_errors = np.abs(device_times_ns - camera_times).astype(np.int64)

    audio_copy: Path | None = None
    transcript_segments: list[dict[str, Any]] = []
    audio_start_ns = int(manifest.get("audio_start_time_ns") or 0)
    audio_stop_ns = int(manifest.get("audio_stop_time_ns") or 0)
    if capture.audio_path is not None:
        audio_copy = root / (
            f"audio/pico_microphone/chunk-{chunk_index:03d}/file-{file_index:03d}.m4a"
        )
        audio_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(capture.audio_path, audio_copy)
        if whisper is not None:
            transcript_segments = _transcribe_audio(whisper, audio_copy, language)
    transcript = _transcript_per_frame(
        device_times_ns,
        audio_start_ns=audio_start_ns,
        segments=transcript_segments,
    )
    episode_transcript = " ".join(
        str(segment["text"]).strip()
        for segment in transcript_segments
        if segment.get("text")
    )
    transcript_path = root / (
        f"transcripts/pico_microphone/chunk-{chunk_index:03d}/file-{file_index:03d}.jsonl"
    )
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        "".join(
            json.dumps(segment, ensure_ascii=False) + "\n"
            for segment in transcript_segments
        ),
        encoding="utf-8",
    )

    audio_active = (
        (device_times_ns >= audio_start_ns) & (device_times_ns <= audio_stop_ns)
        if audio_start_ns > 0 and audio_stop_ns >= audio_start_ns
        else np.zeros(len(device_times_ns), dtype=bool)
    )
    sample_rate_hz = int(manifest.get("audio_sample_rate_hz") or 48000)
    relative_audio_ns = device_times_ns - audio_start_ns
    audio_sample_indices = np.rint(relative_audio_ns * sample_rate_hz / 1e9).astype(
        np.int64
    )
    nearest_audio_times = audio_start_ns + np.rint(
        audio_sample_indices * 1e9 / sample_rate_hz
    ).astype(np.int64)
    audio_sample_indices = np.where(audio_active, audio_sample_indices, -1).astype(
        np.int64
    )
    audio_times = np.where(audio_active, nearest_audio_times, 0).astype(np.int64)
    audio_errors = np.where(
        audio_active,
        np.abs(device_times_ns - nearest_audio_times),
        np.iinfo(np.int64).max,
    ).astype(np.int64)
    camera_healthy = camera_errors <= int(max_sync_skew_s * 1e9)
    healthy_ratio = float(np.mean(camera_healthy))
    if healthy_ratio < 0.95:
        raise RuntimeError(
            f"PICO video sync failed for episode {capture.episode_index}: "
            f"only {healthy_ratio:.1%} of rows are within {max_sync_skew_s * 1000:.1f} ms."
        )
    return _EpisodeAugmentation(
        capture=capture,
        device_times_ns=device_times_ns,
        camera_times_ns=camera_times,
        camera_errors_ns=camera_errors,
        camera_healthy=camera_healthy.astype(np.int64),
        transcript=transcript,
        episode_transcript=episode_transcript,
        audio_sample_indices=audio_sample_indices,
        audio_times_ns=audio_times,
        audio_errors_ns=audio_errors,
        audio_active=audio_active.astype(np.int64),
        duration_s=capture.frame_count / fps,
        video_width=half_width,
        video_height=height,
        left_video=left_path,
        right_video=right_path,
        audio_copy=audio_copy,
    )


def _encode_aligned_eye(
    source: Path,
    destination: Path,
    *,
    crop_x: int,
    width: int,
    height: int,
    fps: int,
    frames: int,
    input_offset_s: float,
    leading_pad_s: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"crop={width}:{height}:{crop_x}:0"]
    if leading_pad_s > 0:
        filters.append(f"tpad=start_mode=clone:start_duration={leading_pad_s:.9f}")
    filters.extend(
        [
            f"fps={fps}:round=near",
            "tpad=stop_mode=clone:stop_duration=3600",
            f"trim=end_frame={frames}",
            f"setpts=N/({fps}*TB)",
        ]
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if input_offset_s > 0:
        command.extend(["-ss", f"{input_offset_s:.9f}"])
    command.extend(
        [
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {destination}: {result.stderr.strip()}")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced no aligned video: {destination}")


def _probe_video_size(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    stream = payload.get("streams", [{}])[0]
    return int(stream["width"]), int(stream["height"])


def _probe_audio_format(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio stream found in {path}.")
    return int(streams[0]["sample_rate"]), int(streams[0]["channels"])


def _load_whisper(model_name: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "PICO transcription requires faster-whisper. Run `uv sync` or use "
            "--no-pico-transcribe."
        ) from exc
    log.info(
        "Loading faster-whisper model %s for PICO audio transcription ...", model_name
    )
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _transcribe_audio(
    model: Any, path: Path, language: str | None
) -> list[dict[str, Any]]:
    segments, info = model.transcribe(
        str(path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )
    output: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.text).strip()
        if not text:
            continue
        output.append(
            {
                "start_s": float(segment.start),
                "end_s": float(segment.end),
                "text": text,
                "language": getattr(info, "language", language),
                "words": [
                    {
                        "start_s": float(word.start),
                        "end_s": float(word.end),
                        "word": str(word.word),
                        "probability": float(word.probability),
                    }
                    for word in (segment.words or [])
                ],
            }
        )
    return output


def _transcript_per_frame(
    device_times_ns: np.ndarray,
    *,
    audio_start_ns: int,
    segments: Sequence[dict[str, Any]],
) -> list[str]:
    if audio_start_ns <= 0 or not segments:
        return [""] * len(device_times_ns)
    output: list[str] = []
    for timestamp in device_times_ns:
        relative_s = (int(timestamp) - audio_start_ns) / 1e9
        active = [
            str(segment["text"])
            for segment in segments
            if float(segment["start_s"]) <= relative_s <= float(segment["end_s"])
        ]
        output.append(" ".join(active))
    return output


def _append_frame_columns(
    root: Path,
    augmentations: dict[int, _EpisodeAugmentation],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys_and_types = {
        TRANSCRIPT_KEY: pa.string(),
        EPISODE_TRANSCRIPT_KEY: pa.string(),
        AUDIO_ACTIVE_KEY: pa.int64(),
        AUDIO_SAMPLE_INDEX_KEY: pa.int64(),
        AUDIO_SAMPLE_TIME_KEY: pa.int64(),
        AUDIO_SYNC_ERROR_KEY: pa.int64(),
    }
    for name in ("pico_head_left", "pico_head_right"):
        keys_and_types[f"observation.camera.{name}.sample_time_ns"] = pa.int64()
        keys_and_types[f"observation.camera.{name}.sequence"] = pa.int64()
        keys_and_types[f"observation.camera.{name}.healthy"] = pa.int64()

    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        if any(key in table.column_names for key in keys_and_types):
            raise RuntimeError(f"PICO AV columns already exist in {path}.")
        episodes = table["episode_index"].to_numpy()
        frames = table["frame_index"].to_numpy()
        values: dict[str, list[Any]] = {key: [] for key in keys_and_types}
        for episode_value, frame_value in zip(episodes, frames, strict=True):
            episode = int(episode_value)
            frame = int(frame_value)
            aug = augmentations[episode]
            healthy = int(aug.camera_healthy[frame])
            values[TRANSCRIPT_KEY].append(aug.transcript[frame])
            values[EPISODE_TRANSCRIPT_KEY].append(aug.episode_transcript)
            values[AUDIO_ACTIVE_KEY].append(int(aug.audio_active[frame]))
            values[AUDIO_SAMPLE_INDEX_KEY].append(int(aug.audio_sample_indices[frame]))
            values[AUDIO_SAMPLE_TIME_KEY].append(int(aug.audio_times_ns[frame]))
            values[AUDIO_SYNC_ERROR_KEY].append(int(aug.audio_errors_ns[frame]))
            for name in ("pico_head_left", "pico_head_right"):
                prefix = f"observation.camera.{name}"
                values[f"{prefix}.sample_time_ns"].append(
                    int(aug.camera_times_ns[frame])
                )
                values[f"{prefix}.sequence"].append(frame)
                values[f"{prefix}.healthy"].append(healthy)
        for key, arrow_type in keys_and_types.items():
            table = table.append_column(key, pa.array(values[key], type=arrow_type))
        temporary = path.with_suffix(".pico-av.tmp.parquet")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(path)


def _append_episode_video_metadata(
    root: Path,
    augmentations: dict[int, _EpisodeAugmentation],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        episodes = table["episode_index"].to_numpy()
        for video_key in (PICO_LEFT_VIDEO_KEY, PICO_RIGHT_VIDEO_KEY):
            columns = {
                f"videos/{video_key}/chunk_index": [],
                f"videos/{video_key}/file_index": [],
                f"videos/{video_key}/from_timestamp": [],
                f"videos/{video_key}/to_timestamp": [],
            }
            for episode_value in episodes:
                episode = int(episode_value)
                aug = augmentations[episode]
                columns[f"videos/{video_key}/chunk_index"].append(episode // 1000)
                columns[f"videos/{video_key}/file_index"].append(episode % 1000)
                columns[f"videos/{video_key}/from_timestamp"].append(0.0)
                columns[f"videos/{video_key}/to_timestamp"].append(aug.duration_s)
            for key, values in columns.items():
                arrow_type = pa.float64() if key.endswith("timestamp") else pa.int64()
                table = table.append_column(key, pa.array(values, type=arrow_type))
        audio_columns: dict[str, list[Any]] = {
            "audios/pico_microphone/chunk_index": [],
            "audios/pico_microphone/file_index": [],
            "audios/pico_microphone/from_timestamp": [],
            "audios/pico_microphone/to_timestamp": [],
            "audios/pico_microphone/path": [],
            "transcripts/pico_microphone/path": [],
        }
        for episode_value in episodes:
            episode = int(episode_value)
            aug = augmentations[episode]
            chunk_index = episode // 1000
            file_index = episode % 1000
            audio_columns["audios/pico_microphone/chunk_index"].append(chunk_index)
            audio_columns["audios/pico_microphone/file_index"].append(file_index)
            audio_columns["audios/pico_microphone/from_timestamp"].append(
                max(
                    0.0,
                    (
                        int(aug.device_times_ns[0])
                        - int(aug.capture.manifest.get("audio_start_time_ns") or 0)
                    )
                    / 1e9,
                )
            )
            audio_columns["audios/pico_microphone/to_timestamp"].append(
                max(
                    0.0,
                    (
                        int(aug.device_times_ns[-1])
                        - int(aug.capture.manifest.get("audio_start_time_ns") or 0)
                    )
                    / 1e9,
                )
            )
            audio_columns["audios/pico_microphone/path"].append(
                f"audio/pico_microphone/chunk-{chunk_index:03d}/file-{file_index:03d}.m4a"
            )
            audio_columns["transcripts/pico_microphone/path"].append(
                f"transcripts/pico_microphone/chunk-{chunk_index:03d}/file-{file_index:03d}.jsonl"
            )
        for key, values in audio_columns.items():
            if key.endswith(("from_timestamp", "to_timestamp")):
                arrow_type = pa.float64()
            elif key.endswith("path"):
                arrow_type = pa.string()
            else:
                arrow_type = pa.int64()
            table = table.append_column(key, pa.array(values, type=arrow_type))
        temporary = path.with_suffix(".pico-av.tmp.parquet")
        pq.write_table(table, temporary, compression="snappy")
        temporary.replace(path)


def _update_dataset_stats(
    root: Path,
    augmentations: dict[int, _EpisodeAugmentation],
) -> None:
    """Keep LeRobot normalization metadata complete for the added features."""
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import load_stats, write_stats

    per_episode: list[dict[str, dict[str, np.ndarray]]] = []
    for episode in sorted(augmentations):
        aug = augmentations[episode]
        stats: dict[str, dict[str, np.ndarray]] = {
            PICO_LEFT_VIDEO_KEY: _video_stats(aug.left_video, aug.capture.frame_count),
            PICO_RIGHT_VIDEO_KEY: _video_stats(
                aug.right_video, aug.capture.frame_count
            ),
        }
        numeric = {
            AUDIO_ACTIVE_KEY: aug.audio_active,
            AUDIO_SAMPLE_INDEX_KEY: aug.audio_sample_indices,
            AUDIO_SAMPLE_TIME_KEY: aug.audio_times_ns,
            AUDIO_SYNC_ERROR_KEY: aug.audio_errors_ns,
        }
        for name in ("pico_head_left", "pico_head_right"):
            prefix = f"observation.camera.{name}"
            numeric[f"{prefix}.sample_time_ns"] = aug.camera_times_ns
            numeric[f"{prefix}.sequence"] = np.arange(
                aug.capture.frame_count, dtype=np.int64
            )
            numeric[f"{prefix}.healthy"] = aug.camera_healthy
        for key, values in numeric.items():
            stats[key] = get_feature_stats(np.asarray(values), axis=0, keepdims=True)
        per_episode.append(stats)

    new_stats = aggregate_stats(per_episode)
    existing_stats = load_stats(root) or {}
    overlap = set(existing_stats) & set(new_stats)
    if overlap:
        raise RuntimeError(f"PICO AV stats already exist for: {sorted(overlap)}")
    existing_stats.update(new_stats)
    write_stats(existing_stats, root)


def _video_stats(path: Path, frame_count: int) -> dict[str, np.ndarray]:
    import av

    from lerobot.datasets.compute_stats import get_feature_stats, sample_indices

    wanted = set(sample_indices(frame_count))
    sampled: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            channel_first = rgb.transpose(2, 0, 1)
            _, height, width = channel_first.shape
            if max(width, height) >= 300:
                factor = int(width / 150) if width > height else int(height / 150)
                channel_first = channel_first[:, ::factor, ::factor]
            sampled.append(channel_first.astype(np.float32) / 255.0)
    if len(sampled) != len(wanted):
        raise RuntimeError(
            f"Decoded {len(sampled)} sampled frames from {path}; expected {len(wanted)}."
        )
    stats = get_feature_stats(np.stack(sampled), axis=(0, 2, 3), keepdims=True)
    return {
        key: value if key == "count" else np.squeeze(value, axis=0)
        for key, value in stats.items()
    }
