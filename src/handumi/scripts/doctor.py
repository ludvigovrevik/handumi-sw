#!/usr/bin/env python3
"""Read-only installation and recording readiness checks for HandUMI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG, load_rig_config


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--robot", default="piper")
    parser.add_argument("--device", choices=("pico", "meta"), default="meta")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a required check fails.",
    )
    return parser.parse_args(argv)


def _voice_check() -> DoctorCheck:
    """Voice control is the recorder's default, so report it before a session.

    A missing model only warns: `handumi record` downloads it on first use.
    """
    from handumi.utils.voice import (
        MODEL_NAME,
        VoiceUnavailableError,
        describe_input_device,
        resolve_model_path,
    )

    try:
        microphone = describe_input_device()
    except ImportError:
        return DoctorCheck("voice", "warn", "vosk/sounddevice not installed")
    except Exception as exc:
        return DoctorCheck("voice", "warn", f"no microphone ({exc})")

    try:
        resolve_model_path(download=False)
    except VoiceUnavailableError:
        return DoctorCheck(
            "voice", "warn", f"mic {microphone!r}; {MODEL_NAME} downloads on first record"
        )
    return DoctorCheck("voice", "pass", f"mic {microphone!r}; model ready")


def collect_doctor_checks(
    rig_config: Path,
    *,
    robot: str = "piper",
    device: str = "meta",
) -> list[DoctorCheck]:
    checks = [
        DoctorCheck(
            "Python",
            "pass" if sys.version_info >= (3, 11) else "fail",
            sys.version.split()[0],
        )
    ]
    try:
        rig = load_rig_config(rig_config)
    except SystemExit as exc:
        checks.append(DoctorCheck("rig.yaml", "fail", str(exc).replace("\n", " ")))
        rig = {}
    else:
        checks.append(DoctorCheck("rig.yaml", "pass", str(rig_config)))

    cameras = rig.get("cameras") if isinstance(rig, dict) else None
    if isinstance(cameras, dict) and cameras:
        for name, raw in cameras.items():
            value = raw.get("index_or_path") if isinstance(raw, dict) else raw
            path = (
                Path(f"/dev/video{value}")
                if isinstance(value, int)
                else Path(str(value))
            )
            status = "pass" if path.exists() else "warn"
            checks.append(DoctorCheck(f"camera:{name}", status, str(path)))
    else:
        checks.append(DoctorCheck("cameras", "fail", "missing cameras mapping"))

    feetech = rig.get("feetech") if isinstance(rig, dict) else None
    if isinstance(feetech, dict):
        for side in ("left", "right"):
            side_config = feetech.get(side)
            port = side_config.get("port") if isinstance(side_config, dict) else None
            if port is None:
                port = feetech.get("port")
            if port:
                path = Path(str(port))
                checks.append(
                    DoctorCheck(
                        f"feetech:{side}",
                        "pass" if path.exists() else "warn",
                        str(path),
                    )
                )
            else:
                checks.append(
                    DoctorCheck(f"feetech:{side}", "warn", "port not configured")
                )
    else:
        checks.append(DoctorCheck("feetech", "warn", "section not configured"))

    checks.append(_voice_check())

    # Pose is solved offline by VIO, so there is no live tracker to check. What
    # matters instead is that the IMU is configured -- without it a session
    # cannot become a trajectory later, no matter how good the video is.
    imu = rig.get("imu") if isinstance(rig, dict) else None
    if isinstance(imu, dict) and imu.get("port"):
        checks.append(DoctorCheck("imu", "pass", str(imu.get("port"))))
    else:
        checks.append(
            DoctorCheck("imu", "fail", "not configured - sessions will not be usable")
        )

    robot_config = Path("configs/robots") / f"{robot}.yaml"
    checks.append(
        DoctorCheck(
            f"robot:{robot}",
            "pass" if robot_config.is_file() else "fail",
            str(robot_config),
        )
    )

    output_root = Path("outputs") if Path("outputs").exists() else Path.cwd()
    free_gib = shutil.disk_usage(output_root).free / (1024**3)
    disk_status = "pass" if free_gib >= 20 else "warn" if free_gib >= 5 else "fail"
    checks.append(
        DoctorCheck("disk", disk_status, f"{free_gib:.1f} GiB free at {output_root}")
    )

    checks.append(_encoder_check())
    checks.append(_python_stack_check())
    return checks


def _encoder_check() -> DoctorCheck:
    try:
        from handumi.scripts.record import _select_video_encoder

        selected = _select_video_encoder(
            policy="auto",
            requested_vcodec=None,
            width=640,
            height=480,
            fps=30,
            camera_count=2,
            requested_threads=None,
        )
    except (Exception, SystemExit) as exc:
        return DoctorCheck("video encoder", "fail", str(exc).replace("\n", " "))
    kind = "hardware" if selected.hardware else "CPU"
    return DoctorCheck("video encoder", "pass", f"{selected.vcodec} ({kind})")


def _python_stack_check() -> DoctorCheck:
    command = [
        sys.executable,
        "-c",
        "import torch, lerobot; print(torch.__version__)",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DoctorCheck("Python stack", "fail", str(exc))
    if result.returncode == 0:
        return DoctorCheck("Python stack", "pass", f"torch {result.stdout.strip()}")
    detail = (result.stderr or result.stdout).strip().splitlines()
    return DoctorCheck(
        "Python stack",
        "fail",
        detail[-1] if detail else f"import exited {result.returncode}",
    )


def print_doctor_report(checks: list[DoctorCheck]) -> None:
    icons = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    print("\nHandUMI readiness")
    for check in checks:
        print(f"  [{icons[check.status]}] {check.name}: {check.detail}")
    passed = sum(check.status == "pass" for check in checks)
    warnings = sum(check.status == "warn" for check in checks)
    failures = sum(check.status == "fail" for check in checks)
    print(f"\nSummary: {passed} passed, {warnings} warning(s), {failures} failure(s).")


def run_doctor(
    rig_config: Path,
    *,
    robot: str = "piper",
    device: str = "meta",
) -> bool:
    checks = collect_doctor_checks(rig_config, robot=robot, device=device)
    print_doctor_report(checks)
    return not any(check.status == "fail" for check in checks)


def main() -> None:
    args = parse_args()
    healthy = run_doctor(args.rig_config, robot=args.robot, device=args.device)
    if args.strict and not healthy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
