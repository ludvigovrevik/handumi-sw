# Lito fork of handumi-sw

Hard fork of [murobotics-ai/handumi-sw](https://github.com/murobotics-ai/handumi-sw)
(Apache-2.0). We no longer merge from upstream.

## Why fork rather than track

Upstream assumes a **VR headset for pose** and a **PC in the loop**, driving both
hands from one process. The Lito rig changes that premise:

| | upstream | Lito |
|---|---|---|
| Pose | VR headset, live | camera + IMU → **offline VIO** |
| Compute | one PC, both hands | **one SBC per hand** (Radxa Zero 3W), untethered |
| Cameras | 2 wrist + workspace | **one per gripper**, no scene cam |
| Tactile | none | **FlexiTac**, 4th synced stream |

Upstream also runs ~5 commits/day, much of it in `record.py` and `tracking/` —
exactly what we replace. Tracking it would mean permanent conflicts over code we
deleted, so we cut the tie. Upstream remains as a read-only git remote for
reference; pushing to it is disabled.

## What was removed (~9,000 lines, 30%)

**Teleoperation** — `teleop/`, `scripts/teleop_{real,sim,record}.py`,
`setup_hardware.py`, `calibrate_openarm_grippers.py`. Upstream's *other* product:
leader-follower control of physical arms. We collect with a worn gripper; there
is no robot in the loop.

**VR tracking** — `tracking/{meta_quest,pico,pico_vision,mock_quest_sender}.py`,
`scripts/pico_camera.py`, `setup/print_controller_pose.py`. We dropped the headset.

**Headset audio** — `audio.py`. `PicoAudioRecorder` pulled mic audio through the
headset's API. Voice *control* still works (`utils/voice.py`, Vosk, plain mic);
only headset audio *recording* is gone.

## What replaced them

`tracking/offline.py` — `OfflinePoseTracker`, a placeholder satisfying
`TrackingProvider` with identity poses marked tracked, so the recorder's health
gate opens and the streams that exist (camera, IMU, gripper, tactile) get captured
and timestamped.

⚠️ **`observation.state` is identity in every raw session.** That is expected: pose
is filled in later by offline VIO. A raw session is *not* trainable as-is — it is
trainable after post-processing. Check `handumi.pose_source` in `meta/info.json`
rather than assuming pose is real.

## What still works

`record` · `validate` · `convert` · `replay` · `doctor` · `calibrate grippers` ·
`calibrate spatial` (subcommands `inspect-board`, `intrinsics`, `workspace`) ·
`calibrate tcp` · `calibrate verify` · `servo home` · `servo set-id` ·
`setup ports` · `completion`

Episode control is unchanged and still hands-free: **voice** ("start recording",
"stop recording") and **Feetech double-squeeze** (right saves + advances, left
discards, both ends the session). ENTER is the manual fallback.

## Known gaps

- **`calibrate spatial` mount/session/verify/visualize** raise a clear error —
  they calibrated against live VR pose. Need VIO-based equivalents. See
  RIG-SPECS.md "Calibration — the fleet question".
- **No IMU stream** in `synchronization.py`. Follow the `feetech` `sample_at()`
  pattern; keep ALL inter-frame samples, not nearest-neighbour (ORB-SLAM3 needs
  the full sequence).
- **No `observation.tactile.*`**. Reader and schema drafted in
  `../tools/tactile_reader.py`; copy `feetech_features()` in `dataset/raw.py`.
- **No cross-hand coordination** — pairing, ready-state, shared episode
  start/stop, clock-offset measurement, gesture propagation. Upstream never
  needed it (one PC). Plan: `chrony` + a clap for verification.
- **No frame-drop detection.** `cameras/opencv.py` increments its own counter
  instead of reading the camera's real sequence, so drops are silent. See
  `../tools/capture_probe.py`.
- **Still PC-shaped.** Deps include JAX, MuJoCo, pyroki. `record.py` does *not*
  import them, so the capture path is nearly clean — but the rig will run a thin
  capture agent, not this package as-is.
- `observation.tracking.*` retains VR-shaped fields (`device_hmd_pose`,
  `hmd_tracked`). Harmless, meaningless; repurpose or drop.

## Verification status

Everything compiles and all `handumi.*` imports resolve statically. **The test
suite has not been run** — the package isn't installed in this environment, so
there was no passing baseline to compare against. Install and run `pytest`
before trusting any of this on hardware.
