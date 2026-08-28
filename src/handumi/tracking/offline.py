"""Placeholder tracker for offline-VIO capture.

Upstream assumed pose arrives live from a VR headset, and gated recording on
both controllers being tracked. The Lito rig has no headset: pose is solved by
visual-inertial SLAM afterwards, on a server, from the recorded camera + IMU.

So during capture there is no pose to report. This provider satisfies the
``TrackingProvider`` shape with identity poses marked tracked, which keeps the
recorder's health gate open so the streams that DO exist -- camera, IMU,
gripper, tactile -- get recorded and timestamped.

``observation.state`` is therefore identity for every frame in a raw session.
That is expected. The offline pipeline solves VIO and fills those columns in
before the dataset is used for training.

⚠️ Consequence: a raw session is NOT trainable as-is. It is trainable after
post-processing. Anything reading these datasets must check
``handumi.pose_source`` in ``meta/info.json`` rather than assuming pose is real.
"""

from __future__ import annotations

from handumi.tracking.base import ControllerPairSample, TrackingProvider

POSE_SOURCE = "offline_vio_pending"


class OfflinePoseTracker(TrackingProvider):
    """Reports 'tracked' with identity poses so capture can proceed without VR."""

    def __init__(self, device: str = "offline") -> None:
        self.device = device
        self._sequence = 0

    def start(self) -> None:  # nothing to connect to
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> ControllerPairSample:
        self._sequence += 1
        sample = ControllerPairSample.empty(self.device)
        # Mark tracked/connected so SustainedHealthGate does not stall the
        # recorder waiting for a tracker that will never report.
        sample.left_tracked = True
        sample.right_tracked = True
        sample.left_pose_valid = True
        sample.right_pose_valid = True
        sample.connected = True
        sample.streaming = True
        sample.clock_synced = True
        sample.sequence = self._sequence
        return sample

    def sample_at(self, target_time_ns: int) -> ControllerPairSample:
        sample = self.latest()
        sample.aligned_time_ns = target_time_ns
        sample.pc_monotonic_ns = target_time_ns
        return sample


__all__ = ["OfflinePoseTracker", "POSE_SOURCE"]
