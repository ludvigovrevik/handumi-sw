"""Tracking interfaces for the Lito rig.

Pose does NOT come from a live tracker here. The rig captures camera + IMU and
pose is solved offline by VIO, so ``TrackingProvider`` survives as the shape a
future offline-pose source fills in -- not as something running during capture.

Upstream's VR backends (Meta Quest, PICO) were removed with the hard fork: we
dropped the headset, and both assumed a PC in the loop.
"""

from handumi.tracking.base import ControllerPairSample, TrackingProvider

__all__ = [
    "ControllerPairSample",
    "TrackingProvider",
]
