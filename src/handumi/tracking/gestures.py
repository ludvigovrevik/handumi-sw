"""Hands-free gesture detection from Feetech gripper widths.

Shared by any script that needs a hands-free control signal while wearing the
HandUMI shells (no free fingers to reach a controller button): today
``handumi.scripts.teleop_sim`` uses it to reset the workspace, and
``handumi.scripts.record`` uses it to start/stop an episode (--clap-control).
"""

from __future__ import annotations


class DoubleClapDetector:
    """Squeeze either gripper shut twice in quick succession.

    Each side is tracked independently. A "clap" fires when that side's width
    drops below ``close_mm``; it must reopen past ``open_mm`` (hysteresis)
    before its next clap counts. Two claps of the *same* gripper at most
    ``window_s`` apart trigger.
    """

    def __init__(
        self,
        *,
        close_mm: float = 12.0,
        open_mm: float = 20.0,
        window_s: float = 1.6,
    ) -> None:
        # Defaults tuned on hardware (2026-07-09): the original 8/25/1.2 was
        # hard to trigger — the squeeze rarely dipped under 8mm between 30Hz
        # samples, and re-opening past 25mm within 1.2s took several tries.
        self._close_mm = close_mm
        self._open_mm = open_mm
        self._window_s = window_s
        self._armed = {"left": True, "right": True}  # seen open since last clap
        self._last_clap_t: dict[str, float | None] = {"left": None, "right": None}
        self._last_clap_edges: tuple[str, ...] = ()

    @property
    def last_clap_edges(self) -> tuple[str, ...]:
        """Sides whose close threshold was crossed in the latest update."""
        return self._last_clap_edges

    def reset(self) -> None:
        """Forget partial gestures and wait for both grippers to reopen.

        Episode transitions commonly happen while the triggering gripper is
        still closed.  Re-arming immediately would count that same closure as
        the first clap of the next gesture, making one later squeeze look like
        a double clap.
        """
        for side in self._armed:
            self._armed[side] = False
            self._last_clap_t[side] = None
        self._last_clap_edges = ()

    def update(self, left_mm: float, right_mm: float, now_s: float) -> bool:
        """Feed one width sample; returns True when either side double-claps."""
        return self.update_side(left_mm, right_mm, now_s) is not None

    def update_side(self, left_mm: float, right_mm: float, now_s: float) -> str | None:
        """Return the side that double-clapped, preferring right if both fire."""
        triggered = self.update_sides(left_mm, right_mm, now_s)
        if "right" in triggered:
            return "right"
        return triggered[0] if triggered else None

    def update_sides(
        self, left_mm: float, right_mm: float, now_s: float
    ) -> tuple[str, ...]:
        """Return every side that completed a double clap in this sample."""
        triggered: list[str] = []
        clap_edges: list[str] = []
        for side, mm in (("left", left_mm), ("right", right_mm)):
            # Calibrated widths are commonly clipped exactly at a configured
            # endpoint.  Inclusive thresholds avoid making one physical side
            # impossible to re-arm merely because its last sample lands on
            # ``open_mm`` (or impossible to clap at exactly ``close_mm``).
            if mm >= self._open_mm:
                self._armed[side] = True
                last = self._last_clap_t[side]
                if last is not None and now_s - last > self._window_s:
                    self._last_clap_t[side] = None  # first clap expired
                continue
            if mm <= self._close_mm and self._armed[side]:
                self._armed[side] = False
                clap_edges.append(side)
                last = self._last_clap_t[side]
                if last is not None and now_s - last <= self._window_s:
                    self._last_clap_t[side] = None
                    triggered.append(side)
                else:
                    self._last_clap_t[side] = now_s
        self._last_clap_edges = tuple(clap_edges)
        return tuple(triggered)


class BilateralClapArbiter:
    """Resolve near-simultaneous left/right double claps into one gesture.

    Two double claps on opposite sides within ``bilateral_window_s`` of each
    other count as a single "both" gesture (for example, a session-ending
    signal) instead of two independent single-side gestures.
    """

    def __init__(self, *, bilateral_window_s: float = 0.2) -> None:
        self._bilateral_window_s = bilateral_window_s
        self._pending_side: str | None = None
        self._pending_since_s = 0.0

    def reset(self) -> None:
        """Forget a partial (single-side) gesture when a boundary is crossed."""
        self._pending_side = None
        self._pending_since_s = 0.0

    def update(
        self,
        detector: "DoubleClapDetector",
        left_mm: float,
        right_mm: float,
        now_s: float,
    ) -> str | None:
        """Return ``left``, ``right`` or ``both`` after chord arbitration."""
        triggered = detector.update_sides(left_mm, right_mm, now_s)
        if len(triggered) == 2:
            self.reset()
            return "both"

        new_side = triggered[0] if triggered else None
        if self._pending_side is not None:
            pending_side = self._pending_side
            deadline_s = self._pending_since_s + self._bilateral_window_s
            if (
                new_side is not None
                and new_side != pending_side
                and now_s <= deadline_s
            ):
                self.reset()
                return "both"
            if now_s >= deadline_s:
                self.reset()
                if new_side is not None:
                    self._pending_side = new_side
                    self._pending_since_s = now_s
                return pending_side

        if new_side is not None:
            self._pending_side = new_side
            self._pending_since_s = now_s
        return None
