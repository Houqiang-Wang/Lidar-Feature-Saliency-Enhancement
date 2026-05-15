"""
Keyframe-based Local & Global Mapping
=====================================
Manages keyframe database and provides local map points for frame-to-map
registration.

Reference: Saliency-LOAM Section II-C (frame-to-map matching)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Keyframe:
    """A keyframe stores a snapshot of the sensor view at a specific pose."""

    pose: np.ndarray          # [x, y, theta] in world frame
    points_local: np.ndarray  # (N, 2) in sensor frame
    saliency: np.ndarray      # (N,)
    timestamp: float


class MapManager:
    """Simple keyframe map with distance/angle-based insertion policy.

    Parameters
    ----------
    keyframe_dist : float
        Minimum translation [m] required to insert a new keyframe.
    keyframe_angle : float
        Minimum rotation [rad] required to insert a new keyframe.
    local_map_frames : int
        Number of most recent keyframes used to build the local map.
    """

    def __init__(
        self,
        keyframe_dist: float = 0.5,
        keyframe_angle: float = np.deg2rad(20.0),
        local_map_frames: int = 5,
    ):
        self.keyframe_dist = keyframe_dist
        self.keyframe_angle = keyframe_angle
        self.local_map_frames = local_map_frames

        self.keyframes: List[Keyframe] = []
        self.poses: List[np.ndarray] = []  # trajectory history

    # ------------------------------------------------------------------ #
    #  Keyframe management
    # ------------------------------------------------------------------ #

    def should_insert(self, pose: np.ndarray) -> bool:
        """Check whether the robot has moved enough to spawn a new keyframe."""
        if not self.keyframes:
            return True
        last = self.keyframes[-1].pose
        dx = pose[0] - last[0]
        dy = pose[1] - last[1]
        dtheta = self._angle_diff(pose[2], last[2])
        trans = float(np.hypot(dx, dy))
        return trans > self.keyframe_dist or abs(dtheta) > self.keyframe_angle

    def add_keyframe(
        self,
        pose: np.ndarray,
        points_local: np.ndarray,
        saliency: np.ndarray,
        timestamp: float,
    ) -> None:
        """Append a new keyframe and record the pose in the trajectory."""
        kf = Keyframe(
            pose=pose.copy(),
            points_local=points_local.copy(),
            saliency=saliency.copy(),
            timestamp=timestamp,
        )
        self.keyframes.append(kf)
        self.poses.append(pose.copy())

    # ------------------------------------------------------------------ #
    #  Local map construction
    # ------------------------------------------------------------------ #

    def get_local_map(self) -> Optional[np.ndarray]:
        """Return concatenated world-frame points from recent keyframes.

        Returns ``None`` if no keyframes exist yet.
        """
        if not self.keyframes:
            return None

        recent = self.keyframes[-self.local_map_frames :]
        world_points: List[np.ndarray] = []

        for kf in recent:
            wp = self._transform_points(kf.points_local, kf.pose)
            world_points.append(wp)

        return np.vstack(world_points)

    def get_last_pose(self) -> Optional[np.ndarray]:
        """Return the most recent keyframe pose (or None)."""
        if not self.keyframes:
            return None
        return self.keyframes[-1].pose.copy()

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _transform_points(points_local: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Transform points from sensor frame to world frame."""
        cos_t, sin_t = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        return (R @ points_local.T).T + pose[:2]

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = a - b
        return (diff + np.pi) % (2.0 * np.pi) - np.pi
