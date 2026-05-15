"""
Dynamic Object Suppression
==========================
Algorithm 1 (2D simplified) from Saliency-LOAM.

Compares frame-to-frame (with saliency) and frame-to-map (without saliency)
estimates.  If the pose discrepancy exceeds a threshold, dynamic interference
is assumed and the frame-to-map result is preferred.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from core.registration import ScanMatcher


class DynamicFilter:
    """Dynamic suppression via dual-estimate consistency check.

    Parameters
    ----------
    th_pose_trans : float
        Translation discrepancy threshold [m].
    th_pose_rot : float
        Rotation discrepancy threshold [rad].
    """

    def __init__(
        self,
        th_pose_trans: float = 0.2,
        th_pose_rot: float = np.deg2rad(15.0),
    ):
        self.th_trans = th_pose_trans
        self.th_rot = th_pose_rot

    def filter(
        self,
        points: np.ndarray,
        saliency: np.ndarray,
        local_map: np.ndarray | None,
        prev_points: np.ndarray | None,
        init_pose: np.ndarray,
        matcher: ScanMatcher,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Algorithm 1 (2D simplified) and return (final_pose, adjusted_saliency).

        Parameters
        ----------
        points : np.ndarray, shape (N, 2)
            Current frame points in sensor frame.
        saliency : np.ndarray, shape (N,)
            Current frame saliency.
        local_map : np.ndarray | None
            Local map points in world frame (``None`` if map empty).
        prev_points : np.ndarray | None
            Previous frame points in world frame (``None`` on first frame).
        init_pose : np.ndarray
            Initial pose guess ``[x, y, theta]``.
        matcher : ScanMatcher
            Instance used for both registrations.

        Returns
        -------
        final_pose : np.ndarray, shape (3,)
            Estimated pose after dynamic filtering.
        adjusted_saliency : np.ndarray
            Saliency map (unchanged in current implementation; can be
            zeroed-out for dynamic regions in future extensions).
        """
        # ------------------------------------------------------------------
        # 1. Frame-to-frame with saliency
        # ------------------------------------------------------------------
        if prev_points is not None and len(prev_points) > 0:
            pose_f2f = matcher.match(points, saliency, prev_points, init_pose)
        else:
            pose_f2f = init_pose.copy()

        # ------------------------------------------------------------------
        # 2. Frame-to-map *without* saliency (as a consistency check)
        # ------------------------------------------------------------------
        if local_map is not None and len(local_map) > 0:
            zero_sal = np.zeros_like(saliency)
            pose_f2m_ns = matcher.match(points, zero_sal, local_map, pose_f2f)
        else:
            # No map yet → fall back to frame-to-frame
            pose_f2m_ns = pose_f2f.copy()

        # ------------------------------------------------------------------
        # 3. Pose discrepancy test
        # ------------------------------------------------------------------
        trans_err = float(np.hypot(pose_f2f[0] - pose_f2m_ns[0],
                                   pose_f2f[1] - pose_f2m_ns[1]))
        rot_err = abs(self._angle_diff(pose_f2f[2], pose_f2m_ns[2]))

        if trans_err >= self.th_trans or rot_err >= self.th_rot:
            # Dynamic interference detected.
            # Trust the frame-to-map result (which did not use saliency and
            # therefore is less affected by dynamic outliers).
            final_pose = pose_f2m_ns
        else:
            # Static scene → use saliency-weighted frame-to-map for refinement
            if local_map is not None and len(local_map) > 0:
                final_pose = matcher.match(points, saliency, local_map, pose_f2m_ns)
            else:
                final_pose = pose_f2f.copy()

        return final_pose, saliency.copy()

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = a - b
        return (diff + np.pi) % (2.0 * np.pi) - np.pi
