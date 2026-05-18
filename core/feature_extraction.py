"""
2D Line Feature Extraction
==========================
Split-and-Merge algorithm for extracting line segments from an ordered
2D laser scan point cloud.

Reference: Saliency-LOAM 2D migration checklist – "线特征提取"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class LineSegment:
    """A 2D line segment extracted from a laser scan.

    Attributes
    ----------
    endpoints : np.ndarray, shape (2, 2)
        ``[[x_start, y_start], [x_end, y_end]]`` in the **sensor** frame.
    point_indices : np.ndarray
        Indices of the supporting points in the original ordered scan.
    saliency_mean : float
        Average saliency of supporting points (used for weighted matching).
    """

    endpoints: np.ndarray
    point_indices: np.ndarray
    saliency_mean: float


class LineFeatureExtractor:
    """Split-and-Merge line extractor.

    Parameters
    ----------
    split_thresh : float
        Maximum perpendicular distance [m] from a point to the candidate
        line before the segment is split.
    min_points : int
        Minimum number of points required to form a valid segment.
    """

    def __init__(self, split_thresh: float = 0.05, min_points: int = 5):
        self.split_thresh = split_thresh
        self.min_points = min_points

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def extract(
        self, points: np.ndarray, saliency: np.ndarray
    ) -> List[LineSegment]:
        """Extract line segments from an *ordered* point cloud.

        Parameters
        ----------
        points : np.ndarray, shape (N, 2)
            Ordered 2D points (e.g. counter-clockwise sweep).
        saliency : np.ndarray, shape (N,)
            Per-point saliency values.

        Returns
        -------
        List[LineSegment]
            Non-overlapping line segments ordered along the scan.
        """
        if len(points) < self.min_points:
            return []

        segments: List[LineSegment] = []
        indices = np.arange(len(points), dtype=np.int32)
        self._split_and_merge(points, saliency, indices, segments)
        return segments

    # ------------------------------------------------------------------ #
    #  Internal recursion
    # ------------------------------------------------------------------ #

    def _split_and_merge(
        self,
        points: np.ndarray,
        saliency: np.ndarray,
        indices: np.ndarray,
        segments: List[LineSegment],
    ) -> None:
        """Recursively split until all points satisfy the distance threshold."""
        if len(indices) < self.min_points:
            return

        seg_pts = points[indices]

        # Candidate line = chord from first to last point of the segment
        p_start, p_end = seg_pts[0], seg_pts[-1]
        chord = p_end - p_start
        chord_len = float(np.hypot(chord[0], chord[1]))

        # Relaxed threshold: 1 cm instead of 1 μm to avoid skipping
        # closed loops (e.g. 360° scans where first/last points overlap)
        if chord_len < 1e-2:
            # Use the pair of points with maximum separation instead
            dists = np.linalg.norm(seg_pts[:, None] - seg_pts[None, :], axis=2)
            i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
            p_start, p_end = seg_pts[i_max], seg_pts[j_max]
            chord = p_end - p_start
            chord_len = float(np.hypot(chord[0], chord[1]))
            if chord_len < 1e-2:
                return

        # Find point with maximum perpendicular distance to the chord
        max_dist = 0.0
        split_local_idx = -1

        for i, pt in enumerate(seg_pts):
            ap = pt - p_start
            # 2D cross product magnitude = |AP x AB|
            cross = abs(ap[0] * chord[1] - ap[1] * chord[0])
            dist = cross / chord_len
            if dist > max_dist:
                max_dist = dist
                split_local_idx = i

        # If the farthest point exceeds the threshold, split at that point
        if (
            max_dist > self.split_thresh
            and 0 < split_local_idx < len(indices) - 1
        ):
            self._split_and_merge(
                points, saliency, indices[: split_local_idx + 1], segments
            )
            self._split_and_merge(
                points, saliency, indices[split_local_idx:], segments
            )
        else:
            # Accept the segment
            endpoints = np.stack([p_start, p_end]).astype(np.float32)
            seg_saliency = float(saliency[indices].mean())
            segments.append(
                LineSegment(
                    endpoints=endpoints,
                    point_indices=indices.copy(),
                    saliency_mean=seg_saliency,
                )
            )
