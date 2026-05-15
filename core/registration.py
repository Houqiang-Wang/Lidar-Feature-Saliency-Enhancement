"""
Saliency-Weighted Scan Matching
===============================
2D point-to-line registration using scipy.optimize.least_squares (LM).

Reference: Saliency-LOAM Section II-C, formula 6 (2D adaptation)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from typing import Optional


class ScanMatcher:
    """Significant-weighted frame-to-frame / frame-to-map scan matcher.

    The residual for each source point is the **saliency-weighted**
    point-to-line distance to its nearest neighbours in the target cloud:

        e_i = w_i * d_pl
        w_i = (a * S_i**2 + b) / 255.0   (simplified to 0.5 + 0.5 * S_i)

    Parameters
    ----------
    max_corr_dist : float
        Correspondence search radius [m].  Points without neighbours inside
        this radius receive a penalty residual.
    """

    def __init__(self, max_corr_dist: float = 1.0):
        self.max_corr_dist = max_corr_dist

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def match(
        self,
        src_points: np.ndarray,
        src_saliency: np.ndarray,
        tgt_points: np.ndarray,
        init_pose: np.ndarray,
    ) -> np.ndarray:
        """Estimate 2D rigid transform that aligns ``src`` to ``tgt``.

        Parameters
        ----------
        src_points : np.ndarray, shape (N, 2)
            Source point cloud (sensor frame).
        src_saliency : np.ndarray, shape (N,)
            Per-source-point saliency weights.
        tgt_points : np.ndarray, shape (M, 2)
            Target point cloud (world or previous frame).
        init_pose : np.ndarray, shape (3,)
            Initial guess ``[x, y, theta]`` [m, m, rad].

        Returns
        -------
        np.ndarray, shape (3,)
            Optimised pose ``[x, y, theta]``.
        """
        if len(src_points) < 3 or len(tgt_points) < 3:
            return init_pose.copy()

        # Build KD-Tree once on the target
        self._tgt_tree = cKDTree(tgt_points)
        self._tgt_points = tgt_points

        result = least_squares(
            fun=self._residuals,
            x0=init_pose.astype(np.float64),
            args=(src_points, src_saliency),
            method="lm",
            max_nfev=50,
        )

        # Clean up temporary references
        del self._tgt_tree, self._tgt_points
        return result.x.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Residual computation
    # ------------------------------------------------------------------ #

    def _residuals(
        self,
        pose: np.ndarray,
        src_points: np.ndarray,
        src_saliency: np.ndarray,
    ) -> np.ndarray:
        """Return fixed-length residual vector (one entry per source point)."""
        n = len(src_points)
        residuals = np.full(n, self.max_corr_dist, dtype=np.float64)

        # 1. Transform source points to target frame
        cos_t, sin_t = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        transformed = (R @ src_points.T).T + pose[:2]

        # 2. Query nearest 3 neighbours for local line fitting
        dists, idxs = self._tgt_tree.query(
            transformed, k=3, distance_upper_bound=self.max_corr_dist
        )

        # Valid mask: all 3 neighbours found within radius
        valid = ~np.any(np.isinf(dists), axis=1)
        if not np.any(valid):
            return residuals

        v_idx = np.where(valid)[0]
        v_tf = transformed[valid]
        v_ix = idxs[valid]

        # End-points of the local line (1st and 3rd neighbour)
        p0 = self._tgt_points[v_ix[:, 0]]
        p2 = self._tgt_points[v_ix[:, 2]]
        line_dir = p2 - p0
        line_len = np.hypot(line_dir[:, 0], line_dir[:, 1])

        # Skip degenerate lines
        good = line_len > 1e-6
        if not np.any(good):
            return residuals

        g_idx = v_idx[good]
        ap = v_tf[good] - p0[good]
        cross = np.abs(
            ap[:, 0] * line_dir[good, 1] - ap[:, 1] * line_dir[good, 0]
        )
        d_pl = cross / line_len[good]

        # Saliency weight (文档公式 6 的 2D 简化) → [0.5, 1.0]
        S = src_saliency[g_idx]
        w = 0.5 + 0.5 * S
        residuals[g_idx] = w * d_pl

        return residuals
