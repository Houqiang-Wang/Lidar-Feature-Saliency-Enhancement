"""
Saliency Estimation Module
==========================
2D LiDAR point saliency computation based on geometric curvature,
intensity gradient, and semantic weight.

Reference: Saliency-LOAM (IEEE TIM 2026), Section II-A
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from core.sensor import LaserScan


class SaliencyEstimator:
    """Compute per-point saliency for an ordered 2D laser scan.

    The fusion follows the 2D hand-crafted rule from Saliency-LOAM:

        S_i = alpha * Norm(c_i) + beta * Norm(|delta I_i|) + gamma * M_sem(i)

    where *Norm* denotes Min-Max normalisation to [0, 1].

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbours for local PCA curvature (default 5).
    alpha, beta, gamma : float
        Fusion weights for curvature, intensity gradient, and semantic term.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        self.k = max(3, k_neighbors)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #

    def compute(
        self,
        scan: LaserScan,
        semantic_mask: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (points_local, valid_mask, saliency) for a single scan.

        Parameters
        ----------
        scan : LaserScan
            Raw laser scan frame.
        semantic_mask : np.ndarray, optional
            Per-point semantic weight (same length as scan).  ``1`` = static,
            ``0`` = dynamic/non-semantic.  If ``None``, all points are treated
            as static (``M_sem = 1``).

        Returns
        -------
        points_local : np.ndarray, shape (M, 2)
            Valid points in the **sensor** frame (x-forward, y-left).
        valid_mask : np.ndarray, shape (N,), dtype=bool
            Boolean mask indexing into the original ``scan.ranges``.
        saliency : np.ndarray, shape (M,)
            Normalised saliency score in ``[0, 1]``.
        """
        points_local, valid_mask = self._scan_to_local_points(scan)
        n_valid = len(points_local)

        if n_valid < self.k:
            saliency = np.zeros(n_valid, dtype=np.float32)
            return points_local, valid_mask, saliency

        # 1. Local curvature via K-NN PCA + scan-order angle curvature
        curvature_knn = self._compute_curvature(points_local)
        curvature_scan = self._compute_scan_angle_curvature(points_local)
        # 扫描顺序转角对尖角更敏感，给予更高权重
        curvature = 0.3 * curvature_knn + 0.7 * curvature_scan
        c_norm = self._min_max_norm(curvature)

        # 2. Intensity gradient (adjacent difference)
        intensities = scan.intensities[valid_mask]
        grad_i = self._compute_intensity_gradient(intensities)
        g_norm = self._min_max_norm(grad_i)

        # 3. Semantic weight
        if semantic_mask is not None and len(semantic_mask) == len(valid_mask):
            m_sem = semantic_mask[valid_mask].astype(np.float32)
        else:
            m_sem = np.ones(n_valid, dtype=np.float32)

        # Fusion
        raw = self.alpha * c_norm + self.beta * g_norm + self.gamma * m_sem
        saliency = self._min_max_norm(raw)

        return points_local, valid_mask, saliency

    # --------------------------------------------------------------------- #
    #  Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _scan_to_local_points(scan: LaserScan) -> Tuple[np.ndarray, np.ndarray]:
        """Convert polar scan to Cartesian **sensor-local** coordinates."""
        valid = np.isfinite(scan.ranges) & (scan.ranges > scan.range_min)
        angles = scan.angles[valid]   # already in sensor frame
        ranges = scan.ranges[valid]
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.column_stack([x, y]).astype(np.float32), valid

    def _compute_curvature(self, points: np.ndarray) -> np.ndarray:
        """Local curvature = lambda_min / lambda_max from K-NN covariance."""
        n = len(points)
        curvature = np.zeros(n, dtype=np.float32)

        for i in range(n):
            # Squared Euclidean distances to all points
            diff = points - points[i]
            dists_sq = np.einsum("ij,ij->i", diff, diff)
            knn_idx = np.argpartition(dists_sq, self.k)[: self.k]

            # Local covariance
            neighbourhood = diff[knn_idx]
            cov = neighbourhood.T @ neighbourhood / self.k

            # Eigenvalues of 2x2 symmetric covariance
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)

            if eigvals[1] > 1e-8:
                curvature[i] = eigvals[0] / eigvals[1]

        return curvature

    def _compute_scan_angle_curvature(self, points: np.ndarray) -> np.ndarray:
        """基于扫描顺序邻域的局部转角曲率。

        对于2D激光雷达，按角度排序的邻居比空间KNN更能准确反映
        局部几何形状。尖角（内角很小）处向量转向接近 π，曲率最大；
        直线段转向接近 0，曲率最小。
        """
        n = len(points)
        curvature = np.zeros(n, dtype=np.float32)
        k = max(2, self.k // 2)  # 前后各k个邻居

        for i in range(n):
            prev_idx = (i - k) % n
            next_idx = (i + k) % n

            p = points[i]
            p_prev = points[prev_idx]
            p_next = points[next_idx]

            v1 = p - p_prev
            v2 = p_next - p
            norm1 = float(np.linalg.norm(v1))
            norm2 = float(np.linalg.norm(v2))

            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                # π - angle: 尖角处 angle 很小 → 曲率接近 π
                curvature[i] = np.pi - angle

        return curvature

    @staticmethod
    def _compute_intensity_gradient(intensities: np.ndarray) -> np.ndarray:
        """Absolute central difference approximating |grad I|."""
        n = len(intensities)
        grad = np.zeros(n, dtype=np.float32)
        if n < 3:
            return grad

        grad[1:-1] = 0.5 * (
            np.abs(intensities[2:] - intensities[1:-1])
            + np.abs(intensities[1:-1] - intensities[:-2])
        )
        grad[0] = np.abs(intensities[1] - intensities[0])
        grad[-1] = np.abs(intensities[-1] - intensities[-2])
        return grad

    @staticmethod
    def _min_max_norm(x: np.ndarray) -> np.ndarray:
        """Min-Max normalise to [0, 1].  All-constant arrays map to 0."""
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
