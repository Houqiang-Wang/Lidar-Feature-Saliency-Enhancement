"""
2D 点云显著性计算（Saliency-LOAM 论文 II-A/B 的 2D 手工版）
无需神经网络，直接利用几何曲率、强度梯度、语义标签融合
"""

import numpy as np
from typing import Optional
from .utils import min_max_norm


class SaliencyCalculator:
    """
    为 2D 激光扫描点计算显著性权重 S_i ∈ [0,1]。
    """

    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 curvature_k: int = 5,
                 use_intensity: bool = False):
        """
        Args:
            alpha: 几何曲率权重
            beta: 强度梯度权重
            gamma: 语义权重
            curvature_k: 计算局部曲率的 K 近邻数（空间邻域，避免角度排序接缝问题）
            use_intensity: 是否使用强度信息（若传感器无 intensity 则自动忽略）
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.curvature_k = curvature_k
        self.use_intensity = use_intensity

    def compute(self,
                points: np.ndarray,
                ranges: Optional[np.ndarray] = None,
                intensities: Optional[np.ndarray] = None,
                semantic_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算显著性。

        Args:
            points: (N, 2) 扫描点坐标
            ranges: (N,) 距离值，可选（若提供可加速 KNN）
            intensities: (N,) 反射强度，可选
            semantic_labels: (N,) 语义标签字符串数组，如 'wall', 'weed', 'dynamic'

        Returns:
            (N,) 显著性权重，范围 [0, 1]
        """
        n = len(points)
        if n == 0:
            return np.array([])

        # 1. 几何显著性：基于空间 KNN 的局部曲率
        geo_sal = self._geometric_saliency(points)

        # 2. 强度显著性
        if self.use_intensity and intensities is not None:
            int_sal = self._intensity_saliency(intensities)
        else:
            int_sal = np.zeros(n)

        # 3. 语义显著性
        if semantic_labels is not None:
            sem_sal = self._semantic_saliency(semantic_labels)
        else:
            sem_sal = np.ones(n) * 0.5  # 无标签时默认中等显著性

        # 4. 加权融合
        saliency = (self.alpha * geo_sal +
                    self.beta * int_sal +
                    self.gamma * sem_sal)

        # 归一化并截断
        saliency = min_max_norm(saliency)
        return np.clip(saliency, 0.0, 1.0)

    def _geometric_saliency(self, points: np.ndarray) -> np.ndarray:
        """
        基于空间 K 近邻的局部曲率。
        曲率越大（点越偏离局部平面/直线），显著性越高（角点/边缘）。
        注意：这里用空间距离找近邻，而非数组下标，避免 0°/360° 接缝伪影。
        """
        n = len(points)
        curvature = np.zeros(n)

        for i in range(n):
            # 计算该点到所有其他点的距离
            dists = np.linalg.norm(points - points[i], axis=1)
            # 找 K 个最近邻（包含自己，所以取 K+1）
            k = min(self.curvature_k + 1, n)
            nearest_idx = np.argpartition(dists, k-1)[:k]

            # 用 PCA 分析局部形状：最小特征值对应曲率
            neighbors = points[nearest_idx]
            centered = neighbors - np.mean(neighbors, axis=0)
            if len(centered) < 2:
                continue

            # 协方差矩阵特征值
            cov = np.cov(centered.T)
            if cov.ndim < 2:
                continue
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)

            # 曲率近似：小特征值 / 大特征值（越小越像直线，越大越像角点）
            # 这里取比值作为"非直线度"
            if eigvals[-1] > 1e-6:
                curvature[i] = 1.0 - eigvals[0] / (eigvals[-1] + 1e-6)

        return min_max_norm(curvature)

    def _intensity_saliency(self, intensities: np.ndarray) -> np.ndarray:
        """
        强度梯度绝对值作为显著性。边缘处材质反射突变 → 梯度大 → 显著性高。
        """
        # 计算相邻差分（环形处理，避免首尾跳变）
        diff = np.abs(np.diff(intensities, prepend=intensities[-1]))
        # 另一种：前后平均差
        grad = np.zeros_like(intensities, dtype=float)
        n = len(intensities)
        for i in range(n):
            prev = intensities[(i - 1) % n]
            next = intensities[(i + 1) % n]
            grad[i] = abs(intensities[i] - prev) + abs(intensities[i] - next)

        return min_max_norm(grad)

    def _semantic_saliency(self, labels: np.ndarray) -> np.ndarray:
        """
        根据语义标签分配显著性权重。
        静态结构高权重，动态/噪声低权重。
        """
        # 标签到权重的映射
        weight_map = {
            'wall': 1.0,
            'pillar': 1.0,
            'static': 1.0,
            'obstacle': 0.8,
            'unknown': 0.5,
            'ground': 0.3,
            'weed': 0.2,
            'dynamic': 0.1,
            'noise': 0.0,
        }

        saliency = np.zeros(len(labels))
        for i, lab in enumerate(labels):
            saliency[i] = weight_map.get(str(lab).lower(), 0.5)

        return saliency