"""
加权 PL-ICP：Saliency-LOAM 论文 II-C 的 2D 实现
继承 BasicICP，仅在残差计算中加入显著性权重 w = (a*S^2 + b)/255
"""

import numpy as np
from typing import List, Tuple
from slam.icp import BasicICP


class SaliencyICP(BasicICP):
    """
    带显著性权重的 ICP。
    公式：e_i = w_i * d_i,  w_i = (a * S_i^2 + b) / 255
    """

    def __init__(self,
                 a: float = 200.0,
                 b: float = 55.0,
                 **kwargs):
        """
        Args:
            a, b: 显著性权重系数，控制 w 的范围
            **kwargs: 传递给 BasicICP 的参数（max_iter, tolerance 等）
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def fit(self,
            current_points: np.ndarray,
            saliency: np.ndarray,
            ref_lines: List[Tuple[np.ndarray, np.ndarray, float]],
            init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """
        执行加权 ICP。

        Args:
            current_points: (N, 2)
            saliency: (N,) 每个点的显著性权重 [0,1]
            ref_lines: 参考帧线特征
            init_pose: 初始位姿

        Returns:
            [x, y, theta]
        """
        # 确保 saliency 维度正确
        assert len(saliency) == len(current_points), "saliency 长度必须等于点数"

        result = least_squares(
            fun=self._weighted_residuals,
            x0=np.array(init_pose, dtype=float),
            args=(current_points, saliency, ref_lines),
            method='lm',
            max_nfev=self.max_iter
        )
        return result.x

    def _weighted_residuals(self,
                              pose: np.ndarray,
                              current_points: np.ndarray,
                              saliency: np.ndarray,
                              ref_lines: List[Tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
        """
        计算加权残差。
        """
        transformed = self.transform_points(current_points, pose)
        residuals = []

        for i, p in enumerate(transformed):
            idx, dist = self.find_nearest_line(p, ref_lines)
            if idx >= 0 and dist < self.min_match_dist:
                # 显著性权重（论文公式 2D 化）
                w = (self.a * (saliency[i] ** 2) + self.b) / 255.0
                residuals.append(w * dist)
            else:
                residuals.append(self.min_match_dist * 2.0)

        return np.array(residuals, dtype=float)