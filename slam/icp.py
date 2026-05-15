"""
基础 2D PL-ICP（点到线 ICP）
不依赖显著性模块，纯几何匹配
"""

import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Optional


class BasicICP:
    """
    2D Point-to-Line ICP。
    将当前帧点云与参考帧的线特征对齐，估计位姿 (x, y, theta)。
    """

    def __init__(self, 
                 max_iter: int = 50, 
                 tolerance: float = 1e-5,
                 min_match_dist: float = 1.0):
        """
        Args:
            max_iter: 内部最大迭代次数（scipy 内部控制，这里用于提前退出）
            tolerance: 收敛阈值
            min_match_dist: 点到线最大匹配距离，超过视为外点
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.min_match_dist = min_match_dist

    def fit(self, 
            current_points: np.ndarray,
            ref_lines: List[Tuple[np.ndarray, np.ndarray, float]],
            init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """
        执行 ICP 匹配。

        Args:
            current_points: (N, 2) 当前帧扫描点
            ref_lines: [((x1,y1), (x2,y2), weight), ...] 参考帧线特征
            init_pose: (x, y, theta) 初始位姿猜测

        Returns:
            np.ndarray: [x, y, theta] 优化后位姿
        """
        result = least_squares(
            fun=self._residuals,
            x0=np.array(init_pose, dtype=float),
            args=(current_points, ref_lines),
            method='lm',
            max_nfev=self.max_iter
        )
        return result.x

    def transform_points(self, 
                         points: np.ndarray, 
                         pose: np.ndarray) -> np.ndarray:
        """
        用位姿 (x, y, theta) 变换点云。

        Args:
            points: (N, 2)
            pose: (3,) [x, y, theta]

        Returns:
            (N, 2) 变换后点云
        """
        x, y, theta = pose
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        return points @ R.T + np.array([x, y])

    def point_to_line_distance(self, 
                               p: np.ndarray, 
                               line_start: np.ndarray, 
                               line_end: np.ndarray) -> float:
        """
        计算 2D 点到线段的垂直距离（标量）。
        """
        AB = line_end - line_start
        AP = p - line_start
        cross = np.abs(AP[0] * AB[1] - AP[1] * AB[0])
        line_len = np.linalg.norm(AB)
        if line_len < 1e-6:
            return np.linalg.norm(AP)
        return cross / line_len

    def find_nearest_line(self, 
                          p: np.ndarray, 
                          ref_lines: List[Tuple[np.ndarray, np.ndarray, float]]) -> Tuple[int, float]:
        """
        查找距离点 p 最近的线段索引和距离。

        Returns:
            (best_idx, min_distance)
        """
        min_dist = float('inf')
        best_idx = -1

        for i, (A, B, _) in enumerate(ref_lines):
            dist = self.point_to_line_distance(p, A, B)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

        return best_idx, min_dist

    def _residuals(self, 
                   pose: np.ndarray, 
                   current_points: np.ndarray,
                   ref_lines: List[Tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
        """
        计算残差向量，供 least_squares 调用。
        """
        transformed = self.transform_points(current_points, pose)
        residuals = []

        for p in transformed:
            idx, dist = self.find_nearest_line(p, ref_lines)
            if idx >= 0 and dist < self.min_match_dist:
                residuals.append(dist)
            else:
                # 外点给一个大残差但保留梯度
                residuals.append(self.min_match_dist * 2.0)

        return np.array(residuals, dtype=float)