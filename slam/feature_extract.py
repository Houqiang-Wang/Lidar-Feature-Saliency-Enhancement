"""
2D 特征提取模块
基于 Split-and-Merge 从有序扫描点中提取线段和角点
"""

import numpy as np
from typing import List, Tuple, Optional


class FeatureExtractor2D:
    """
    从 2D 激光扫描点中提取线特征和角点特征。
    假设输入点按扫描角度有序排列（RPLIDAR 标准输出格式）。
    """

    def __init__(self, 
                 line_threshold: float = 0.05, 
                 min_line_points: int = 5,
                 corner_angle_threshold: float = np.deg2rad(30)):
        """
        Args:
            line_threshold: Split-and-Merge 的点到线距离阈值 (m)
            min_line_points: 一条有效线段最少包含的点数
            corner_angle_threshold: 判定为角点的最小夹角 (rad)
        """
        self.line_threshold = line_threshold
        self.min_line_points = min_line_points
        self.corner_angle_threshold = corner_angle_threshold

    def extract(self, 
                points: np.ndarray, 
                saliency: Optional[np.ndarray] = None) -> dict:
        """
        提取 2D 特征。

        Args:
            points: (N, 2) 扫描点坐标，要求按角度有序
            saliency: (N,) 可选的显著性权重，仅用于给特征附加平均权重

        Returns:
            dict: {
                'lines': [((x1,y1), (x2,y2), avg_saliency), ...],
                'corners': [(x, y, saliency), ...]
            }
        """
        if len(points) < 3:
            return {'lines': [], 'corners': []}

        # 1. Split-and-Merge 提取线段
        point_indices = np.arange(len(points))
        segments = self._split_and_merge(points, point_indices)

        # 2. 从线段端点和分割点提取角点
        lines = []
        corners = []
        seen_corner = set()

        for seg_idx, idx_list in enumerate(segments):
            if len(idx_list) < self.min_line_points:
                continue

            seg_pts = points[idx_list]
            line_start = seg_pts[0]
            line_end = seg_pts[-1]
            
            # 计算该线段的平均显著性（如果提供）
            avg_sal = 1.0
            if saliency is not None:
                avg_sal = float(np.mean(saliency[idx_list]))

            lines.append((line_start, line_end, avg_sal))

            # 角点：线段的两个端点（如果是真实分割点而非首尾）
            for idx in [idx_list[0], idx_list[-1]]:
                if idx == 0 or idx == len(points) - 1:
                    continue  # 跳过扫描链的物理首尾，避免虚假角点
                if idx not in seen_corner:
                    seen_corner.add(idx)
                    sal = float(saliency[idx]) if saliency is not None else 1.0
                    corners.append((points[idx][0], points[idx][1], sal))

        # 3. 额外角点：局部曲率极大值（补充 Split-and-Merge 漏掉的尖锐角）
        extra_corners = self._detect_curvature_corners(points, saliency)
        corners.extend(extra_corners)

        return {'lines': lines, 'corners': corners}

    def _split_and_merge(self, 
                         points: np.ndarray, 
                         indices: np.ndarray) -> List[np.ndarray]:
        """
        递归 Split-and-Merge 算法。
        返回线段索引列表。
        """
        if len(indices) < 3:
            return [indices]

        pts = points[indices]
        start, end = pts[0], pts[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return [indices]

        # 计算所有点到首尾连线的垂直距离
        # 叉积公式：|(P - A) x (B - A)| / |B - A|
        cross = np.abs(np.cross(pts - start, line_vec))
        distances = cross / line_len

        max_idx = np.argmax(distances)
        max_dist = distances[max_idx]

        if max_dist > self.line_threshold:
            # 在最大距离点处分裂
            left = self._split_and_merge(points, indices[:max_idx+1])
            right = self._split_and_merge(points, indices[max_idx:])
            return left + right
        else:
            return [indices]

    def _detect_curvature_corners(self, 
                                  points: np.ndarray, 
                                  saliency: Optional[np.ndarray] = None,
                                  window: int = 3) -> List[Tuple[float, float, float]]:
        """
        基于局部向量夹角检测曲率角点。
        避开首尾，避免 0°/360° 接缝伪影。
        """
        corners = []
        n = len(points)
        if n < 2 * window + 1:
            return corners

        # 只处理中间段，避开首尾
        for i in range(window, n - window):
            prev_vec = points[i] - points[i - window]
            next_vec = points[i + window] - points[i]
            
            # 计算夹角
            norm_p = np.linalg.norm(prev_vec)
            norm_n = np.linalg.norm(next_vec)
            if norm_p < 1e-6 or norm_n < 1e-6:
                continue

            cos_angle = np.dot(prev_vec, next_vec) / (norm_p * norm_n)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # 夹角越小（cos 越大），越不是角点；夹角大（转向明显）→ 角点
            if angle > self.corner_angle_threshold:
                sal = float(saliency[i]) if saliency is not None else 1.0
                corners.append((points[i][0], points[i][1], sal))

        return corners