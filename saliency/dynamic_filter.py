"""
动态物体抑制：Saliency-LOAM 论文算法 1 的 2D 简化版
通过对比 frame-to-frame（带显著性）和 frame-to-map（无显著性）的位姿偏差，
检测并抑制动态点。
"""

import numpy as np
from typing import Tuple, Optional
from .utils import pose_distance


class DynamicFilter:
    """
    双估计动态抑制滤波器。
    """

    def __init__(self,
                 th_pose_translation: float = 0.05,
                 th_pose_rotation: float = 0.02):
        """
        Args:
            th_pose_translation: 位移偏差阈值 (m)
            th_pose_rotation: 角度偏差阈值 (rad)
        """
        self.th_t = th_pose_translation
        self.th_r = th_pose_rotation

    def filter(self,
               pose_scan2scan: np.ndarray,
               pose_scan2map: np.ndarray,
               current_points: np.ndarray,
               saliency: np.ndarray,
               point_to_map_dists: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        判断是否发生动态干扰，并过滤 saliency。

        Args:
            pose_scan2scan: 帧间估计位姿 (x,y,theta)
            pose_scan2map: 地图估计位姿 (x,y,theta)
            current_points: (N, 2) 当前帧点
            saliency: (N,) 当前显著性
            point_to_map_dists: (N,) 可选，每个点在 scan2map 下的重投影距离

        Returns:
            (final_saliency, final_pose, dynamic_mask)
            - final_saliency: (N,) 过滤后的显著性（动态点置 0）
            - final_pose: (3,) 推荐的最终位姿
            - dynamic_mask: (N,) bool，True 表示该点被判定为动态
        """
        t_err, r_err = pose_distance(pose_scan2scan, pose_scan2map)

        is_dynamic_scene = (t_err > self.th_t) or (r_err > self.th_r)

        if not is_dynamic_scene:
            # 偏差小，信任帧间估计（显著性约束有效）
            return saliency.copy(), pose_scan2scan.copy(), np.zeros(len(saliency), dtype=bool)

        # 偏差大：存在动态干扰
        # 策略：信任 scan2map（无显著性约束，但对长期地图更鲁棒）
        # 同时尝试识别哪些点导致了偏差
        dynamic_mask = self._identify_dynamic_points(
            current_points, saliency, pose_scan2scan, pose_scan2map, point_to_map_dists
        )

        final_saliency = saliency.copy()
        final_saliency[dynamic_mask] = 0.0

        # 最终位姿：优先用 scan2map，因为它不受动态点显著性误导
        return final_saliency, pose_scan2map.copy(), dynamic_mask

    def _identify_dynamic_points(self,
                                  points: np.ndarray,
                                  saliency: np.ndarray,
                                  pose_s2s: np.ndarray,
                                  pose_s2m: np.ndarray,
                                  dists: Optional[np.ndarray] = None) -> np.ndarray:
        """
        启发式识别动态点：
        1. 显著性高但在两种位姿下重投影误差都大的点 → 可能是动态
        2. 若提供了 dists，直接阈值筛选
        """
        n = len(points)
        mask = np.zeros(n, dtype=bool)

        # 简单启发：显著性 > 0.7 且位于扫描边缘附近的点（远距离）更容易是噪声/动态
        # 这里用更直接的方法：如果提供了 dists，找残差最大的前 10%
        if dists is not None and len(dists) == n:
            threshold = np.percentile(dists, 90)
            mask = (dists > threshold) & (saliency > 0.5)
        else:
            # 备用策略：显著性极高但空间上孤立的点（动态物体往往面积小）
            # 这里简化处理：显著性 > 0.8 的点暂时降权（保守策略）
            mask = saliency > 0.8

        return mask