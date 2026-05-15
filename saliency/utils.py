"""
显著性模块公共工具函数
"""

import numpy as np


def min_max_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Min-Max 归一化到 [0, 1]。
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)


def pose_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> Tuple[float, float]:
    """
    计算两个 2D 位姿的位移差和角度差。

    Returns:
        (translation_error, rotation_error_rad)
    """
    t_err = np.linalg.norm(pose_a[:2] - pose_b[:2])
    # 角度差归一化到 [-pi, pi]
    r_diff = pose_a[2] - pose_b[2]
    r_diff = (r_diff + np.pi) % (2 * np.pi) - np.pi
    r_err = np.abs(r_diff)
    return t_err, r_err