"""
Star Shape Fitter
=================
将粗糙轮廓直接拟合为参数化理想五角星。

五角星参数 (cx, cy, R, r, theta)
    cx, cy : 中心坐标
    R      : 外顶点半径（5个尖角）
    r      : 内顶点半径（5个内凹角）
    theta  : 整体旋转角

优化目标：最小化轮廓点到最近五角星边的垂直距离。
"""

from __future__ import annotations

import numpy as np


def generate_star_vertices(cx: float, cy: float, R: float, r: float,
                           theta: float) -> np.ndarray:
    """生成五角星10个顶点，按逆时针排列。"""
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + theta
    radii = np.array([R if i % 2 == 0 else r for i in range(10)])
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    return np.column_stack([x, y])


def point_to_segment_distance_sq(pxy: np.ndarray, a: np.ndarray,
                                  b: np.ndarray) -> np.ndarray:
    """点 p 到线段 ab 的最短距离平方（向量化）。"""
    ab = b - a
    ap = pxy - a
    ab_b = np.broadcast_to(ab, ap.shape)
    t = np.clip(np.einsum('ij,ij->i', ap, ab_b) /
                (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    closest = a + t[:, None] * ab
    diff = pxy - closest
    return np.einsum('ij,ij->i', diff, diff)


def star_residuals(params: np.ndarray, pts: np.ndarray) -> float:
    """轮廓点到五角星最近边的均方距离。"""
    cx, cy, R, r, theta = params
    if R <= r or r <= 0:
        return 1e6
    verts = generate_star_vertices(cx, cy, R, r, theta)
    dists_sq = np.full(len(pts), np.inf)
    for i in range(10):
        a = verts[i]
        b = verts[(i + 1) % 10]
        d2 = point_to_segment_distance_sq(pts, a, b)
        dists_sq = np.minimum(dists_sq, d2)
    return float(np.mean(dists_sq))


def fit_star(pts: np.ndarray,
             n_theta: int = 72,
             refine_steps: int = 200,
             lr: float = 0.02) -> np.ndarray | None:
    """把轮廓点拟合为理想五角星，返回10个规则顶点。

    Parameters
    ----------
    pts : (N, 2)
        输入轮廓点。
    n_theta : int
        旋转角网格搜索步数。
    refine_steps : int
        梯度下降精调步数。
    lr : float
        学习率。

    Returns
    -------
    (10, 2) 拟合后的五角星顶点，逆时针排列。
    """
    if pts is None or len(pts) < 20:
        return None

    # ------------------------------------------------------------------
    # 1. 初始估计
    # ------------------------------------------------------------------
    center = np.mean(pts, axis=0)
    deltas = pts - center
    radii = np.linalg.norm(deltas, axis=1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    # 外半径 ≈ 最大距离，内半径 ≈ 最小距离
    R0 = float(np.percentile(radii, 95))
    r0 = float(np.percentile(radii, 20))

    # ------------------------------------------------------------------
    # 2. 粗搜索：在旋转角上做网格搜索
    # ------------------------------------------------------------------
    best_cost = np.inf
    best_params = None
    theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    for theta in theta_grid:
        cost = star_residuals(np.array([center[0], center[1], R0, r0, theta]), pts)
        if cost < best_cost:
            best_cost = cost
            best_params = np.array([center[0], center[1], R0, r0, theta])

    # ------------------------------------------------------------------
    # 3. 精调：简单的坐标下降（手动实现，无需 scipy）
    # ------------------------------------------------------------------
    params = best_params.copy().astype(np.float64)
    current_cost = best_cost

    # 可学习参数及其合理范围
    mins = np.array([center[0] - 1.0, center[1] - 1.0, 0.3, 0.05, -np.pi])
    maxs = np.array([center[0] + 1.0, center[1] + 1.0, 2.5, 1.0, 3 * np.pi])
    step_scales = np.array([0.01, 0.01, 0.02, 0.01, 0.02])

    for step in range(refine_steps):
        improved = False
        for idx in range(5):
            delta = step_scales[idx] * lr * (1.0 - step / refine_steps)
            for sign in [-1, 1]:
                trial = params.copy()
                trial[idx] += sign * delta
                trial[idx] = np.clip(trial[idx], mins[idx], maxs[idx])
                # 保持 R > r > 0
                if trial[2] <= trial[3] + 0.05:
                    continue
                cost = star_residuals(trial, pts)
                if cost < current_cost:
                    current_cost = cost
                    params = trial
                    improved = True
                    break
        if not improved and step > refine_steps // 2:
            # 后期收敛困难，缩小步长继续
            step_scales *= 0.5

    cx, cy, R, r, theta = params
    return generate_star_vertices(cx, cy, R, r, theta)
