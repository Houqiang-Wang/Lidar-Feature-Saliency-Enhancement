"""
Star Shape Regularizer
======================
将粗糙的障碍物轮廓规则化为 clean 的五角星多边形。

流程：
    1. 角点检测（外凸5个 + 内凹5个）
    2. 分段直线拟合
    3. 相邻直线求交 → 10个顶点
    4. 径向规则化（外5点共圆、内5点共圆）
    5. 输出闭合多边形
"""

from __future__ import annotations

import numpy as np


def regularize_star(pts: np.ndarray,
                    outer_r_tol: float = 0.15,
                    inner_r_tol: float = 0.10) -> np.ndarray | None:
    """把粗糙轮廓规则化为 clean 五角星。

    Parameters
    ----------
    pts : (N, 2)
        输入轮廓点（闭合，首尾不相等）。N 建议 >= 30。
    outer_r_tol, inner_r_tol : float
        外/内顶点半径规则化强度（0=不规则化，1=完全拉到平均半径）。

    Returns
    -------
    (10, 2) 规则化后的五角星顶点，按逆时针排列。
    """
    if pts is None or len(pts) < 20:
        return None

    n = len(pts)

    # ======================================================================
    # 1. 角点检测：计算每个点的外角，找出峰值
    # ======================================================================
    window = 4  # 固定小窗口，避免跨越多个边
    exterior = np.zeros(n)
    for i in range(n):
        p_prev = pts[(i - window) % n]
        p = pts[i]
        p_next = pts[(i + window) % n]
        v1 = p - p_prev
        v2 = p_next - p
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            exterior[i] = np.pi - np.arccos(cos_a)

    # 找局部峰值（角点候选）
    peak_idx = []
    for i in range(n):
        if exterior[i] < np.deg2rad(30):
            continue
        # 局部最大值
        neigh = [(i + k) % n for k in range(-window, window + 1) if k != 0]
        if all(exterior[i] >= exterior[j] for j in neigh):
            peak_idx.append(i)

    if len(peak_idx) < 6:
        return None

    # 按外角大小排序，取前10个最强角点
    peak_idx = sorted(peak_idx, key=lambda i: exterior[i], reverse=True)[:10]
    peak_idx = sorted(peak_idx)  # 恢复轮廓顺序

    # ======================================================================
    # 2. 分段直线拟合（每两个相邻角点之间拟合一条直线）
    # ======================================================================
    lines = []  # list of (point, direction)
    for k in range(len(peak_idx)):
        i_start = peak_idx[k]
        i_end = peak_idx[(k + 1) % len(peak_idx)]

        # 提取段内点（去掉两端各1点，避免角点干扰）
        if i_end > i_start:
            idx = list(range(i_start + 1, i_end))
        else:
            idx = list(range(i_start + 1, n)) + list(range(0, i_end))

        if len(idx) < 2:
            # 段太短，直接用两端点定义直线
            p1, p2 = pts[i_start], pts[i_end]
            lines.append((p1, p2 - p1))
            continue

        seg_pts = pts[idx]
        # PCA 找主方向
        c = np.mean(seg_pts, axis=0)
        cov = np.cov((seg_pts - c).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
        lines.append((c, direction))

    # ======================================================================
    # 3. 相邻直线求交 → 新顶点
    # ======================================================================
    vertices = []
    for k in range(len(lines)):
        c1, d1 = lines[k]
        c2, d2 = lines[(k + 1) % len(lines)]
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
        b = c2 - c1
        try:
            t = np.linalg.solve(A, b)
            intersection = c1 + t[0] * d1
            vertices.append(intersection)
        except np.linalg.LinAlgError:
            # 平行，取中点
            vertices.append((c1 + c2) / 2)

    vertices = np.array(vertices)
    if len(vertices) != 10:
        return None

    # ======================================================================
    # 4. 径向规则化：外5点共圆、内5点共圆
    # ======================================================================
    center = np.mean(vertices, axis=0)
    deltas = vertices - center
    radii = np.linalg.norm(deltas, axis=1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    # 按半径排序区分外点和内点
    sorted_by_r = np.argsort(radii)
    inner_candidates = sorted_by_r[:5]
    outer_candidates = sorted_by_r[5:]

    # 规则化半径
    mean_inner_r = float(np.mean(radii[inner_candidates]))
    mean_outer_r = float(np.mean(radii[outer_candidates]))

    new_radii = radii.copy()
    new_radii[inner_candidates] = radii[inner_candidates] * (1 - inner_r_tol) + mean_inner_r * inner_r_tol
    new_radii[outer_candidates] = radii[outer_candidates] * (1 - outer_r_tol) + mean_outer_r * outer_r_tol

    # 保持原角度，用新半径重建
    reg_vertices = np.column_stack([
        center[0] + new_radii * np.cos(angles),
        center[1] + new_radii * np.sin(angles),
    ])

    # 按角度排序，确保逆时针
    order = np.argsort(np.arctan2(reg_vertices[:, 1] - center[1],
                                   reg_vertices[:, 0] - center[0]))
    reg_vertices = reg_vertices[order]

    return reg_vertices
