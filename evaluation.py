"""
论文对比实验：五角星特征识别 — Baseline vs Saliency-Enhanced
===============================================================
生成论文所需的对比图和定量数据表。

运行：
    python evaluation.py
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from core.environment import MapEnvironment
from core.sensor import LidarA1, LidarConfig
from core.saliency import SaliencyEstimator
from core.feature_extraction import LineFeatureExtractor, LineSegment

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
#  工具函数
# --------------------------------------------------------------------------- #

def scan_to_local_points(scan) -> Tuple[np.ndarray, np.ndarray]:
    """Convert polar scan to Cartesian sensor-local coordinates."""
    valid = np.isfinite(scan.ranges) & (scan.ranges > scan.range_min)
    angles = scan.angles[valid]
    ranges = scan.ranges[valid]
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.column_stack([x, y]).astype(np.float32), valid


def local_to_world(pose: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
    """Transform points from sensor frame to world frame."""
    cos_t, sin_t = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return (R @ pts_local.T).T + pose[:2]


def get_star_ground_truth(center: Tuple[float, float] = (10.0, 10.0)) -> np.ndarray:
    """Return the true star outline vertices (world coordinates, closed polygon)."""
    cx, cy = center
    R, r = 0.6, 0.2
    n = 10
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    radii = np.array([R if i % 2 == 0 else r for i in range(n)])
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    pts = np.column_stack([x, y])
    return np.vstack([pts, pts[0]])


def in_star_region(
    points: np.ndarray, center: Tuple[float, float] = (10.0, 10.0), radius: float = 1.5
) -> np.ndarray:
    """Boolean mask for points inside a circular ROI around the star."""
    return np.linalg.norm(points - np.array(center), axis=1) < radius


def segment_fully_in_region(
    seg_endpoints: np.ndarray, center: Tuple[float, float] = (10.0, 10.0), radius: float = 1.5
) -> bool:
    """Check whether *both* endpoints of a segment lie inside the star ROI."""
    dists = np.linalg.norm(seg_endpoints - np.array(center), axis=1)
    return bool(np.all(dists < radius))


def contour_matching_error(points: np.ndarray, gt_contour: np.ndarray) -> float:
    """Mean distance from each point to the nearest ground-truth contour vertex."""
    if len(points) == 0:
        return float("inf")
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_contour)
    dists, _ = tree.query(points, k=1)
    return float(np.mean(dists))


# --------------------------------------------------------------------------- #
#  主实验
# --------------------------------------------------------------------------- #

def run_comparison() -> None:
    env = MapEnvironment()
    
    # ===== 高噪声配置：提升环境复杂度 =====
    # 增加噪声比例：近距离从2%→4%，远距离从3%→5%，模拟真实复杂环境
    high_noise_config = LidarConfig(
        noise_ratio_near=0.04,      # 从 0.02 增加到 0.04
        noise_ratio_far=0.05,       # 从 0.03 增加到 0.05
        angle_noise_std=np.deg2rad(0.4),  # 从 0.25° 增加到 0.4°
        dropout_rate=0.01,           # 增加到 1% 的丢包率
    )
    lidar = LidarA1(config=high_noise_config)

    # ===== 优化显著性估计：强化角点/边缘检测 =====
    # 提高 alpha（曲率权重）：0.7 → 0.88，强化几何特征
    # 降低 beta（强度权重）：0.1 → 0.03，减少强度干扰
    # 降低 gamma（语义权重）：0.2 → 0.09，减少非几何因素
    sal_est = SaliencyEstimator(k_neighbors=4, alpha=0.88, beta=0.03, gamma=0.09)
    
    # ===== 优化线特征提取：提高精细度 =====
    # 降低分割阈值：0.03 → 0.012，提高角点识别精度
    # 最小点数：3 → 2，允许极短的特征边
    feat = LineFeatureExtractor(split_thresh=0.012, min_points=2)

    # 机器人正对五角星（从正南方 6 m 处）
    pose = np.array([10.0, 4.0, np.pi / 2], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 1. Collect 5 frames (simulating temporal multi-scan fusion)
    # ------------------------------------------------------------------ #
    frames: List[dict] = []
    for _ in range(5):
        scan = lidar.scan(pose, env)
        pts_local, valid = scan_to_local_points(scan)
        _, _, saliency = sal_est.compute(scan)
        pts_world = local_to_world(pose, pts_local)
        frames.append(
            {
                "points_local": pts_local,
                "points_world": pts_world,
                "saliency": saliency,
            }
        )

    # ------------------------------------------------------------------ #
    # 2. Single-frame analysis
    # ------------------------------------------------------------------ #
    f0 = frames[0]
    base_pts = f0["points_world"]
    base_pts_local = f0["points_local"]

    # ===== 优化显著性过滤策略 =====
    # 采用自适应阈值：取显著性的 25% 分位数而非固定值 0.25
    # 这样在高噪声环境下能自动调整，保留最显著的特征点
    sal_thresh = np.percentile(f0["saliency"], 25)  # 动态阈值
    sal_mask = f0["saliency"] > sal_thresh
    sal_pts_local = base_pts_local[sal_mask]
    sal_pts = base_pts[sal_mask]
    sal_values = f0["saliency"][sal_mask]

    # Baseline line extraction (on full cloud)
    base_segments = feat.extract(base_pts_local, np.ones(len(base_pts_local), dtype=np.float32))
    base_segs_world = [local_to_world(pose, seg.endpoints) for seg in base_segments]

    # Saliency-weighted line extraction (on filtered cloud)
    sal_segments = feat.extract(sal_pts_local, sal_values.astype(np.float32))
    sal_segs_world = [local_to_world(pose, seg.endpoints) for seg in sal_segments]

    # Region masks
    star_mask_base = in_star_region(base_pts)
    star_mask_sal = in_star_region(sal_pts)

    # Ground truth contour
    star_gt = get_star_ground_truth()

    # ================================================================-- #
    # 3. Multi-frame fusion
    # ================================================================-- #
    fused_pts = np.vstack([fr["points_world"] for fr in frames])
    fused_sal = np.concatenate([fr["saliency"] for fr in frames])
    
    # ===== 融合也采用自适应阈值 =====
    fused_thresh = np.percentile(fused_sal, 25)  # 融合数据的动态阈值
    fused_mask = fused_sal > fused_thresh
    fused_star_mask = in_star_region(fused_pts)

    # Extract features on fused cloud
    fused_pts_local = np.vstack([fr["points_local"] for fr in frames])
    fused_segments = feat.extract(fused_pts_local, fused_sal.astype(np.float32))
    fused_segs_world = [local_to_world(pose, seg.endpoints) for seg in fused_segments]

    # ------------------------------------------------------------------ #
    # 4. Quantitative metrics
    # ------------------------------------------------------------------ #
    base_star_pts = int(np.sum(star_mask_base))
    sal_star_pts = int(np.sum(star_mask_sal))
    fused_star_pts = int(np.sum(fused_star_mask))
    fused_high_sal_star_pts = int(np.sum(fused_star_mask & fused_mask))

    base_star_segs = sum(
        1 for seg in base_segs_world if segment_fully_in_region(seg, radius=1.5)
    )
    sal_star_segs = sum(
        1 for seg in sal_segs_world if segment_fully_in_region(seg, radius=1.5)
    )
    fused_star_segs = sum(
        1 for seg in fused_segs_world if segment_fully_in_region(seg, radius=1.5)
    )

    err_base = contour_matching_error(base_pts[star_mask_base], star_gt[:-1])
    err_sal = contour_matching_error(sal_pts[star_mask_sal], star_gt[:-1])
    err_fused = contour_matching_error(
        fused_pts[fused_star_mask & fused_mask], star_gt[:-1]
    )

    # ------------------------------------------------------------------ #
    # 5. Visualization — Top row: qualitative comparison (star zoom)
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.35)

    def _draw_star_panel(ax, pts, segs, title, color_pts, color_segs, pt_size=60,
                         seg_width=2.5, show_gt=True, annotate_text=""):
        """Helper to draw a consistent star close-up panel."""
        if show_gt:
            ax.plot(star_gt[:, 0], star_gt[:, 1], "g--", linewidth=2.5, label="Ground Truth")
        # Draw star-region points with large markers
        ax.scatter(pts[:, 0], pts[:, 1], s=pt_size, c=color_pts, alpha=0.85,
                   edgecolors="black", linewidths=0.3, zorder=5)
        # Draw only fully-contained segments with thick lines
        drawn = False
        for seg in segs:
            if segment_fully_in_region(seg, radius=1.5):
                lbl = "线特征" if not drawn else None
                ax.plot(seg[:, 0], seg[:, 1], color=color_segs, linewidth=seg_width,
                        solid_capstyle="round", label=lbl, zorder=4)
                drawn = True
        # Draw a subtle circular ROI background
        circle = plt.Circle((10, 10), 1.5, color="lightgray", alpha=0.15, zorder=1)
        ax.add_patch(circle)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(8.3, 11.7)
        ax.set_ylim(8.3, 11.7)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)
        if annotate_text:
            ax.text(0.97, 0.03, annotate_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
        if drawn or show_gt:
            ax.legend(loc="upper left", fontsize=9)

    # (a) Baseline single-frame
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_star_panel(
        ax_a, base_pts[star_mask_base], base_segs_world,
        "(a) Baseline — 单帧", "#A0A0A0", "#0066CC",
        annotate_text=f"点数: {base_star_pts}\n线段: {base_star_segs}\n误差: {err_base:.3f} m"
    )

    # (b) Ours single-frame
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_star_panel(
        ax_b, sal_pts[star_mask_sal], sal_segs_world,
        "(b) Ours — 单帧 (显著性>0.25)", sal_values[star_mask_sal], "#CC0000",
        pt_size=80, annotate_text=f"点数: {sal_star_pts}\n线段: {sal_star_segs}\n误差: {err_sal:.3f} m"
    )

    # (c) Ours 5-frame fusion
    ax_c = fig.add_subplot(gs[0, 2])
    fused_star_pts_arr = fused_pts[fused_star_mask & fused_mask]
    fused_star_sal_arr = fused_sal[fused_star_mask & fused_mask]
    _draw_star_panel(
        ax_c, fused_star_pts_arr, fused_segs_world,
        "(c) Ours — 5帧时序融合", fused_star_sal_arr, "#CC0000",
        pt_size=80, annotate_text=f"点数: {fused_high_sal_star_pts}\n线段: {fused_star_segs}\n误差: {err_fused:.3f} m"
    )

    # ------------------------------------------------------------------ #
    # 6. Visualization — Bottom row: quantitative bar charts
    # ------------------------------------------------------------------ #
    categories = ["Baseline\n单帧", "Ours\n单帧", "Ours\n5帧融合"]
    x = np.arange(len(categories))
    width = 0.35

    # (d) Point count bar chart
    ax_d = fig.add_subplot(gs[1, 0])
    bar_pts = [base_star_pts, sal_star_pts, fused_high_sal_star_pts]
    bars = ax_d.bar(x, bar_pts, width=0.5, color=["#A0A0A0", "#FF6666", "#CC0000"],
                    edgecolor="black", linewidth=1.2)
    ax_d.set_ylabel("点数", fontsize=12)
    ax_d.set_title("(d) 五角星区域内点云数量", fontsize=13, fontweight="bold")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(categories, fontsize=10)
    ax_d.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        ax_d.annotate(f"{int(height)}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=12, fontweight="bold")

    # (e) Segment count bar chart
    ax_e = fig.add_subplot(gs[1, 1])
    bar_segs = [base_star_segs, sal_star_segs, fused_star_segs]
    bars = ax_e.bar(x, bar_segs, width=0.5, color=["#A0A0A0", "#FF6666", "#CC0000"],
                    edgecolor="black", linewidth=1.2)
    ax_e.set_ylabel("线段数", fontsize=12)
    ax_e.set_title("(e) 五角星区域内提取线段数", fontsize=13, fontweight="bold")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(categories, fontsize=10)
    ax_e.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        ax_e.annotate(f"{int(height)}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=12, fontweight="bold")

    # (f) Contour error bar chart
    ax_f = fig.add_subplot(gs[1, 2])
    bar_err = [err_base, err_sal, err_fused]
    bars = ax_f.bar(x, bar_err, width=0.5, color=["#A0A0A0", "#FF6666", "#CC0000"],
                    edgecolor="black", linewidth=1.2)
    ax_f.set_ylabel("平均距离 (m)", fontsize=12)
    ax_f.set_title("(f) 轮廓匹配误差", fontsize=13, fontweight="bold")
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(categories, fontsize=10)
    ax_f.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        height = bar.get_height()
        ax_f.annotate(f"{height:.4f}",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.savefig("paper_comparison.png", dpi=300, bbox_inches="tight")
    print("对比图已保存：paper_comparison.png")

    # ------------------------------------------------------------------ #
    # 7. Print quantitative table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72, file=sys.stderr)
    print("表1  五角星特征识别对比实验定量结果", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(f"{'指标':<46s} {'Baseline':>12s} {'Ours单帧':>12s} {'Ours融合':>12s}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    rows = [
        ("五角星区域内点数", base_star_pts, sal_star_pts, fused_high_sal_star_pts),
        ("五角星区域内线段数", base_star_segs, sal_star_segs, fused_star_segs),
        ("轮廓匹配误差 (m)", err_base, err_sal, err_fused),
    ]
    for name, v1, v2, v3 in rows:
        print(f"{name:<46s} {str(v1):>12s} {str(v2):>12s} {str(v3):>12s}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


if __name__ == "__main__":
    run_comparison()
