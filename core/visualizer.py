"""
Saliency-LOAM Visualiser
========================
Multi-panel Matplotlib dashboard showing:

1. Global trajectory & status
2. Raw scan + extracted line segments
3. Saliency heat-map overlay
4. Incremental occupancy map (all past scans)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Optional

# 中文字体支持（Windows 常用黑体/雅黑，无则回退）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from core.sensor import LaserScan
from core.feature_extraction import LineSegment


class SaliencyVisualizer:
    """Real-time 4-panel visualisation for the Saliency-LOAM pipeline."""

    def __init__(self, env):
        self.env = env
        # Use a 2x3 grid: 4 main panels + 1 narrow column for the colorbar
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        self.ax_traj = axes[0, 0]
        self.ax_feature = axes[0, 1]
        self.ax_saliency = axes[1, 0]
        self.ax_map = axes[1, 1]

        # Colorbar 紧贴显著性子图右侧，不破坏 2x2 整体对齐
        divider = make_axes_locatable(self.ax_saliency)
        self._cax = divider.append_axes("right", size="3%", pad=0.08)

        # Trajectory accumulators
        self.gt_x: List[float] = []
        self.gt_y: List[float] = []
        self.est_x: List[float] = []
        self.est_y: List[float] = []

        # Pre-create a fixed colorbar for saliency so it never accumulates
        self._saliency_sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=0, vmax=1), cmap="jet"
        )
        self._cbar = self.fig.colorbar(
            self._saliency_sm, cax=self._cax
        )
        self._cbar.set_label("显著性")

    # ------------------------------------------------------------------ #
    #  Public update
    # ------------------------------------------------------------------ #

    def update(
        self,
        gt_pose: np.ndarray,
        est_pose: np.ndarray,
        scan: LaserScan,
        points_local: np.ndarray,
        saliency: np.ndarray,
        segments: Optional[List[LineSegment]] = None,
        sim_time: float = 0.0,
        v: float = 0.0,
        w: float = 0.0,
    ) -> None:
        """Refresh all four sub-plots."""
        # Record trajectories
        self.gt_x.append(float(gt_pose[0]))
        self.gt_y.append(float(gt_pose[1]))
        self.est_x.append(float(est_pose[0]))
        self.est_y.append(float(est_pose[1]))

        # --- 1. Trajectory & status ------------------------------------
        self.ax_traj.clear()
        self.ax_traj.imshow(
            self.env.grid_map,
            cmap="gray",
            extent=[0, self.env.width, 0, self.env.height],
            origin="lower",
        )
        self.ax_traj.plot(self.gt_x, self.gt_y, "g-", linewidth=1.5, label="GT")
        self.ax_traj.plot(self.est_x, self.est_y, "r--", linewidth=1.5, label="Est")
        self.ax_traj.plot(gt_pose[0], gt_pose[1], "go", markersize=6)
        self.ax_traj.plot(est_pose[0], est_pose[1], "ro", markersize=6)

        info = (
            f"Sim Time: {sim_time:.2f} s\n"
            f"Linear Vel: {v:.2f} m/s\n"
            f"Angular Vel: {w:.2f} rad/s\n"
            f"Lidar Freq: 5.5 Hz\n"
            f"Saliency mean: {np.mean(saliency):.3f}"
        )
        self.ax_traj.text(
            0.5,
            self.env.height - 0.5,
            info,
            color="white",
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.6),
        )
        self.ax_traj.set_title("轨迹与状态")
        self.ax_traj.set_xlim(0, self.env.width)
        self.ax_traj.set_ylim(0, self.env.height)
        self.ax_traj.legend(loc="lower right")

        # --- 2. Current scan + line features ---------------------------
        self.ax_feature.clear()
        self._draw_scan_and_features(
            self.ax_feature, gt_pose, points_local, segments
        )
        self.ax_feature.set_title("扫描与线特征")
        self.ax_feature.set_aspect("equal")

        # --- 3. Saliency heat-map --------------------------------------
        self.ax_saliency.clear()
        self._draw_saliency(self.ax_saliency, gt_pose, points_local, saliency)
        self.ax_saliency.set_title("显著性热力图")
        self.ax_saliency.set_aspect("equal")

        # --- 4. Incremental map (all past scans) -----------------------
        self.ax_map.clear()
        self._draw_incremental_map(self.ax_map, gt_pose, scan)
        self.ax_map.set_title("增量地图")
        self.ax_map.set_aspect("equal")

        # tight_layout skipped: GridSpec already handles alignment
        self.fig.canvas.draw()
        plt.pause(0.001)

    # ------------------------------------------------------------------ #
    #  Drawing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _transform_local_to_world(pose: np.ndarray, pts_local: np.ndarray) -> np.ndarray:
        cos_t, sin_t = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return (R @ pts_local.T).T + pose[:2]

    def _draw_scan_and_features(
        self,
        ax,
        pose: np.ndarray,
        points_local: np.ndarray,
        segments: Optional[List[LineSegment]],
    ) -> None:
        """Plot raw scan points and overlaid line segments."""
        if len(points_local) == 0:
            return
        world_pts = self._transform_local_to_world(pose, points_local)
        ax.scatter(world_pts[:, 0], world_pts[:, 1], s=5, c="grey", alpha=0.5)

        if segments:
            for seg in segments:
                ends_w = self._transform_local_to_world(pose, seg.endpoints)
                ax.plot(ends_w[:, 0], ends_w[:, 1], "b-", linewidth=2)
                ax.plot(ends_w[:, 0], ends_w[:, 1], "bo", markersize=4)

        # Zoom to robot vicinity (must cover room-scale obstacles)
        ax.set_xlim(pose[0] - 10, pose[0] + 10)
        ax.set_ylim(pose[1] - 10, pose[1] + 10)

    def _draw_saliency(
        self,
        ax,
        pose: np.ndarray,
        points_local: np.ndarray,
        saliency: np.ndarray,
    ) -> None:
        """Scatter plot coloured by saliency score."""
        if len(points_local) == 0:
            return
        world_pts = self._transform_local_to_world(pose, points_local)
        ax.scatter(
            world_pts[:, 0],
            world_pts[:, 1],
            c=saliency,
            cmap="jet",
            s=15,
            vmin=0,
            vmax=1,
        )
        ax.set_xlim(pose[0] - 10, pose[0] + 10)
        ax.set_ylim(pose[1] - 10, pose[1] + 10)

    def _draw_incremental_map(self, ax, pose: np.ndarray, scan: LaserScan) -> None:
        """Draw the current scan in world frame as a red point cloud."""
        valid = np.isfinite(scan.ranges) & (scan.ranges > scan.range_min)
        angles = scan.angles[valid] + pose[2]
        ranges = scan.ranges[valid]
        wx = pose[0] + ranges * np.cos(angles)
        wy = pose[1] + ranges * np.sin(angles)

        ax.scatter(wx, wy, s=2, c="red", alpha=0.4)
        ax.plot(pose[0], pose[1], "go", markersize=6)
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)

    def show_final(self) -> None:
        """Block until the user closes the figure window."""
        plt.show()
