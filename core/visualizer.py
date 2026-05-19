"""
Saliency-LOAM Visualiser — 6 面板实时仪表盘
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Optional
from core.sensor import LaserScan
from core.feature_extraction import LineSegment

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def _extract_contour(pts, xlim, ylim, grid_res=0.02, min_hits=1, dilate_r=0.03):
    """点云→占据栅格→最大连通域→轮廓（参数相同，差异来自数据）"""
    if len(pts) == 0:
        return None
    x0, x1 = xlim; y0, y1 = ylim
    w = int((x1 - x0) / grid_res) + 1
    h = int((y1 - y0) / grid_res) + 1
    col = ((pts[:,0] - x0) / grid_res).astype(np.int32)
    row = ((pts[:,1] - y0) / grid_res).astype(np.int32)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    col, row = col[ok], row[ok]
    if len(col) == 0:
        return None
    dens = np.zeros((h, w), dtype=np.int32)
    np.add.at(dens, (row, col), 1)
    bin_ = (dens >= min_hits).astype(np.uint8)
    if not np.any(bin_):
        return None
    if dilate_r > 0:
        ks = max(1, int(dilate_r / grid_res))
        kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ks+1, 2*ks+1))
        bin_ = cv2.dilate(bin_, kr, iterations=1)
    nl, lb, st, _ = cv2.connectedComponentsWithStats(bin_, 8)
    if nl <= 1:
        return None
    big = np.argmax(st[1:, cv2.CC_STAT_AREA]) + 1
    msk = (lb == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).squeeze(1).astype(np.float32)
    if c.ndim == 1 or len(c) < 3:
        return None
    c[:,0] = c[:,0] * grid_res + x0
    c[:,1] = c[:,1] * grid_res + y0
    return c


class SaliencyVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        self.ax_traj   = axes[0, 0]
        self.ax_feat   = axes[0, 1]
        self.ax_raw_c  = axes[0, 2]
        self.ax_sal    = axes[1, 0]
        self.ax_map    = axes[1, 1]
        self.ax_enh_c  = axes[1, 2]

        d = make_axes_locatable(self.ax_sal)
        ca = d.append_axes("right", size="3%", pad=0.08)
        self._cbar = self.fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0,1), cmap="jet"), cax=ca)
        self._cbar.set_label("显著性")

        self.gt_x: List[float] = [];  self.gt_y: List[float] = []
        self.est_x: List[float] = []; self.est_y: List[float] = []
        self._raw: List[np.ndarray] = []
        self._enh: List[np.ndarray] = []
        self._last_t = -10.0

    # ==================================================================
    def update(self, gt_pose, est_pose, scan, points_local, saliency,
               segments=None, sim_time=0.0, v=0.0, w=0.0):
        self.gt_x.append(float(gt_pose[0])); self.gt_y.append(float(gt_pose[1]))
        self.est_x.append(float(est_pose[0])); self.est_y.append(float(est_pose[1]))

        cg,sg=np.cos(gt_pose[2]),np.sin(gt_pose[2]); Rg=np.array([[cg,-sg],[sg,cg]])
        self._raw.append((Rg @ points_local.T).T + gt_pose[:2])

        th = np.percentile(saliency, 60)
        hi = saliency > th
        if np.any(hi):
            ce,se=np.cos(est_pose[2]),np.sin(est_pose[2]); Re=np.array([[ce,-se],[se,ce]])
            self._enh.append((Re @ points_local[hi].T).T + est_pose[:2])

        # --- 1 ---
        self.ax_traj.clear()
        self.ax_traj.imshow(self.env.grid_map, cmap="gray",
                            extent=[0,self.env.width,0,self.env.height], origin="lower")
        self.ax_traj.plot(self.gt_x,self.gt_y,"g-",lw=1.5,label="GT")
        self.ax_traj.plot(self.est_x,self.est_y,"r--",lw=1.5,label="Est")
        self.ax_traj.plot(gt_pose[0],gt_pose[1],"go",ms=6)
        self.ax_traj.plot(est_pose[0],est_pose[1],"ro",ms=6)
        self.ax_traj.text(0.5,self.env.height-0.5,
                          f"t={sim_time:.1f}s  v={v:.2f}  w={w:.2f}",
                          color="white",fontsize=9,va="top",
                          bbox=dict(facecolor="black",alpha=0.6))
        self.ax_traj.set_title("轨迹与状态")
        self.ax_traj.set_xlim(0,self.env.width); self.ax_traj.set_ylim(0,self.env.height)
        self.ax_traj.legend(loc="lower right")

        # --- 2 ---
        self.ax_feat.clear(); self._draw_scan(self.ax_feat,gt_pose,points_local,segments)
        self.ax_feat.set_title("扫描与线特征"); self.ax_feat.set_aspect("equal")

        # --- 3 ---
        self.ax_sal.clear(); self._draw_sal(self.ax_sal,gt_pose,points_local,saliency)
        self.ax_sal.set_title("显著性热力图"); self.ax_sal.set_aspect("equal")

        # --- 4 ---
        self.ax_map.clear(); self._draw_map(self.ax_map,gt_pose,scan)
        self.ax_map.set_title("增量地图"); self.ax_map.set_aspect("equal")

        # --- 5 & 6 ---
        if sim_time - self._last_t > 2.0:
            self._last_t = sim_time
            self.ax_raw_c.clear(); self.ax_enh_c.clear()
            self._draw_contours()
        for ax in [self.ax_raw_c, self.ax_enh_c]:
            ax.set_xlim(7.5,12.5); ax.set_ylim(7.5,12.5); ax.set_aspect("equal")
        self.ax_raw_c.set_title("原始点云 — 障碍物轮廓")
        self.ax_enh_c.set_title("显著性增强 — 障碍物轮廓")

        self.fig.canvas.draw(); plt.pause(0.001)

    # ==================================================================
    def _draw_contours(self):
        xl,yl = (7.5,12.5),(7.5,12.5)
        star = np.vstack([_star_gt(), _star_gt()[0]])

        rp = np.vstack(self._raw) if self._raw else np.zeros((0,2))
        ep = np.vstack(self._enh) if self._enh else np.zeros((0,2))

        # 完全相同参数，差异仅来自输入数据质量
        rc = _extract_contour(rp, xl, yl, 0.02, 1, 0.03)
        ec = _extract_contour(ep, xl, yl, 0.02, 1, 0.03)

        def _d(ax, ct, title, n, tot, c):
            ax.clear()
            ax.plot(star[:,0],star[:,1],"g--",lw=2.5,label="真实轮廓",zorder=5)
            if ct is not None and len(ct)>2:
                ax.plot(ct[:,0],ct[:,1],c,lw=2.8,label="识别轮廓",zorder=4)
            ax.text(0.97,0.03, f"累积 {n} 帧 | {tot} 点 | 占据≥1 膨胀0.03m",
                    transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.9))
            ax.set_title(title,fontsize=13,fontweight="bold")
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
            ax.legend(loc="upper left",fontsize=10); ax.grid(True,ls=":",alpha=0.4)

        _d(self.ax_raw_c, rc, "原始点云 — 障碍物轮廓", len(self._raw), len(rp), "#CC4444")
        _d(self.ax_enh_c, ec, "显著性增强 — 障碍物轮廓", len(self._enh), len(ep), "#22AA22")

    # ==================================================================
    @staticmethod
    def _to_world(p, pts):
        c,s=np.cos(p[2]),np.sin(p[2]); R=np.array([[c,-s],[s,c]])
        return (R @ pts.T).T + p[:2]

    def _draw_scan(self,ax,p,pts,segs):
        if len(pts)==0: return
        w=self._to_world(p,pts); ax.scatter(w[:,0],w[:,1],s=5,c="grey",alpha=0.5)
        if segs:
            for s in segs:
                e=self._to_world(p,s.endpoints); ax.plot(e[:,0],e[:,1],"b-",lw=2)
        ax.set_xlim(p[0]-10,p[0]+10); ax.set_ylim(p[1]-10,p[1]+10)

    def _draw_sal(self,ax,p,pts,sal):
        if len(pts)==0: return
        w=self._to_world(p,pts)
        ax.scatter(w[:,0],w[:,1],c=sal,cmap="jet",s=15,vmin=0,vmax=1)
        ax.set_xlim(p[0]-10,p[0]+10); ax.set_ylim(p[1]-10,p[1]+10)

    def _draw_map(self,ax,p,scan):
        v=np.isfinite(scan.ranges)&(scan.ranges>scan.range_min)
        a=scan.angles[v]+p[2]; r=scan.ranges[v]
        ax.scatter(p[0]+r*np.cos(a),p[1]+r*np.sin(a),s=2,c="red",alpha=0.4)
        ax.plot(p[0],p[1],"go",ms=6)
        ax.set_xlim(0,self.env.width); ax.set_ylim(0,self.env.height)

    def show_final(self): plt.show()


def _star_gt():
    cx,cy=10.0,10.0; R,r=1.2,0.4; n=10
    a=np.linspace(0,2*np.pi,n,endpoint=False)-np.pi/2
    rad=np.array([R if i%2==0 else r for i in range(n)])
    return np.column_stack([cx+rad*np.cos(a), cy+rad*np.sin(a)])
