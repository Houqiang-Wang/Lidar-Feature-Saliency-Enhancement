"""
Saliency-LOAM 2D Simulation Demo
================================
参考 Saliency-LOAM (IEEE TIM 2026) 的 2D 迁移思路，构建完整的前端
仿真管线：

    原始 Scan → 显著性估计 → 线特征提取 → 帧间/帧到地图配准
    → 动态抑制 → 关键帧管理 → 可视化

运行：
    python main.py
"""

from __future__ import annotations

import numpy as np

from core.engine import SimulationEngine
from core.environment import MapEnvironment
from core.sensor import LidarA1, LidarConfig
from core.saliency import SaliencyEstimator
from core.feature_extraction import LineFeatureExtractor
from core.mapping import MapManager
from core.registration import ScanMatcher
from core.dynamic_filter import DynamicFilter
from core.visualizer import SaliencyVisualizer
from core.utils import apply_diff_drive_kinematics


def main() -> None:
    # =====================================================================
    # 1. 仿真参数
    # =====================================================================
    dt = 0.02                     # 物理步长 50 Hz
    real_time_factor = 1.0        # 1.0 = 实时
    sim_duration = 60.0           # 总仿真时长 (约一圈 62.8s，取 60s)

    engine = SimulationEngine(dt=dt, rtf=real_time_factor)
    env = MapEnvironment()

    # 高噪声雷达配置，凸显轮廓识别差异
    high_noise_cfg = LidarConfig(
        noise_ratio_near=0.04,    # 近距离噪声 4%
        noise_ratio_far=0.06,     # 远距离噪声 6%
        angle_noise_std=np.deg2rad(0.5),  # 角度抖动 0.5°
        dropout_rate=0.02,        # 丢包率 2%
    )
    lidar = LidarA1(config=high_noise_cfg)

    # =====================================================================
    # 2. Saliency-LOAM 模块初始化
    # =====================================================================
    # 提高曲率权重(alpha)，降低语义/强度权重，使几何棱角更突出
    saliency_est = SaliencyEstimator(k_neighbors=4, alpha=0.92, beta=0.03, gamma=0.05)
    line_extractor = LineFeatureExtractor(split_thresh=0.05, min_points=5)
    mapper = MapManager(keyframe_dist=0.5, keyframe_angle=np.deg2rad(20.0))
    matcher = ScanMatcher(max_corr_dist=1.0)
    dyn_filter = DynamicFilter(th_pose_trans=0.2, th_pose_rot=np.deg2rad(15.0))
    viz = SaliencyVisualizer(env)

    # =====================================================================
    # 3. 机器人控制参数 (绕圆)
    # =====================================================================
    gt_pose = np.array([15.0, 10.0, np.pi / 2], dtype=np.float32)
    v = 0.5                       # 线速度 [m/s]
    R = 5.0                       # 旋转半径 [m]
    w = v / R                     # 角速度 [rad/s]

    # 用于里程计漂移演示：给估计位姿一个微小初始误差
    est_pose = gt_pose.copy()
    est_pose[0] += 0.1
    est_pose[1] += 0.05

    print(f"开始 Saliency-LOAM 仿真：半径 {R} m, 预计耗时 ~{sim_duration:.1f} s")

    # 前一帧点云（世界坐标），用于帧间配准
    prev_world_points: np.ndarray | None = None

    # =====================================================================
    # 4. 主循环
    # =====================================================================
    while engine.get_time() < sim_duration:
        # ---- 4.1 物理更新 (真值) ------------------------------------
        gt_pose = apply_diff_drive_kinematics(gt_pose, v, w, dt)

        # ---- 4.2 里程计递推 (估计位姿也需要同步推进) ----------------
        est_pose = apply_diff_drive_kinematics(est_pose, v, w, dt)

        if lidar.ready(engine.get_time()):
            scan = lidar.scan(gt_pose, env)

            # ---- 4.4 显著性估计 --------------------------------------
            points_local, valid_mask, saliency = saliency_est.compute(scan)

            # ---- 4.4 线特征提取 --------------------------------------
            segments = line_extractor.extract(points_local, saliency)

            # ---- 4.5 动态滤波 + 位姿估计 -----------------------------
            local_map = mapper.get_local_map()
            init_guess = est_pose.copy()

            est_pose, saliency = dyn_filter.filter(
                points=points_local,
                saliency=saliency,
                local_map=local_map,
                prev_points=prev_world_points,
                init_pose=init_guess,
                matcher=matcher,
            )

            # ---- 4.6 点云转到世界坐标 (使用优化后的位姿) --------------
            cos_t, sin_t = np.cos(est_pose[2]), np.sin(est_pose[2])
            R_est = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            world_points = (R_est @ points_local.T).T + est_pose[:2]

            # ---- 4.7 关键帧管理 --------------------------------------
            if mapper.should_insert(est_pose):
                mapper.add_keyframe(
                    pose=est_pose,
                    points_local=points_local,
                    saliency=saliency,
                    timestamp=engine.get_time(),
                )

            # ---- 4.9 可视化 ------------------------------------------
            viz.update(
                gt_pose=gt_pose,
                est_pose=est_pose,
                scan=scan,
                points_local=points_local,
                saliency=saliency,
                segments=segments,
                sim_time=engine.get_time(),
                v=v,
                w=w,
            )

            # 保存当前帧世界点云供下一帧帧间配准使用
            prev_world_points = world_points.copy()

        # ---- 4.10 推进仿真时钟 ---------------------------------------
        engine.step()

    # =====================================================================
    # 5. 结束
    # =====================================================================
    print("仿真结束。关闭图形窗口以退出。")
    viz.show_final()


if __name__ == "__main__":
    main()
