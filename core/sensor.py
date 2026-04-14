import numpy as np


class LidarA1:
    def __init__(self):
        self.range_min = 0.15                   # 最小测距 15cm
        self.range_max = 12.0                   # 最大测距 12m
        self.angle_res = np.deg2rad(0.5)        # 角分辨率 0.5度，360度共720条射线
        
        # 新增频率控制参数
        self.scan_period = 1.0 / 5.5            # 5.5Hz，约 0.1818s 更新一次数据
        self.last_update_time = -self.scan_period

    def ready(self, current_sim_time):
        """判断是否到达 5.5Hz 的采样时刻"""
        if current_sim_time - self.last_update_time >= self.scan_period:
            self.last_update_time = current_sim_time
            return True
        return False
    


    def _get_noise_std(self, distance):                         # 测距精度
        if isinstance(distance, np.ndarray):
            noise_scales = np.where(distance <= 3.0, 0.01, 
                           np.where(distance <= 5.0, 0.02, 0.025))
            return distance * noise_scales
        # 兼容单个数值
        std = 0.01 if distance <= 3.0 else (0.02 if distance <= 5.0 else 0.025)
        return distance * std

    def scan(self, robot_pose, env):
        rx, ry, rtheta = robot_pose
        angles = np.arange(-np.pi, np.pi, self.angle_res)       # 扫描范围 -180° 到 180°
        num_beams = len(angles)
        
        final_ranges = np.full(num_beams, self.range_max, dtype=np.float32)
        finished = np.zeros(num_beams, dtype=bool)

        # 1. 预计算所有射线的方向向量
        beam_angles = angles + rtheta
        cos_a = np.cos(beam_angles)
        sin_a = np.sin(beam_angles)

        # 2. 射线步进
        # 步长建议设为固定值，比如 0.05m (5cm)，兼顾速度与精度
        step = 0.05 
        dist_steps = np.arange(self.range_min, self.range_max, step)

        for d in dist_steps:
            active = ~finished
            if not np.any(active): break

            # 计算所有活跃射线的物理坐标 (x, y)
            tx_m = rx + d * cos_a[active]
            ty_m = ry + d * sin_a[active]

            # 转换为像素坐标
            px = (tx_m * env.res).astype(np.int32)
            py = (ty_m * env.res).astype(np.int32)

            # 越界检查
            in_bounds = (px >= 0) & (px < env.size_px) & (py >= 0) & (py < env.size_px)
            
            if not np.any(in_bounds): continue

            # 只检查在地图内的射线
            # 注意：NumPy 矩阵索引是 [行, 列]，即 [y, x]
            # 如果之前的 MapEnvironment 绘图是正常的，这里必须对齐
            hit_mask = np.zeros(len(tx_m), dtype=bool)
            hit_mask[in_bounds] = env.grid_map[py[in_bounds], px[in_bounds]] > 0
            
            if np.any(hit_mask):
                # 找到这些击中点在原 angles 数组中的索引
                active_indices = np.where(active)[0]
                hit_indices = active_indices[hit_mask]
                
                # 记录距离并加上噪声
                noise = np.random.normal(0, self._get_noise_std(d), size=len(hit_indices))
                final_ranges[hit_indices] = d + noise
                finished[hit_indices] = True

        return final_ranges, angles