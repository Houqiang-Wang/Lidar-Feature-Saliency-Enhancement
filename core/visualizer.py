import numpy as np
import matplotlib.pyplot as plt

class LidarVisualizer:
    def __init__(self, env):
        self.env = env
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.acc_x = []
        self.acc_y = []

    def update(self, pose, current_scan, lidar_range_max, sim_time, v, w):
        """
        新增参数:
        sim_time: 当前仿真累计时间 (s)
        v: 机器人当前线速度 (m/s)
        w: 机器人当前角速度 (rad/s)
        """
        ranges, angles = map(np.asarray, current_scan)
        valid = ranges < lidar_range_max

        # 1. 计算当前帧全局点
        cx = pose[0] + ranges[valid] * np.cos(angles[valid] + pose[2])
        cy = pose[1] + ranges[valid] * np.sin(angles[valid] + pose[2])

        # 2. 存入增量池
        self.acc_x.extend(cx.tolist())
        self.acc_y.extend(cy.tolist())

        # 3. 绘图逻辑
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # 子图 1: 轨迹 & 状态面板 (Dashboard)
        self.ax1.imshow(
            self.env.grid_map,
            cmap='gray',
            extent=[0, self.env.width, 0, self.env.height],
            origin='lower'
        )
        self.ax1.plot(pose[0], pose[1], 'go', markersize=8)
        
        # --- 绘制实时状态数据文本 ---
        # 根据 RPLIDAR A1 规格参数显示
        info_str = (
            f"Sim Time: {sim_time:.2f} s\n"
            f"Linear Vel: {v:.2f} m/s\n"
            f"Angular Vel: {w:.2f} rad/s\n"
            f"Lidar Freq: 5.5 Hz"
        )
        # 在左上角添加半透明文本框
        self.ax1.text(0.5, self.env.height - 0.5, info_str, color='white', fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6))

        self.ax1.set_title("Robot Pose & Status")
        self.ax1.set_xlim(0, self.env.width)
        self.ax1.set_ylim(0, self.env.height)

        # 子图 2: 当前扫描
        self.ax2.scatter(cx, cy, s=10, c='r', marker='.')
        self.ax2.set_xlim(8, 12)
        self.ax2.set_ylim(8, 12)
        self.ax2.set_title("Current Scan (5.5Hz)")

        # 子图 3: 增量拼接
        self.ax3.scatter(self.acc_x, self.acc_y, s=2, c='b', alpha=0.5)
        self.ax3.set_xlim(8, 12)
        self.ax3.set_ylim(8, 12)
        self.ax3.set_title("Incremental Map")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001) # 缩短暂停时间提高响应速度

    def show_final(self):
        plt.show()