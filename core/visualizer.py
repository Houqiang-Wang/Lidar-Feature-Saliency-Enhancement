import numpy as np
import matplotlib.pyplot as plt

class LidarVisualizer:
    def __init__(self, env):
        self.fig = plt.figure(figsize=(18, 6))
        self.env = env
        # 预存累计点
        self.acc_x = []
        self.acc_y = []

    def update(self, pose, current_scan, lidar_range_max):
        """
        pose: [x, y, theta]
        current_scan: (ranges, angles)
        """
        ranges, angles = current_scan
        valid = ranges < lidar_range_max
        
        # 1. 计算当前帧全局点
        cx = pose[0] + ranges[valid] * np.cos(angles[valid] + pose[2])
        cy = pose[1] + ranges[valid] * np.sin(angles[valid] + pose[2])
        
        # 2. 存入增量池
        self.acc_x.extend(cx)
        self.acc_y.extend(cy)

        # 3. 绘图逻辑
        plt.clf()
        
        # 子图 1: 轨迹
        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(self.env.grid_map, cmap='gray', extent=[0, 20, 0, 20], origin='lower')
        ax1.plot(pose[0], pose[1], 'go', markersize=8)
        ax1.set_title("Robot Pose")

        # 子图 2: 当前扫描
        ax2 = self.fig.add_subplot(1, 3, 2)
        ax2.scatter(cx, cy, s=10, c='r', marker='.')
        ax2.set_xlim(8, 12); ax2.set_ylim(8, 12)
        ax2.set_title("Current Scan")

        # 子图 3: 增量拼接
        ax3 = self.fig.add_subplot(1, 3, 3)
        ax3.scatter(self.acc_x, self.acc_y, s=2, c='b', alpha=0.5)
        ax3.set_xlim(8, 12); ax3.set_ylim(8, 12)
        ax3.set_title("Incremental Map")
        
        plt.pause(0.01)

    def show_final(self):
        plt.show()