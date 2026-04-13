import numpy as np
import matplotlib.pyplot as plt
from core.environment import MapEnvironment
from core.sensor import LidarA1

def generate_circle_path(center=(10, 10), radius=5, num_steps=40):
    theta = linspace(0, 2*np.pi, num_steps)
    path = []
    for t in theta:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        angle = t + np.pi # 始终指向圆心
        path.append([x, y, angle])
    return path

def main():
    # 初始化模块
    env = MapEnvironment()
    lidar = LidarA1()
    path = generate_circle_path()
    
    plt.figure(figsize=(12, 6))
    all_points = []

    for pose in path:
        # 执行扫描
        ranges, angles = lidar.scan(pose, env)
        
        # 坐标转换：从极坐标到全局直角坐标
        valid = ranges < lidar.range_max
        gx = pose[0] + ranges[valid] * np.cos(angles[valid] + pose[2])
        gy = pose[1] + ranges[valid] * np.sin(angles[valid] + pose[2])
        all_points.extend(zip(gx, gy))

        # 实时动态绘图
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(env.grid_map, cmap='gray', extent=[0, 20, 0, 20], origin='lower')
        plt.plot(pose[0], pose[1], 'go') # 机器人位置
        
        plt.subplot(1, 2, 2)
        plt.scatter(gx, gy, s=5, c='r')
        plt.axis([8, 12, 8, 12]) # 聚焦五角星
        plt.title("Real-time Feature Capture")
        plt.pause(0.05)

    plt.show()

if __name__ == "__main__":
    main()