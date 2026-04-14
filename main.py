import matplotlib.pyplot as plt
from core.environment import MapEnvironment
from core.sensor import LidarA1
from core.visualizer import LidarVisualizer
from core.utils import generate_circle_path, generate_linear_path, generate_spiral_path

def main():
    # 初始化环境
    env = MapEnvironment()
    lidar = LidarA1()
    viz = LidarVisualizer(env)
    
    # 调用utils生成路径
    #path = generate_circle_path(center=(10, 10), radius=5)
    path = generate_linear_path(start_pos=(3, 2), end_pos=(3, 18))
    #path = generate_spiral_path()
    

    for i, pose in enumerate(path):
        scan_data = lidar.scan(pose, env)

        # 每 3 帧刷新一次 UI
        #if i % 3 == 0:
        viz.update(pose, scan_data, lidar.range_max)
    viz.show_final()

if __name__ == "__main__":
    main()