import matplotlib.pyplot as plt
from core.environment import MapEnvironment
from core.sensor import LidarA1
from core.visualizer import LidarVisualizer
from core.utils import generate_circle_path  

def main():
    # 初始化环境
    env = MapEnvironment()
    lidar = LidarA1()
    viz = LidarVisualizer(env)
    
    # 现在这里可以正常调用了
    path = generate_circle_path(center=(10, 10), radius=5)
    
    for pose in path:
        scan_data = lidar.scan(pose, env)
        viz.update(pose, scan_data, lidar.range_max)

    viz.show_final()

if __name__ == "__main__":
    main()