import numpy as np
from core.engine import SimulationEngine
from core.environment import MapEnvironment
from core.sensor import LidarA1
from core.visualizer import LidarVisualizer
from core.utils import apply_diff_drive_kinematics

def main():
    # 1. 平台参数设置
    dt = 0.02                # 仿真频率 50Hz (物理一致性)
    sim_duration = 20.0      # 总仿真时长 20秒
    real_time_factor = 1.0   # 设置为 1.0 即与现实秒钟同步
    
    engine = SimulationEngine(dt=dt, rtf=real_time_factor)
    env = MapEnvironment()
    lidar = LidarA1()
    viz = LidarVisualizer(env)

    # 2. 机器人控制参数 (完全拟真控制)
    pose = np.array([15.0, 10.0, np.pi/2])
    v = 0.5   # 线速度 0.5 m/s
    R = 5.0   # 期望的旋转半径 5m
    w = v / R # 计算得出角速度 0.1 rad/s

    # 3. 计算绕行一圈所需的时间
    # 周长 = 2 * pi * R, 时间 = 周长 / v
    duration = (2 * np.pi * R) / v  # 约 62.8 秒

    print(f"开始绕行任务：半径 {R}m, 预计耗时 {duration:.2f}s")




    # 3. 仿真主循环
    while engine.get_time() < duration:
        # 物理更新
        pose = apply_diff_drive_kinematics(pose, v, w, dt)
        
        # 雷达采样 (5.5Hz)
        if lidar.ready(engine.get_time()):
            scan_data = lidar.scan(pose, env)
            viz.update(pose, scan_data, lidar.range_max, engine.get_time(), v, w)
            
        engine.step()

    viz.show_final()

if __name__ == "__main__":
    main()