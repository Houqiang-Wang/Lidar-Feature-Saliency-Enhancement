import numpy as np

def apply_diff_drive_kinematics(pose, v, w, dt):
    """
    更新机器人位姿，并进行角度归一化
    pose: [x, y, theta]
    v: 线速度 (m/s)
    w: 角速度 (rad/s)
    dt: 仿真步长 (s)
    """
    new_pose = np.array(pose, dtype=np.float64)
    
    # 1. 位移计算
    new_pose[0] += v * np.cos(pose[2]) * dt
    new_pose[1] += v * np.sin(pose[2]) * dt
    
    # 2. 角度更新
    new_pose[2] += w * dt
    
    # 3. 角度归一化 (行业标准操作：限制在 -pi 到 pi)
    new_pose[2] = (new_pose[2] + np.pi) % (2 * np.pi) - np.pi
    
    return new_pose