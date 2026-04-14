import numpy as np

def generate_circle_path(center=(10, 10), radius=5, num_steps=600):
    """
    圆形路径：绕中心旋转，面向圆心
    """
    theta = np.linspace(0, 2 * np.pi, num_steps)
    path = []
    for t in theta:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        angle = t + np.pi 
        path.append([x, y, angle])
    return path

def generate_linear_path(start_pos=(2, 2), end_pos=(2, 18), num_steps=600):
    """
    直线路径：从房间一端走到另一端
    默认在左侧 (x=2) 从 y=2 走到 y=18，面向右侧（朝向五角星）
    """
    y_coords = np.linspace(start_pos[1], end_pos[1], num_steps)
    x_coords = np.linspace(start_pos[0], end_pos[0], num_steps)
    path = []
    for i in range(num_steps):
        x = x_coords[i]
        y = y_coords[i]
        # 始终面向右侧 (0弧度) 观察中心障碍物
        angle = 0.0 
        path.append([x, y, angle])
    return path

def generate_spiral_path(center=(10, 10), start_radius=8, end_radius=2, num_steps=100):
    """
    螺旋渐近路径：边绕圈边靠近中心
    """
    theta = np.linspace(0, 1.6 * np.pi, num_steps) # 3/4圈
    radii = np.linspace(start_radius, end_radius, num_steps)
    path = []
    for i in range(num_steps):
        t = theta[i]
        r = radii[i]
        x = center[0] + r * np.cos(t)
        y = center[1] + r * np.sin(t)
        # 航向角指向中心
        angle = t + np.pi
        path.append([x, y, angle])
    return path