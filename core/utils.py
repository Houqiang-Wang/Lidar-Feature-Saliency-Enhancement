import numpy as np

def generate_circle_path(center=(10, 10), radius=5, num_steps=60):
    """
    生成绕中心点旋转的圆形路径，并确保机器人朝向始终指向中心。
    """
    theta = np.linspace(0, 2 * np.pi, num_steps)
    path = []
    for t in theta:
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        angle = t + np.pi  # 指向圆心
        path.append([x, y, angle])
    return path