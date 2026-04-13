import numpy as np

class LidarA1:
    def __init__(self, range_max=12.0, noise=0.05):
        self.range_min = 0.15
        self.range_max = range_max
        self.angle_res = np.deg2rad(1.0) # 1度分辨率
        self.noise_std = noise          # 测距噪声

    def scan(self, robot_pose, env):
        """
        robot_pose: [x, y, theta]
        env: MapEnvironment 对象
        """
        angles = np.arange(-np.pi, np.pi, self.angle_res)
        ranges = []
        rx, ry, rtheta = robot_pose

        for angle in angles:
            beam_angle = angle + rtheta
            hit_dist = self.range_max
            
            # 射线步进探测 (模拟激光传播)
            step = 1.0 / env.res
            for d in np.arange(self.range_min, self.range_max, step):
                tx = rx + d * np.cos(beam_angle)
                ty = ry + d * np.sin(beam_angle)
                
                if env.is_occupied(tx, ty):
                    hit_dist = d + np.random.normal(0, self.noise_std)
                    break
            ranges.append(hit_dist)
            
        return np.array(ranges), angles