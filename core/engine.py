# core/engine.py
import time

class SimulationEngine:
    def __init__(self, dt=0.02, rtf=1.0):
        self.dt = dt                 # 仿真步长 (s)，0.02s 对应 50Hz 物理频率
        self.sim_time = 0.0          # 累计仿真时间 (秒)
        self.rtf = rtf  # 1.0 为等速，0 为不限速(跑多快跑多快)
        self.start_wall_time = time.time()

    def step(self):
        """推进一个物理步长"""
        self.sim_time += self.dt
        if self.rtf > 0:
            # 强制对齐现实时间
            expected = self.sim_time / self.rtf
            elapsed = time.time() - self.start_wall_time
            if expected > elapsed:
                time.sleep(expected - elapsed)

    def get_time(self):
        return self.sim_time