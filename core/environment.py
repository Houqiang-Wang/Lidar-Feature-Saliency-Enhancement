import numpy as np
import cv2

class MapEnvironment:
    def __init__(self, width=20, height=20, resolution=50):
        self.width = width
        self.height = height
        self.res = resolution
        self.size_px = width * resolution
        # 初始化空白栅格地图 (0: 空闲, 255: 占据)
        self.grid_map = np.zeros((self.size_px, self.size_px), dtype=np.uint8)
        self._build_room()
        self._add_star_obstacle()

    def _build_room(self):
        # 建立 19x19 的外墙 (留出1米边距)
        margin = self.res
        thickness = 2
        cv2.rectangle(self.grid_map, (margin, margin), 
                      (self.size_px - margin, self.size_px - margin), 255, thickness)

    def _add_star_obstacle(self):
        # 建立直径 1.2m 的五角星 (R=0.6, r=0.2)
        cx, cy = self.size_px // 2, self.size_px // 2
        R, r = 0.6 * self.res, 0.2 * self.res
        pts = []
        for i in range(11):
            curr_r = R if i % 2 == 0 else r
            angle = i * 2 * np.pi / 10 - np.pi/2
            x = cx + curr_r * np.cos(angle)
            y = cy + curr_r * np.sin(angle)
            pts.append([x, y])
        
        pts = np.array(pts, np.int32)
        cv2.fillPoly(self.grid_map, [pts], 255) # 实心填充

    def is_occupied(self, x_m, y_m):
        # 将物理坐标转换为像素索引检查是否碰撞
        px = int(x_m * self.res)
        py = int(y_m * self.res)
        if 0 <= px < self.size_px and 0 <= py < self.size_px:
            return self.grid_map[py, px] > 0
        return True