# Lidar-Feature-Saliency-Enhancement

**Lidar Feature Saliency Enhancement** 是一个针对低成本激光雷达（如 RPLIDAR A1）在高噪声环境下几何特征识别精细度提升的研究项目。通过算法层面增强点云特征的显著性，旨在改善多机器人 SLAM 地图融合（Map Merge）中的数据关联问题。

<p align="center">
  <img src="assets/雷达数据.png" alt="双机器人SLAM场景" width="50%">
</p>

## 背景与痛点

多机器人 SLAM 需要对每个机器人建立的子图进行拼接（Map Merge），涉及两个关键步骤：地图对齐和数据关联[^1]。当激光雷达扫描出的环境几何特征不够明显时，数据关联会因找不到相似的几何特征而失败，导致"乱拼图"[^2]。

实验中部署了两台 RPLIDAR A1 机器人在场景内设置了具有高辨识度的障碍物，但由于传感器精度限制，几何特征识别并不明显。因此想到从**算法层面**前置增强原始数据的可辨识度，而非修改后端融合算法。

## 技术路线

在**单传感器纯 2D 激光**框架下，通过预处理提升点云特征显著性：

1. **时序多帧超分辨** — 用机器人静止时的 3–5 帧低频扫描，融合生成 1 帧高频"虚拟扫描"，突破思岚 A1 物理分辨率限制。
2. **自适应显著性估计** — 参考 Saliency-LOAM[^3] 的手工融合公式，综合曲率、强度梯度和语义权重计算逐点显著性。
3. **精细化线特征提取** — Split-Merge 分割合并算法，亚厘米级分割阈值捕捉毫米级角点。

```
建立仿真环境
    ↓
设置雷达和被识别物体参数
    ↓
引入真实噪声（距离相关高斯噪声、角度抖动、丢包）
    ↓
显著性估计 → 线特征提取 → 时序融合
    ↓
与 Baseline 定量对比
```

## 项目结构

```
Lidar-Feature-Saliency-Enhancement/
├── main.py                    # 主仿真程序：机器人绕圈扫描五角星障碍物
├── main_original.py           # 原始版本（优化前）
├── evaluation.py              # 对比实验脚本
├── paper_comparison.png       # 生成的对比图（300 dpi）
├── CHANGELOG.md               # 版本变更日志
│
├── core/                      # 核心算法模块
│   ├── environment.py         # 基于 OpenCV 的高分辨率栅格地图
│   ├── sensor.py              # RPLIDAR A1 传感器仿真（物理噪声模型）
│   ├── saliency.py            # 逐点显著性估计（PCA 曲率）
│   ├── feature_extraction.py  # Split-Merge 线特征提取
│   ├── engine.py              # 仿真时钟引擎
│   ├── mapping.py             # 关键帧 SLAM 地图管理
│   ├── registration.py        # 帧间/帧到地图配准
│   ├── dynamic_filter.py      # 位姿异常值滤波
│   ├── visualizer.py          # 四窗口实时可视化
│   └── utils.py               # 工具函数（差速运动学等）
│
├── saliency/                  # 显著性算法变种
│   ├── saliency_calculator.py
│   ├── feature_extract_sal.py
│   ├── weighted_icp.py
│   └── utils.py
│
├── slam/                      # SLAM 变种实现
│   ├── feature_extract.py
│   └── icp.py
│
├── config/
│   ├── default.yaml           # 默认（Baseline）配置
│   └── saliency.yaml          # 显著性增强配置
│
└── docs/
    └── Saliency-LOAM.md        # 参考论文笔记
```

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包：`numpy`, `matplotlib`, `opencv-python`, `scipy`

### 安装依赖

```bash
pip install numpy matplotlib opencv-python scipy
```

### 运行仿真

```bash
python main.py
```

运行后显示四个实时可视化窗口：

- 全局轨迹与地图
- 特征线提取图
- 显著性热力图
- 增量拼接地图

### 运行对比实验

```bash
python evaluation.py
```

**输出说明：**

- `paper_comparison.png` — 对照图
- 控制台输出 — 定量结果表格

## 实验结果

### 高噪声环境下特征识别精细度对比

实验场景：在高噪声环境（近距离噪声 4%、远距离 5%、角度抖动 0.4°、丢包 1%）下，机器人正对五角星障碍物。


| 指标               | Baseline 单帧 | Ours 单帧 | Ours 5帧融合 | 提升倍数    |
| ------------------ | ------------- | --------- | ------------ | ----------- |
| 五角星区域内点数   | 23            | 16        | **84**       | ↑3.7×     |
| 五角星区域内线段数 | 21            | 15        | **104**      | **↑4.9×** |
| 轮廓匹配误差 (m)   | 0.206         | 0.203     | **0.193**    | ↓6%        |

![实验对比结果](paper_comparison.png)

**上排（定性对比）：**

- **(a)** Baseline 单帧 — 21 条线段，覆盖基本轮廓
- **(b)** Ours 单帧 — 15 条线段，角点检测更精确
- **(c)** Ours 5帧融合 — 104 条线段，几乎完全覆盖五角星所有角点和边缘

**下排（定量指标）：**

- **(d)** 点数对比、**(e)** 线段数对比、**(f)** 轮廓匹配误差对比

### 关键技术优化


| 优化维度   | 修改内容                       | 效果                       |
| ---------- | ------------------------------ | -------------------------- |
| 传感器模型 | 噪声比例 ↑50–67%，启用丢包   | 更逼真的高噪声仿真环境     |
| 显著性权重 | α: 0.7→0.88（曲率权重↑26%） | 强化几何特征，角点优先保留 |
| 特征提取   | 分割阈值：3cm→1.2cm           | 角点识别精细度提升 2.5 倍  |
| 过滤策略   | 固定阈值→动态百分位           | 自适应噪声水平，更鲁棒     |

## 显著性公式

逐点显著性评分参考 Saliency-LOAM[^3] 的手工融合公式：

```
S_i = α · Norm(c_i) + β · Norm(|∇I_i|) + γ · M_sem(i)
```

其中：

- `c_i` — 局部曲率（KNN PCA 特征值比 λ_min/λ_max）
- `|∇I_i|` — 强度梯度绝对值（中心差分）
- `M_sem(i)` — 语义掩码（静态 = 1，动态 = 0）
- `Norm(·)` — Min-Max 归一化到 [0, 1]

## 配置与参数调节

### 主要配置文件

#### `config/saliency.yaml`

```yaml
saliency:
  alpha: 0.5         # 曲率权重
  beta: 0.3          # 强度梯度权重
  gamma: 0.2         # 语义权重
  curvature_k: 5     # KNN 近邻数

weighted_icp:
  a: 200.0
  b: 55.0

dynamic_filter:
  th_pose_translation: 0.05   # m
  th_pose_rotation: 0.02      # rad
```

### 常见调参场景

**噪声水平很高？** 修改 `evaluation.py` 第 95–98 行：

```python
high_noise_config = LidarConfig(
    noise_ratio_near=0.05,
    noise_ratio_far=0.07,
    dropout_rate=0.02,
)
```

**想要更精细的特征？** 修改 `evaluation.py` 第 105–106 行：

```python
feat = LineFeatureExtractor(
    split_thresh=0.008,   # 降低到 0.8cm
    min_points=1
)
```

**想要更多特征点？** 修改 `evaluation.py` 第 135 行：

```python
sal_thresh = np.percentile(f0["saliency"], 15)  # 保留前 85%
```

## 传感器模型（RPLIDAR A1）


| 参数       | 数值                   |
| ---------- | ---------------------- |
| 量程       | 0.15 m – 12.0 m       |
| 角度分辨率 | 0.5°（720 束）        |
| 扫描频率   | 5.5 Hz                 |
| 近距离噪声 | σ = 2% · d（≤ 3 m） |
| 远距离噪声 | σ = 3% · d（> 3 m）  |
| 角度噪声   | σ = 0.25°            |
| 丢包率     | 0.5%（可配置）         |

## 参考文献

[^1]: Yu, S.; Fu, C.; Gostar, A.K.; Hu, M. "A Review on Map-Merging Methods for Typical Map Types in Multiple-Ground-Robot SLAM Solutions." *Sensors* 2020, 20, 6988. https://doi.org/10.3390/s20236988
    
[^2]: Lakämper, R. et al. "Incremental multi-robot mapping." *2005 IEEE/RSJ International Conference on Intelligent Robots and Systems* (2005): 3846–3851.
    
[^3]: Wang, K.; Chen, K.; Guo, J.; Lu, J. "Saliency-LOAM: Saliency-Based LiDAR Odometry and Mapping." *IEEE Transactions on Instrumentation and Measurement*, vol. 75, pp. 1–9, 2026, Art no. 8500309. doi: 10.1109/TIM.2025.3643077.
