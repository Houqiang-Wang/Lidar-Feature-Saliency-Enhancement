# Saliency-LOAM 技术文档

来源：IEEE TIM Vol.75 2026, Saliency-LOAM: Saliency-Based LiDAR Odometry and Mapping
整理目的：为 2D LiDAR 仿真环境提供算法实现依据

---

## 1. 系统架构（Fig.1 简化）

整体流程：
原始点云 -&gt; 圆柱投影(3D)/直接使用(2D) -&gt; 显著性预测 -&gt; 特征提取(线+面/2D线+角点) -&gt; 显著性加权匹配 -&gt; 位姿估计

2D 简化：

- 跳过圆柱投影，2D scan 直接是 (x,y) 数组
- 3D 的"面特征"在 2D 中不存在，只保留"线特征"和"角点"
- Saliency-Net(U-Net+FIDNet) 替换为手工显著性融合规则

---

## 2. 显著性权重设计（Section II-A）

输入信息类型：

1. 几何距离 r_i = sqrt(x_i^2 + y_i^2 + z_i^2)  -&gt;  2D: r_i = sqrt(x_i^2 + y_i^2)
2. 局部曲率 c_i：邻域点距离方差
   2D 简化：用空间 K 近邻 PCA，小特征值/大特征值比值作为曲率度量
3. 强度梯度 |grad I_i|：KNN 高斯加权梯度
   2D 简化：相邻 scan 点 intensity 差分
4. 语义权重 M_sem：静态=1，动态=0，非语义=0

显著性融合公式（双层 Min-Max 归一化）：
S = Norm( Norm(I_avg * I_s) + Norm(w_d * D_s) + Norm(C_avg) + M_sem )

其中 Norm(x) = (x - min(x)) / (max(x) - min(x))

2D 手工版简化：
S_i = alpha * Norm(c_i) + beta * Norm(|delta I_i|) + gamma * M_sem(i)
然后 Min-Max 归一化一次到 [0,1]

参数建议：
alpha=0.5, beta=0.3, gamma=0.2
如果无 intensity，则 beta=0，alpha=0.6, gamma=0.4

---

## 3. Saliency-Net（Section II-B）-&gt; 2D 替代方案

原论文网络结构：

- 初始显著性预测：U-Net + 残差块，编码器-解码器，下采样只在宽度方向(stride=2)
- 语义分割：FIDNet 分类头，空洞卷积(dilation 3/6/9)
- 输出拼接后下采样得最终显著性图

2D 仿真替代：
无需神经网络。直接基于上述 2. 中的手工规则计算显著性。
原因：仿真环境已知 ground truth 语义标签，无需学习。

---

## 4. 特征匹配与位姿估计（Section II-C）

### 4.1 线特征残差（公式 6 的 2D 化）

3D 原文：
e_l = w_l * d_l = (a*S_Pl^2 + b)/255 * |cross(P_lA_l, P_lB_l)| / |A_lB_l|

2D 实现：

- P_l：当前帧线特征点 (x,y)
- A_l, B_l：参考帧中对应线段的两个端点
- cross_2d(AP, AB) = AP_x * AB_y - AP_y * AB_x （标量）
- d_l = |cross_2d| / |AB|
- w_l = (a * S_i^2 + b) / 255.0
- e_l = w_l * d_l

### 4.2 面特征残差（3D 公式 7）

2D 中不存在面特征，此部分忽略。

### 4.3 总损失函数

loss = sum(e_l for all line features) + sum(e_p for all plane features)
2D 版：loss = sum(e_l for all matched line points)

### 4.4 位姿优化

优化变量：T = [x, y, theta] （2D 刚体变换，3自由度）
变换公式：
x' = x*cos(theta) - y*sin(theta) + tx
y' = x*sin(theta) + y*cos(theta) + ty

迭代优化：高斯-牛顿或 Levenberg-Marquardt
Jacobian：J = d(loss)/d(T) 通过链式法则分解

- de/dP * d(TP)/d(delta_xi)
- 第一部分：残差对点的导数
- 第二部分：点对视姿扰动的导数（2D 旋转矩阵求导）

实际实现：可直接用 scipy.optimize.least_squares(method='lm')

---

## 5. 动态物体抑制（Algorithm 1）

伪代码逻辑：

输入：

- 当前帧点云 P_i (N个点)
- 显著性值 S_i
- 标签特征比例阈值 TH_per
- 位姿偏差阈值 TH_pose

输出：最终位姿估计 pose_f

步骤：

1. 计算标签特征比例 per = N_label / N
2. 如果 per &lt;= TH_per：
   - 标签特征不足，禁用显著性约束：S_label = 0
   - pose1 = ICP(P_i, S_i)  // 帧间，带显著性
   - pose_f = Optimize(MapICP(P_i, S_i, pose1), pose1)
3. 否则：
   - pose2 = ICP(P_i, S_i)  // 帧间，带显著性
   - 临时保存 S_label，然后设 S_label = 0
   - pose3 = MapICP(P_i, S_i, pose2)  // 地图匹配，无显著性约束
   - 如果 pose3 与 pose2 的偏差 &gt;= TH_pose：
     - 存在动态点，pose_f = Optimize(pose3, pose2)
       否则：
     - 恢复 S_label
     - pose_f = Optimize(MapICP(P_i, S_i, pose3), pose2)

2D 简化：

- 直接比较 frame-to-frame(带显著性) 和 frame-to-map(无显著性) 的位姿
- 计算位移差和角度差
- 如果超过阈值，认为有动态干扰，降低动态区域显著性为 0
- 最终位姿优先采用 frame-to-map 结果

位姿偏差计算：
translation_error = sqrt((x1-x2)^2 + (y1-y2)^2)
rotation_error = abs(theta1 - theta2) 并归一化到 [-pi, pi]

---

## 6. 实验参数参考（Section III）

KITTI 数据集上的对比方法：A-LOAM, F-LOAM, KISS-ICP 等
关键指标：ATE (Absolute Trajectory Error)

计算时间：

- 显著性预测 + 特征匹配最耗时
- 目标：&lt;100ms/scan (&gt;10Hz)

2D 仿真预期：

- 手工显著性计算：O(N*K) N=点数, K=近邻数，极快
- ICP 优化：取决于 scipy 迭代次数，通常 &lt;50ms

---

## 7. 2D 迁移检查清单


| 3D 原模块             | 2D 对应                      | 状态 |
| --------------------- | ---------------------------- | ---- |
| 圆柱投影              | 无需，scan 直接是 (x,y)      |      |
| 线特征提取            | Split-and-Merge / RANSAC     |      |
| 面特征提取            | 不存在                       |      |
| Saliency-Net          | 手工规则融合                 |      |
| 加权残差 e=w*d        | 点到线距离 * 显著性权重      |      |
| Lie 代数优化          | scipy.optimize.least_squares |      |
| 动态抑制(Algorithm 1) | 双估计阈值判断               |      |
