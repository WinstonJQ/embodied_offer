# 卷七 · 3D 感知 / SLAM / VLN / 零样本 ObjectNav / Embodied VLM · 高频面试题

> 中文具身智能秋招高频面试题库 · **第七卷**
> 题源：牛客 / 知乎 / CSDN 公开面经 / 深蓝学院题库 / GankInterview 总结 / VLN/ObjectNav arXiv 高引论文与项目页 / GitHub awesome-* 仓库
> 同义题合并后 **60 题**（主表 57 题：频次 ≥3 或权威破例 + 低频备选 3 题）

**难度分布**：L1（必会） **10** · L2（进阶） **35** · L3（顶级 / 前沿） **12** （低频备选另计：L2 ×2 / L3 ×1）

**使用方式**：题目默认折叠，点开看答案。建议按 **L1 → L2 → L3** 顺序刷；同级内按频次从高到低。手机端原生支持。

**重点章节**：本卷 §5 VLN / §6 零样本 ObjectNav / §7 Embodied VLM 三节共 30 题，是 2024-2026 视觉语言导航 / VLA 算法岗的核心考察面，L3 难题多集中在这三节。§1-§4 是支撑性章节（3D 感知 / NeRF / SLAM / 经典导航栈），偏 L1/L2，是上层方法的"基本功"。

**与其他卷的关系**：本卷与卷三（VLA / 模仿学习）在 prompt 工程 / VLM 调用 / cross-modal 表征上有少量重叠——视角不同：卷三讲 manipulation 上的 VLA 架构，本卷讲 navigation 上的 VLM 应用。

---

## §1 3D 感知基础（6 题）

> 点云 / 深度 / 占用 — 这些是几乎所有上层导航算法的输入侧基本功。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q01</b> · PointNet 怎么解决点云的"无序性 / 置换不变性"问题？为什么能做点云分类的开山工作？</summary>

**答**：点云的根本性质是**无序集合**——同一个物体的 $N$ 个点排列方式有 $N!$ 种，网络对任意置换必须输出同一个特征。PointNet（Qi 等 CVPR 2017）的解决方案：

1. **共享 MLP**：每个点独立过同一个 MLP 升维，逐点变换不会引入顺序依赖
2. **对称聚合函数**：用 max-pooling 把 $N$ 个点的特征聚合成全局特征——max 是对称函数，对置换天然不变
3. **T-Net**：轻量 STN-like 子网络估计 $3\times3$ / $64\times64$ 变换矩阵，对齐输入和特征

**为什么开山**：之前主流是把点云体素化（voxelization）后用 3D CNN，分辨率受限于内存；PointNet 直接吃点、参数少（约 3.5 M），ModelNet40 准确率 89.2%。

**易错**：max-pooling 对**离群点敏感**——异常远的一两个点会主导全局特征；PointNet++ 用层次化 FPS + grouping 缓解。"置换不变"和"旋转不变"是两回事，T-Net 只对**仿射变换**部分对齐，不能完全解决旋转不变。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q02</b> · PointNet++ 相比 PointNet 改进了什么？hierarchical FPS + grouping 解决什么问题？</summary>

**答**：PointNet 的核心缺陷是**没有局部结构**——只有 per-point MLP + 全局 max-pool，丢掉"哪些点是邻居"的几何信息，复杂场景（语义分割、室内检测）效果差。PointNet++（NeurIPS 2017）三步改造：

1. **FPS**（Farthest Point Sampling）：从 N 个点采 M 个中心点，比随机均匀
2. **Grouping**：以每中心点 ball-query 半径 r 内邻居形成局部簇
3. **局部 PointNet**：每簇上小 PointNet 提特征 → 再 FPS+Group → 层次化

**密度自适应**：MSG（multi-scale）多尺度并联；MRG（multi-resolution）多层级拼接——稀疏区粗特征兜底。

**易错**：ball-query 半径是**超参数**，调不对漏邻居或纳入太多噪声；近年 PointMLP / Point Transformer 都用 KNN 省掉此超参。ModelNet40 ↑1.3pt 不大，**真正提升在 ScanNet**：PointNet 53% mIoU → PointNet++ 64%。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q03</b> · Point Transformer / PointMLP / PVCNN 各自定位？为什么近几年 PointNet++ 仍是 baseline 但很少作 SOTA？</summary>

**答**：

| 方法 | 核心机制 | 优势 / 劣势 |
|---|---|---|
| **PointNet++** | FPS + ball-query + 局部 MLP | 简单兼容 / 局部表达力弱 |
| **PVCNN** | Point + Voxel 双分支并联 | 快、内存低 / 体素分辨率受限 |
| **Point Transformer**（Zhao 2021） | vector self-attention 替 MLP | 几何感强 SOTA / 慢，$O(Nk)$ |
| **PointMLP**（ICLR 2022） | 纯 MLP + 几何 affine 归一 | 训练快超 Transformer / 缺解释性 |

**为什么 PN++ 还在**：① 代码极轻；② 工业部署友好（无 attention）；③ "Revisiting PointNet++"（BAAI 2022）显示改训练策略 + 大模型 PN++ 也能逼近 SOTA。

**易错**：sparse convolution（Minkowski / SPVNAS）是**另一条线**——对大规模室外场景（自驾 LiDAR）显著有优势；室内点数 < 100k，点级方法仍主流。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q04</b> · 单目深度估计：metric depth 和 relative depth 的区别？什么时候必须用 metric？</summary>

**答**：

- **Relative depth**：只输出**相对**深度（哪个点更远），整体可以差一个未知缩放和偏移因子。训练时用"尺度无关" loss（如 SSI-MAE）；可在跨数据集预训练得到强泛化
- **Metric depth**：输出**绝对**深度（米为单位），必须有相机内参、训练数据有真值标定。跨域泛化差，相机 FoV / 焦距变了就崩

**什么时候必须用 metric**：
- **导航 / 避障**：要知道"墙在 1.2 m 外"才能给出速度命令
- **建图 / SLAM 融合**：把单目深度灌进占用栅格必须米单位
- **抓取**：末端位置需要绝对距离

**什么时候 relative 够用**：纯视觉显著性、portrait 模式背景虚化、单图深度可视化。

**易错**：MiDaS 是 **relative**，直接拿来做导航**必崩**；ZoeDepth / Metric3D / Depth Anything v2（metric 微调版）是 metric。Depth Anything v1 默认是 relative，需要在 NYU / KITTI 上 metric 微调才能给绝对深度。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q05</b> · Depth Anything v2 vs MiDaS vs ZoeDepth 怎么选？室内 / 室外有差别吗？</summary>

**答**：

| 模型 | 类型 | 强项 / 弱项 |
|---|---|---|
| **MiDaS v3.1**（2022） | relative | 跨域鲁棒老 baseline / 不是 metric，细节模糊 |
| **ZoeDepth**（2023） | metric | 真米单位零样本可用 / 边缘退化，室外一般 |
| **Depth Anything v2**（2024） | relative + metric | SOTA 细节 + 强泛化 / 算量比 v1 大 |

**选择**：室内导航 / 抓取 → DA v2 metric 或 Metric3D v2；室外远距 → ZoeDepth (KITTI) 或 Metric3D v2；只要相对结构 → DA v2 relative 默认版。

**易错**：DA v2 默认输出是 **disparity**（视差），需 $1/d$ 反演才是深度；KB/KITTI 模式直接输出米。ViT-L 单帧 CUDA / TensorRT 约 30-80 ms，端到端实时需要 ViT-S。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q06</b> · occupancy network 和 2D occupancy grid map 的本质区别？BEV 和 voxel 各自适用什么场景？</summary>

**答**：两类思想不同：

| | 2D Occupancy Grid | Occupancy Network |
|---|---|---|
| 结构 | 平面栅格 + 占用概率 | 3D voxel / 隐式连续 |
| 学习 | log-odds Bayes 更新 | 神经网络端到端预测 |
| 代表 | gmapping / Cartographer | Tesla Occupancy / SurroundOcc |
| 用途 | 2D 规划 / SLAM | 自驾 BEV / 机械臂避障 |

**BEV vs 体素**：BEV 省内存规划友好但**忽略高度**（桌下椅腿、低矮障碍物会丢）；Voxel 保 z 维能表达悬空 / 透空 / 多层，但内存 $O(LWH)$ 分辨率受限。

**实务**：自驾 2023 后从纯 BEV 转向 occupancy（Tesla AI Day 2022 推动）；室内移动机器人仍以 2D grid 为主。

**易错**：occupancy network 是**端到端学习**输出，与 grid map 的 Bayesian filter 累积是两种范式；别说成"grid map 的 3D 版"。

</details>



## §2 NeRF / 3DGS 在机器人里（5 题）

> 神经渲染 2023 年后从计算机视觉外溢到机器人——核心驱动是"用少量 view 重建可查询的 3D 场景表示"。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q07</b> · NeRF 和 3D Gaussian Splatting 的核心区别？为什么 3DGS 在机器人圈快速取代 NeRF？</summary>

**答**：

| | NeRF | 3DGS |
|---|---|---|
| 表征 | 隐式 MLP $(x,y,z,\theta,\phi) \to (\sigma, c)$ | 显式高斯椭球 $\{\mu, \Sigma, c, \alpha\}$ |
| 渲染 | volumetric ray marching | 光栅化 splatting（GPU 友好） |
| 训练 | 数小时（原文 12h） | 几十分钟（20-30 min）|
| 渲染速度 | ~1 FPS（原版） | 100+ FPS |
| 编辑 | 难（黑箱 MLP） | 容易（增删高斯） |

**为什么 3DGS 在机器人圈走红**：实时渲染 100+ FPS（closed-loop 控制 / VR teleop 可行）；可编辑（物体级增删移动）；显式几何（高斯中心可直接当点云用，GaussianGrasper / SplatSim 下游友好）。

**易错**：3DGS 训练快主要是**前向便宜**；反光 / 透明 / 大尺度室外仍劣于最强 NeRF（Zip-NeRF）。"SLAM 后端拿稀疏 GS 当地图"这条路还在演进，没有定论。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q08</b> · F3RM（Feature Fields for Robotic Manipulation）是怎么把 CLIP 特征蒸馏到 NeRF 里的？为什么操作场景 view 少是个大问题？</summary>

**答**：F3RM（Shen 等 CoRL 2023）思路：

1. 用 **DINO + CLIP** 对每帧图像抽 patch 级特征
2. NeRF 学一个**额外的特征通道** $f(x)$：把多视角特征"提升"到 3D 空间
3. 推理时给文本 query（如"红色杯子"），用 CLIP 文本编码器算 cosine similarity 选最相关 3D 区域
4. 用经典 Anti-Podal grasp planner 在该区域生成抓取候选

**为什么 view 少是大问题**：原版 NeRF 需要 ~30 个视角才能 converge；操作任务里相机机位往往**只能从机械臂第三视角扫几个角度**（5-10 view），稀疏视角下 NeRF 会出现"漂浮云""空洞"伪影。F3RM 用 50 view 已经算比较奢侈。

**易错**：F3RM 蒸馏的是 **patch 级**特征，不是 pixel 级——细节上的对象边界仍模糊；后续 GaussianGrasper / LERF-TOGO 都改进了这一点。F3RM 不是 zero-shot，每个新场景**仍需要重训 NeRF**（虽然 patch encoder 是预训练的）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q09</b> · GaussianGrasper 相对 F3RM 的优势是什么？怎么做 open-vocabulary grasping？</summary>

**答**：GaussianGrasper（Zheng 等 IEEE RA-L 2024，arXiv 2403.09637）三大改进：

1. **3DGS 代替 NeRF**：训练 30 min → < 10 min；推理 100+ FPS vs F3RM 1-2 FPS
2. **少视角友好**：原文用 ~10 view RGB-D 就能重建（F3RM 需 50 view）
3. **EFD 蒸馏 + 对比学习**：把 SAM 分割 + CLIP 特征在 GS 上做对比学习蒸馏，特征更精确

**open-vocabulary grasping 流程**：
- 文本 query → CLIP 编码 → 与每个高斯的特征 cos similarity → 阈值得 mask → 该 mask 区域作为抓取候选 region → 经典 GraspNet 或 AnyGrasp 在该 region 生成 6-DoF 抓取

**易错**：GaussianGrasper 仍**每场景重训**，不是 zero-shot 部署。其优势更多在"**特征蒸馏 + GS 几何**联合"，而不仅是"换了 GS 就快"。真正实时 zero-shot grasping 还得靠 RGB-D 端到端模型（如 AnyGrasp）。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2 (破例)</span> <b>Q10</b> · NeRF / 3DGS 实时性瓶颈在哪？训练 / 渲染 / 编辑 哪个最难做 < 30 Hz？</summary>

**答**：三环节延迟来源不同：

| | NeRF 瓶颈 | 3DGS 瓶颈 |
|---|---|---|
| 训练 | 每像素数百次 MLP 前向 | 每像素排序 + α-blend |
| 渲染 | volumetric ray marching | tile-based 光栅化（已较快） |
| 编辑 | 改一处需重训 | 改局部高斯即可 |

**< 30 Hz 难点**：
- NeRF：**渲染就难**——Instant-NGP / NeRFAcc 加速到 30+ FPS（中分辨率），但训练仍 5-10 min
- 3DGS：**渲染稳定 100+ FPS**；**训练**是真正瓶颈（10+ min/场景，不能在线建图）
- **编辑**：NeRF 几乎无法在线编辑；3DGS 可逐高斯增删，但保持自洽（光照、阴影）仍困难

**实务**：机器人圈"地图建好后查询"做到了实时，"在线建图同时控制"仍是 2025-2026 开放问题。Gaussian-SLAM（SplaTAM / MonoGS）能 10-30 FPS 在线建图，但精度鲁棒性显著低于 ORB-SLAM3。

**破例说明**：频次 2 但来自具身/SLAM 圈"实时性瓶颈"高频开放题，来源多 work（Instant-NGP / SplaTAM）权威。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q11</b> · 端到端 NeRF / GS-based policy（如 SplatRL / GaussianActor）相比传统 BC + RGB-D 输入有什么收益和代价？</summary>

**答**：

**收益**：
- **显式 3D 几何**：策略不需要从 RGB-D 隐式重建空间，几何先验更强 → **样本效率 +**
- **跨视角一致**：相机视角变了 GS 表征不变，**视角泛化 +**
- **多模态融合**：可灌 CLIP/SAM 特征到高斯上，**语义一致性 +**

**代价**：
- **训练管线复杂**：每个新场景要先建 GS（10+ min），不能像 BC 那样直接吃 RGB
- **GS 可微渲染但策略联合优化难**：GS 渲染本身是可微的，但 densification / pruning / 在线 map 更新与策略联合优化困难，多数工作把 GS 当**固定输入**
- **泛化到新场景**：依赖 GS 重建质量，未见物体 / 遮挡仍易崩
- **算力**：高斯数 100k-1M 量级，单帧策略推理慢于纯 vision encoder

**实务**：2025 端到端 GS-policy 仍是研究阶段，**没有像 Diffusion Policy 那样可即插即用**的工业方案；主流仍是"GS 当 perception 输入"+"DP / ACT 当 policy"分离架构。

**易错**：不要把"GS-based policy"等同于"GS 重建后查询"——前者是**联合学习**，后者是**两阶段**；只有联合学习才有"端到端"含义。

</details>



## §3 SLAM 与状态估计（6 题）

> SLAM 是机器人长期工业基本盘——传统几何 SLAM 仍是 2024-2026 主流方案。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q12</b> · ORB-SLAM3 和 VINS-Fusion 的核心区别？工程上一般怎么选？</summary>

**答**：两者都是经典 VIO / V-SLAM，路线不同：

| | **ORB-SLAM3**（Campos 2021） | **VINS-Fusion**（Qin 2019） |
|---|---|---|
| 前端 | ORB 特征点匹配 | 光流跟踪（KLT） |
| 后端 | BA | 滑动窗口 BA + 边缘化 |
| 回环 | DBoW2 + 全局 BA | 词袋 + pose graph |
| 多地图 | Atlas | 单地图 |
| 传感器 | mono/stereo/RGB-D + IMU | mono/stereo + IMU + GPS |

**工程选择**：室内相机不快 → ORB-SLAM3（精度高、回环鲁棒、长期建图）；快速运动 / 弱纹理 / 无人机 → VINS-Fusion（光流动态下更稳，IMU 紧耦合更激进）；嵌入式 → VINS-Fusion 略轻，ORB3 全套 1-2 GB RAM。

**易错**：两者都用 **GPLv3** copyleft 协议——开源但商用闭源产品要谨慎（衍生必须同 GPL）；商用闭源首选 OpenVINS / Kimera / Maplab 等 BSD/MIT 系。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q13</b> · VIO 相比纯视觉里程计的优势？IMU 预积分解决了什么核心问题？</summary>

**答**：

**VIO 优势**：
- **绝对尺度**：单目纯视觉**只能恢复相对尺度**（5 倍小或大都对），IMU 给出加速度积分提供米单位真尺度
- **快速运动鲁棒**：相机在快速旋转 / 模糊时跟踪失败，IMU 在毫秒级仍稳定
- **退化恢复**：低纹理 / 黑屋子里相机失效时 IMU 可以"撑过"
- **频率高**：IMU 100-1000 Hz，能给控制器更频繁的位姿

**IMU 预积分（Forster 2017）**：
- **问题**：每次优化变量（位姿）更新，都要把窗口内全部 IMU 测量重新积分——计算爆炸
- **解决**：用"**相对量** $\Delta R, \Delta v, \Delta p$"代替绝对积分，把窗口的 IMU 测量预积分到一个**与状态无关**的常量。状态更新时只需对常量做一阶修正，不必重积分

**易错**：预积分是**离散 SO(3) 流形上的积分**，不能直接用欧氏空间公式；常用 Lie group exp/log 处理旋转。预积分**仍受 IMU 零偏 / 噪声**影响，零偏估计是 VIO 收敛关键。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q14</b> · 回环检测的两条主流路线：词袋（BoW）vs 学习型（NetVLAD / EigenPlaces）。优劣势？</summary>

**答**：

| | **BoW**（DBoW2 / DBoW3 / FBoW） | **学习型**（NetVLAD / EigenPlaces / SuperGlue） |
|---|---|---|
| 原理 | ORB/SIFT 特征聚类成视觉词典，TF-IDF 检索 | CNN 全局描述子 + 度量学习 |
| 内存 | 字典约 10-100 MB | 模型 < 100 MB，描述子 256-4096 维 |
| 速度 | < 10 ms / query | 50-200 ms / query（GPU） |
| 鲁棒性 | 光照 / 视角变化敏感 | 强（端到端学到不变性） |
| 部署 | 嵌入式友好 | 依赖 GPU |

**ORB-SLAM3 / VINS-Fusion 都用 BoW**——工业稳定、可解释。**机器学习时代**学习型方法在跨季节 / 昼夜 / 视角变换上显著更鲁棒（VPR benchmark 上 NetVLAD recall@1 比 BoW 高 10-20pt）。

**易错**：回环**检测**和回环**校正**是两件事——检测告诉你"这里来过"，校正用 pose graph 优化把累积漂移摊平；BoW 只解决检测。学习型方法可与几何验证（如 RANSAC + PnP）级联，避免 false positive。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q15</b> · 因子图优化和滤波（EKF）的区别？为什么现代 SLAM 几乎都用因子图？</summary>

**答**：

| | **EKF** | **因子图 / 非线性优化** |
|---|---|---|
| 思想 | 高斯递推 + 一阶线性化 | 所有约束建图，全局优化 |
| 时序 | 增量式（一来一更新） | 批处理 / 滑动窗口 |
| 线性化点 | 当前状态（一固定无法修） | 每次迭代重线性化 |
| 多假设 / 回环 | 难 | 容易（加一条因子边） |
| 计算量 | $O(n^2)$ | 稀疏 $O(n)$-$O(n^{1.5})$ |

**因子图胜出根本原因**：① 可重新线性化（EKF 卡死，因子图每次重算雅可比，对非线性/大旋转友好）；② 信息矩阵天然稀疏（Cholesky / iSAM 高效）；③ 多传感器融合干净（IMU/轮速/GPS/回环都"加一条因子"）。

**易错**：EKF 在**算力极小**场景仍在用（小型微控制器）。现代 SLAM "用因子图"通常是**滑动窗口优化**，保留最近 N 帧因子老的 marginalize，不是真正全局 BA。GTSAM 的 iSAM2 是增量因子图，复杂度接近 EKF 但精度逼近批 BA。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q16</b> · Bundle Adjustment 在 SLAM 后端的角色？滑动窗口 BA 怎么控制规模？</summary>

**答**：**BA**：以**重投影误差**为代价，联合优化关键帧位姿 $\{T_i\}$ 和路标点 $\{P_j\}$：

$$\min \sum_{i,j} \rho\left( \| u_{ij} - \pi(T_i, P_j) \|^2 \right)$$

求解器：Levenberg-Marquardt + Schur complement 利用稀疏性消除路标点。

**滑动窗口 BA 控制规模**：① 关键帧选择（旋转/平移阈值）；② 窗口大小（最近 10-20 关键帧 + 共视帧，老帧 marginalize）；③ 共视图 covisibility graph（local BA，全局 BA 仅回环触发）。

**易错**：滑动窗口边缘化老变量时**信息矩阵 fill-in**（变稠密），用 first-estimate Jacobian / iSAM2 保持稀疏。路标点数 >> 位姿数，BA 计算量主要被路标点 dominated，必须 Schur 消元。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q17</b> · sparse SLAM（ORB-SLAM）和 dense SLAM（KinectFusion / NICE-SLAM）的取舍？</summary>

**答**：

| | sparse SLAM | dense SLAM |
|---|---|---|
| 地图 | 数百 - 数千 ORB 路标点 | 体素 / TSDF / mesh / NeRF |
| 内存 | 几 MB - 几十 MB | 几百 MB - 几 GB |
| 用途 | 定位 / 路径回放 | 建图 / 避障 / 重建 |
| 实时性 | 30+ Hz | KinectFusion 30 Hz（小场景）；NeRF-SLAM 1-10 Hz |
| 代表 | ORB-SLAM3 / VINS-Fusion | KinectFusion / ElasticFusion / NICE-SLAM / Gaussian-SLAM |

**取舍核心**：
- 只要"机器人在哪" → sparse 够用
- 要"机器人能不能撞墙 / 抓桌子" → dense（要稠密几何）
- 现代实践常**分离**：sparse SLAM 做实时定位，dense map 离线后处理或低频构建

**易错**：dense SLAM 不等于"更准"——稀疏 SLAM 的**关键帧位姿精度**通常更高（路标点少，BA 更稳）；dense 的优势是**几何完整性**。NeRF-SLAM / Gaussian-SLAM 是近 2 年的方向，工业可用度低于 ORB-SLAM3。

</details>



## §4 经典导航栈与地图表征（5 题）

> 经典栈仍是工业落地（家庭服务机器人、AGV、扫地机）的主流——VLN / ObjectNav 也常用它做底层 controller。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q18</b> · A* / Dijkstra / RRT / RRT* 怎么选？哪个能找最优？哪个能高维？</summary>

**答**：

| 算法 | 类型 | 最优性 | 维度 | 适用 |
|---|---|---|---|---|
| **Dijkstra** | 图搜索 | 全局最优 | 低（2D） | 已知静态地图、规模小 |
| **A\*** | 图搜索 + 启发 | 启发可采纳时最优 | 低 | 加速版 Dijkstra |
| **RRT** | 采样 | 概率完备**非**最优 | 高 | 高维状态空间 / 复杂约束 |
| **RRT\*** | 采样 + rewire | **渐近最优** | 高 | RRT 但允许迭代到最优 |

**选择**：2D 平面 / 栅格 → A\*（最常用）；机械臂 7-DoF → RRT-Connect / RRT\*；移动机器人已知地图要最优 → A\*；动态环境 → A\* 全局 + DWA/MPC 局部。

**易错**：A\* 启发函数必须**可采纳**（不高估真实代价）才保证最优——曼哈顿距离对 4-邻接可采纳，8-邻接要用欧氏。RRT 不能保证最优解（可能绕大圈），必须 RRT\* 的 rewire 才能逼近最优。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q19</b> · Nav2 vs ROS1 move_base：behavior tree + lifecycle 解决了什么痛点？</summary>

**答**：ROS1 `move_base` 是单体大节点，状态机硬编码在 C++ 里——**痛点**：故障恢复策略写死难定制；单节点宕机整栈崩；状态机不可视化不可热改。

**Nav2 三大改造**：① **Behavior Tree**：导航流程用 BT XML 描述，节点（ComputePathToPose / FollowPath / Recovery）可拖拽组合——故障恢复可写"先转 30 度再重规划"；② **lifecycle 节点**：每 server 独立 unconfigured → configured → activated 生命周期，单点故障可重启不影响整栈；③ **插件化**：planner / controller / behavior / costmap layer 都 pluginlib 动态加载——可热插拔 A\* / NavFn / Smac / TEB / DWB。

**易错**：Nav2 不只"move_base 移植到 ROS2"——是**架构重写**。但学习曲线陡（BT + lifecycle + 插件），新手项目慎用，BT 调试难（状态多日志多）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q20</b> · 全局规划 vs 局部规划（DWA / TEB）：分工和接口是什么？</summary>

**答**：

| | **全局 Planner** | **局部 Controller** |
|---|---|---|
| 目的 | 起点到终点长程最优路径 | 沿全局路径短程避障 |
| 输入 | 全局 costmap | 局部 costmap（实时传感器） |
| 输出 | path 路径点 | cmd_vel ($v, \omega$) |
| 算法 | A* / Dijkstra / NavFn / Smac | DWA / TEB / MPC / Pure Pursuit |
| 频率 | 1-5 Hz | 10-30 Hz |

**接口**：全局发 `Path`，局部订阅 path + odom + scan 输出 `Twist`。两者解耦可独立替换。

**DWA vs TEB**：DWA 速度空间采样 + 评分（路径偏离、目标距离、速度），快稳但只算下一步；TEB 轨迹建成弹性带 + 时间维度优化整段，能处理倒车 / 复杂动态约束，但慢。

**易错**：局部规划器**不应**重规划全局——会死锁。卡住时触发 Recovery Behavior（旋转 / 后退 / 清 costmap），不是叫起全局规划器。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q21</b> · 占用栅格地图：log-odds 更新、未知/空闲/占用三态怎么编码？</summary>

**答**：栅格地图本质：每个 cell 维护占用概率 $p \in [0,1]$，0 空闲 / 1 占用 / 0.5 未知。

**为什么用 log-odds**：直接用 $p$ 更新数值不稳（接近 0/1 时溢出）；log-odds $l = \log \frac{p}{1-p}$ 无界，更新变加法 $l_t = l_{t-1} + l_\text{sensor}$。

**典型实现**：三个常量 $l_\text{free} = -0.4$、$l_\text{occ} = +0.85$、$l_0 = 0$；ray casting 从机器人到激光击中点，沿途 cell 减 $|l_\text{free}|$，击中点加 $l_\text{occ}$；加 clamp 防饱和。

**三态查询**：$p < 0.3$ free / $p > 0.7$ occupied / 否则 unknown。

**易错**：cell 独立假设其实不成立（邻居相关），但工程够用。激光 ray casting $O(L)$/ray，LiDAR 一帧 1000+ 线量大，gmapping / Cartographer 用 Bresenham 加速。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q22</b> · 拓扑地图 vs 度量地图 vs 语义地图 各自适用于什么任务？</summary>

**答**：

| | **度量地图** | **拓扑地图** | **语义地图** |
|---|---|---|---|
| 表征 | 占用栅格 / 点云 / mesh | 节点（场所）+ 边（连通） | 度量 + 物体类别 / 关系 |
| 精度 | 厘米级 | 房间级 | 厘米 + 类别 |
| 内存 | 大 | 极小 | 中-大 |
| 适合查询 | "距离墙 0.5 m" | "客厅到厨房怎么走" | "找冰箱" |
| 代表 | gmapping / Cartographer | 老式 NavGraph、Hyam | VLMaps / ConceptFusion / SemExp 语义图 |

**实务组合**：
- 长程 VLN / ObjectNav → **语义图**（VLMaps）+ 度量底图（占用栅格）
- 多楼层导航 → **拓扑图**（楼层节点）+ 每层度量图
- 局部避障 → 纯度量栅格

**易错**：语义地图不等于"在度量图上贴标签"——更现代的做法是**特征场**（VLMaps / ConceptFusion）：每个体素存 CLIP / SigLIP 特征向量，查询时用文本编码做 cos similarity，**零样本**支持任意类别。

</details>



## §5 VLN 视觉语言导航（10 题）

> 用户研究方向核心节。VLN 是 2024-2026 具身 AI 最活跃方向之一——R2R 数据集 2018 年提出，至今几乎每个 NeurIPS / CVPR / RSS / CoRL 都有新方法。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q23</b> · R2R / RxR / REVERIE / SOON 这几个 VLN 数据集的差别？指令复杂度和任务定义</summary>

**答**：

| 数据集 | 平台 / 长度 | 任务定义 | 核心差别 |
|---|---|---|---|
| **R2R**（Anderson CVPR 2018） | MP3D / ~29 词 | 指令导航到点 | VLN 鼻祖，step-by-step |
| **R4R / R8R**（Jain 2019） | MP3D / 58-116 词 | R2R 路径拼接 | 长路径泛化压测 |
| **RxR**（Ku EMNLP 2020） | MP3D / ~78 词 | 英语 + 印地语 + 泰卢固语 | 多语 + 长指令 + 全程动作对齐 |
| **REVERIE**（Qi CVPR 2020） | MP3D / ~21 词 | 找物体并指认 | 任务变"找物体" + 输出 bbox ID |
| **SOON**（Zhu CVPR 2021） | MP3D / ~38 词 | 找属性匹配物体 | 加空间关系 + 属性（颜色、相对位置） |

**难度演进**：动作级（R2R） → 目标级（REVERIE/SOON） → 多语长指令（RxR）。

**易错**：REVERIE 是"**找物体 + 指认**"，最后必须输出 **bbox ID**，是"指代消解 + 导航"联合任务，不要等同于 ObjectNav 的"接近物体"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q24</b> · VLN-CE 相比离散 VLN 的核心区别？waypoint predictor 解决了什么问题？</summary>

**答**：

**离散 VLN**（R2R 原版）：MP3D 每房间预定义**有向图**（节点 + 边），agent 在节点间跳跃。**与真机脱节**——真机要走米单位连续动作。

**VLN-CE**（Krantz ECCV 2020）：Habitat 里跑 R2R，agent 输出**低层离散动作**（前进 0.25 m / 转 15° / 停止——动作集是离散的，但状态空间连续）。**搜索空间爆炸**——每步十几动作连乘，长指令规划难。

**waypoint predictor 折中**：先预测"下一个可达候选点"集合（12 方向、0.5/1/2 m），选一个候选再用 PointNav controller 走过去。代表：CMA-Waypoint（Hong 2022）、ETPNav（An et al. TPAMI 2024）。

**易错**：VLN-CE 难度**远高于**离散 VLN——SOTA SR 从 70%+ 掉到 50% 上下；很多论文报"VLN 60% SR"不区分 VLN / VLN-CE 是面试雷点。NaVid / VLN-R1 都直接在 VLN-CE 上做评测。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q25</b> · VLN-BERT / HAMT / DUET 的演进路线？history memory / panorama / cross-modal attention</summary>

**答**：VLN 模型架构三代演进：

1. **VLN-BERT**（Hong CVPR 2021）：BERT 套 VLN，文本-视觉 cross-attention，每步独立看 panorama，**无显式历史**——长指令易迷路
2. **HAMT**（Chen NeurIPS 2021）：引入**历史 transformer**，把过去 K 步 panorama + 动作编码成 history tokens 与当前一起 cross-attention；解决"看 panorama 不记得"
3. **DUET**（Chen CVPR 2022）：**dual-scale 双尺度**——粗尺度 topology graph（去过哪些节点）+ 细尺度局部 panorama（当前看到啥），两分支 cross-fuse

**演进核心**：无历史 → 时序历史 → 拓扑空间历史。R2R SR：VLN-BERT 63% → HAMT 66% → DUET 72%。

**易错**：DUET topology graph **在线构建**——agent 走过自动加，不需要先验地图。HAMT history token 不是无限增长——窗口 K 是超参，过长 token 数太多。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q26</b> · NaVid 怎么做到"无地图、无深度、无里程计"也能做 VLN-CE？</summary>

**答**：NaVid（Zhang RSS 2024）核心思路：**把 VLN-CE 当视频理解任务**，用 LLaMA-VID 改造的 video-VLM 直接吃单目 RGB 视频流，输出下一步低层动作。

**架构**：输入 = 当前帧 + 过去 32 帧 RGB（不要 panorama）+ 文本指令；视觉编码每帧 EVA-CLIP 抽特征 → 时序压缩到 4 token/帧；LLaMA-7B LoRA 微调输出文本动作（"前进 0.25 m" / "右转 15°"）；训练 510k VLN trajectory。

**为什么"无地图无深度"能 work**：视频隐式编码空间几何（光流暗含深度）；LLM 常识助推（"看到楼梯就上楼"）；指令本身是高层语义，不需厘米级精度。

**易错**：NaVid **不是 zero-shot**——仍需 510k trajectory SFT；相比 BERT 系是"VLM 替代任务专家网络"。SR 约 37% on R2R-CE val-unseen——仍低于专家方法（ETPNav 57%）但接近 SOTA。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q27</b> · Uni-NaVid 怎么统一 VLN / ObjectNav / EQA / 人物跟随四类任务？token merging 起什么作用？</summary>

**答**：Uni-NaVid（Zhang RSS 2025，arXiv 2412.06224）三大设计：

1. **统一任务模板**：四类任务都改写为"指令 + 视频流 → 文本动作"——VLN（自然语言指令）/ ObjectNav（`find a chair`）/ EQA（`Q: where is the bed?`）/ Tracking（`follow the person in red`）
2. **3.6M 多任务联合训练**：4 类任务轨迹混合微调
3. **在线 token merging**：类似 ToMe 逐帧合并相似 token，把单帧 4 token 压到全程 < 100 token

**token merging 作用**：推理加速（单 A100 5 Hz）+ 长程记忆（100+ 帧不爆显存）+ 隐式去冗余（相似帧合并 = 自动 keyframe）。

**易错**：Uni-NaVid 是 **NaVid 统一升级版**——四个任务都跑赢专门 baseline 但通常不及最强专家。task synergy 中等收益（+2-5pt）不是"涌现"。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q28</b> · NaVILA 的两级框架：VLM 出 mid-level action + RL low-level locomotion，分层为什么必要？</summary>

**答**：NaVILA（Lin RSS 2025，UC San Diego）针对**腿足机器人 VLN**——四足/人形不能"瞬移到 waypoint"，必须考虑步态、平衡、地形。

**两级架构**：High-level VLM 吃图像 + 指令输出 mid-level action（"右转 30°"/"前进 1 m"），1-2 Hz；Low-level RL locomotion 吃 mid-level + 本体状态输出关节力矩，50-100 Hz。

**为什么分层必要**：① 频率失配（VLM 1 Hz vs 控制 50+ Hz）单网难兼顾；② 语义 vs 动力学训练数据来源不同（VLM 互联网视频；locomotion sim2real RL），**分别迭代不互相干扰**。

**对比 NaVid**：NaVid 输出"前进 0.25 m"适合**轮式**；NaVILA 输出"前进 1 m"让 locomotion 自己规划步态，适合腿足。

**易错**：NaVILA low-level 仍是预训练 RL 策略（如 Walk These Ways），非端到端联合学习；两级解耦意味着 mid-level command 不合理（"穿墙"）底层不会反馈纠错。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q29</b> · VLN 评估指标 SR / SPL / nDTW 各自衡量什么？SR 高但 SPL 低说明什么？</summary>

**答**：

- **SR（Success Rate）**：最终停止时距目标 < 3 m（R2R 阈值）的轨迹占比。**只关心是否到达**
- **SPL（Success weighted by Path Length）**：$\text{SPL} = \text{SR} \cdot \frac{\ell^*}{\max(\ell^*, \ell_\text{agent})}$。**惩罚绕路**——agent 走的越远 SPL 越低
- **nDTW（normalized Dynamic Time Warping）**：把 agent 轨迹和参考轨迹做 DTW 对齐，归一化到 $[0, 1]$。**衡量轨迹形状是否贴近参考**——更适合 RxR 这种"按指令走"任务
- **NE（Navigation Error）**：终点距目标的距离，越小越好
- **OSR（Oracle SR）**：路径中**任一点**离目标 < 3 m 就算成功——衡量"看到过没"

**SR 高 SPL 低意味着**：到达目标但**绕了一大圈**——典型如随机探索 + 偶然撞上。reward 设计偏向 SR 时（如 IL）会出现此情况；reward 加入路径长度惩罚后 SPL 提升。

**易错**：R2R 的"成功"阈值 3 m 是**任务约定**，不是物理意义——agent 可能"看到了目标但没靠近"也算失败。SPL 是 SR 的**严格上界**（$\text{SPL} \le \text{SR}$），看到 SPL > SR 就是 bug。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q30</b> · low-level action space vs high-level action space 在 VLN 里的取舍？</summary>

**答**：

| | **low-level**（VLN-CE 默认） | **high-level**（waypoint） |
|---|---|---|
| 动作 | 前进 0.25 m / 转 15° / 停 | 选 N 个候选 waypoint 之一 |
| 决策频率 | 每米 ~4 步 | 每路口 1 步 |
| 序列长度 | 长（30-100 步） | 短（5-15 步） |
| 真机迁移 | 接近真实控制 | 需 PointNav controller 落地 |
| 长程规划 | 容易迷路（短视） | 容易（macro 决策） |

**实务**：学术 R2R 离散 benchmark → high-level；VLN-CE / 真机 → 多数论文用 **waypoint predictor 折中**（high-level 决策 + low-level 跟踪）；端到端 VLM（NaVid / VLN-R1）→ 直接 low-level，靠大模型容量克服短视。

**易错**：high-level action ≠ "真机可部署"——你仍需 PointNav 控制器把"去那个 waypoint"落到 cmd_vel。真正的 sim2real gap 在 PointNav，不在 VLN 上层决策。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2 (破例)</span> <b>Q31</b> · VLN-R1：RFT（reinforcement fine-tuning）解决了 SFT 的什么短板？</summary>

**答**：VLN-R1（Qi 等 2025，arXiv 2506.17221）针对 SFT 核心问题：**仅模仿专家无法处理 OOD**——没见过的环境就崩。

**RFT 改造**：SFT 阶段在 VLN-Ego 数据集（HK 大 + 上海 AI Lab）模仿专家；RFT 阶段用 **GRPO**（DeepSeek-R1 同款）微调，reward = 成功 + SPL；关键 trick **Time-Decayed Reward**——离目标近的步奖励越大。

**收益**：R2R-CE val-unseen SR：SFT 41% → RFT 49%（+8pt）；OOD 长指令鲁棒性提升；sim→real gap 缩小。RFT 有效原因：SFT 教模仿，RFT 教达成目标，目标导向 reward 更鲁棒。

**易错**：VLN-R1 仍**两阶段**——纯 RL from scratch 在 VLN 上不收敛（探索太难），SFT 必须先做。RFT ≠ RLHF，前者是任务奖励，后者是偏好建模。

**破例说明**：频次 2 但来源权威（HK 大 + 上海 AI Lab arXiv 高引），代表 2025"RL 微调 VLM" 大方向，主表保留。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q32</b> · panorama 全景输入 vs 单目 RGB 输入：哪种更接近真实部署？SmartWay / Fast-SmartWay 怎么取舍？</summary>

**答**：

| | **panorama**（12 视角 RGB-D） | **单目 RGB** |
|---|---|---|
| FoV | 360° 无盲区 | 60-90° 有盲区 |
| 硬件 | 12 路相机 / 全景相机（贵） | 普通单目 |
| 计算 | 视觉特征 ×12 | 单帧 |
| 主要工作 | HAMT / DUET / ETPNav | NaVid / Fast-SmartWay / VLN-R1 |

**SmartWay / Fast-SmartWay**（2025）：前者用 panorama 简化 waypoint 预测 + MLLM；后者 **panoramic-free**——只 3 张前向 RGB-D 做端到端 MLLM 决策，速度 +50%，真机友好。

**实务**：学术 SOTA 仍 panorama 为主（信息全），**工业趋势是单目 + 头部转动**。NaVid / Fast-SmartWay 是后路线代表。

**易错**：单目 ≠ 无空间感——agent 可转身获等效信息，只是多几步。真正取舍在"训练时是否给 panorama"，给了再单目蒸馏比直接单目训练好。

</details>



## §6 零样本 ObjectNav（10 题）

> 用户研究方向第二核心节。ObjectNav 任务定义简单（"找到 chair"）但**零样本**部署难——这是 2024-2026 VLM-driven 导航的主战场。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q33</b> · ObjectNav 任务定义？success / SPL 在 HM3D-Sem / MP3D 上的标准评测设定</summary>

**答**：

**任务**：给定一个**类别名**（如 `chair` / `bed` / `tv`），agent 从随机起点开始探索房屋，最终停在目标类物体附近视野内。

**成功条件**（Habitat ObjectNav Challenge 标准）：
- 停止时距任意目标类物体实例 < 1 m
- 目标类物体在 agent 视野内
- 不超过 500 步

**指标**：
- **Success**：满足上述三条件
- **SPL**：成功率 × 最短路径占比
- **SoftSPL**：放宽 SPL 阈值（部分成功）

**数据集**：
- **HM3D-Sem**（Yadav 2022，1000+ 真实扫描场景）：当前 Challenge 主数据集
- **MP3D**：90 场景，更老
- **Gibson**：早期室内数据集

**易错**：ObjectNav 的"目标"是**类别**而不是具体实例——房间里有 3 把椅子都行，只要走到任意一把视野内。这与 InstanceNav（找特定实例）和 ImageNav（找特定图像目标）不同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q34</b> · SemExp 的语义探索是怎么做的？为什么是 CVPR 2020 Habitat Challenge 冠军 baseline？</summary>

**答**：SemExp（Chaplot NeurIPS 2020）三模块：① **Semantic Map**：Mask R-CNN 检测 RGB-D 投到 2D BEV（每类一通道）；② **Goal-Oriented Semantic Policy**：在 semantic map 上**用 RL（PPO）学**选 long-term goal（哪个 frontier）；③ **Local Policy**：Fast Marching 走到 goal。

**为什么 work**：模块化（感知/探索/规划解耦）、类别先验（"chair 在客厅"等共现学到）、长期记忆（map 累积避免重搜）。

**Habitat 2020 Challenge 冠军**——SR 17.85%（当时 SOTA）；后续 PIRLNav 等在其上加 RL fine-tune。

**易错**：SemExp **不是 zero-shot**——Mask R-CNN 闭集，新类别（不在 COCO 80）不行；VLFM / OpenFMNav 才是"open-set 升级"。SemExp 强在 closed-set + 数据驱动，给足训练数据仍有竞争力。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q35</b> · ZSON 怎么实现 zero-shot？image-goal pretraining + text/image goal embedding 共享</summary>

**答**：ZSON（Majumdar NeurIPS 2022）核心：**zero-shot 来自 CLIP 跨模态对齐**。

**做法**：
1. **预训练**：在 ImageNav（给目标图像导航到拍照地）训 RL agent，目标用 CLIP image encoder 编码
2. **零样本部署到 ObjectNav**：把目标类别名走 CLIP text encoder，**text embedding 直接替代 image embedding** 喂给 agent
3. CLIP 图文对齐保证两者空间相近，agent 不微调可用

**巧妙之处**：避开了 ObjectNav 数据少的问题——ImageNav 合成便宜（任意场景任意位置都能合成图像目标），训了 100M+ 帧。

**结果**：HM3D ObjectNav zero-shot SR 25.5%（原文）；与 CoW 等不同 setting 数字不可直比，但 ZSON 是首个 image-goal 预训练 + CLIP 跨模态做到 SOTA 的方法。

**易错**：ZSON 的"zero-shot"指**对 ObjectNav**——其 ImageNav 训练仍是大规模监督，不是完全无监督。CLIP 对齐精度也限制 ZSON 上限。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q36</b> · L3MVN：LLM 怎么参与 frontier 选择？相比 SemExp 的核心收益？</summary>

**答**：L3MVN（Yu IROS 2023）思路：**用 LLM 作 frontier 选择的语义打分器**。

**做法**：① 维护 semantic map；② 在边界 cell 提取 frontier 候选；③ 对每个 frontier 拼接 prompt：`"I want to find {goal}. Around frontier-i, I see {nearby_objects}. Score 0-10."`；④ LLM 输出分数选最高分作 long-term goal。

**相比 SemExp 收益**：零样本类别支持（LLM 知道 "toaster 在 kitchen" 常识）；解释性强（每步有 LLM 解释）；数据效率（不需要海量 ObjectNav 训练轨迹）。

**结果**：HM3D zero-shot SR 35.2%（vs SemExp 22.7% same setting）。

**易错**：L3MVN 仍需**闭集语义分割**（COCO/LVIS）填 semantic map——只是 LLM 替代 goal policy，不是端到端零样本。完全开放词汇要等 VLFM/OpenFMNav。LLM 推理延迟（每步调一次）也是工程瓶颈。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q37</b> · VLFM 的 value map 是什么？怎么把 BLIP-2 cos similarity 灌到 frontier 选择？</summary>

**答**：VLFM（Yokoyama ICRA 2024，arXiv 2312.03275）三步：

1. **occupancy map**：深度反投影 → 2D 占用栅格 + frontier
2. **value map**：每帧 RGB 用 BLIP-2 算 `"a photo of {goal}"` cos similarity → 投到对应 BEV cell（前向 cone 内），多帧累积取 max
3. **frontier 选择**：选 value map 最大的 frontier 作 long-term goal，Fast Marching 走过去

**核心创新**：把 VLM 图文 similarity 当**密集的目标存在性 prior**——不是闭集类别，而是任意 prompt。完全开放词汇 + 空间记忆 + 零训练。

**结果**：HM3D zero-shot SR 52.5%——VLM-based zero-shot ObjectNav 2024 SOTA baseline。

**易错**：prompt 工程敏感——`"a photo of {goal}"` 比裸 `"{goal}"` 显著更好。value map 投影是前向 cone，假设目标在视野中央方向，侧向目标会丢。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q38</b> · OpenFMNav 相比 VLFM 的优势？open-set 怎么做？</summary>

**答**：OpenFMNav（Kuang NAACL 2024）针对 VLFM 两问题：① BLIP-2 query 粗粒度（全图 cos sim）不精确；② 自由形式指令（"找绿色马克杯"）效果差。

**改造**：① **PNP**（Prompt-based Navigation Planner）：LLM 解析指令拆出关键物体 + 属性；② **Open-Set Detector**：GLIP / Grounding DINO 替代 BLIP-2 cos sim 给 bbox；③ **VSSM**（Versatile Semantic Score Map）：聚合 open-set 检测分数到地图，支持 PNP 引导下的目标 / 中间物体打分。

**核心区别**：VLFM 图文整体 similarity 粒度粗；OpenFMNav 开放词汇 detection 粒度精 + 属性可控。结果：HM3D zero-shot SR ~54.9%（vs VLFM 52.5%，不同 setting 略有出入）；自由形式指令 +10pt。

**易错**：OpenFMNav 调用链长（LLM + Grounding DINO + LLM scoring）单步 1-3 s；VLFM BLIP-2 单步 300-500 ms 更接近实时。工程上 VLFM 仍是首选 baseline。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q39</b> · CoW (CLIP-on-Wheels)：为什么 zero-shot 简单方法能打过学习方法？</summary>

**答**：CoW（Gadre CVPR 2023，arXiv 2203.10421）极简方案：CLIP 定位（每帧 RGB 拆 patch + image-text similarity 找最像目标的 patch）+ 经典 FBE 探索（1997 算法）+ depth 反投影；**无任何训练**。

**为什么打过学习方法**：当时学习方法（SemExp、ZER）**强依赖训练分布**，HM3D 训练在 Gibson 测就崩；CLIP 4 亿图文对预训练泛化强；经典 FBE 简单可靠不受数据漂移影响。

**结果**：HM3D ObjectNav zero-shot SR 12-25%（不同 CLIP backbone），在零样本设定下优于同期 ZSON / SemExp。

**深层意义**：CoW 揭示了"**学习方法在 OOD 上 vs zero-shot 基线**"——2023 之后 zero-shot ObjectNav 主线开端。

**易错**：CoW 在**复杂指令**（自由形式）上不行——只能吃单类别名，属性 / 关系 / 空间介词都 handle 不了。VLFM / OpenFMNav 是 CoW 思路的 "VLM 升级版"。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q40</b> · Frontier + LLM 评分 vs 端到端 RL：哪种更适合零样本泛化？训练-推理代价对比</summary>

**答**：

| | **Frontier + LLM 评分**（L3MVN / VLFM 系） | **端到端 RL**（DD-PPO / Habitat-Web） |
|---|---|---|
| 训练 | 零训练 | 1B+ steps RL（数千 GPU 小时） |
| 推理延迟 | 每步 0.3-3 s（VLM call） | 单帧 < 30 ms |
| 零样本类别 | ✅ 任意类别 | ❌ 只能训练集见过 |
| 跨场景泛化 | ✅ 强（VLM 内置常识） | ⚠️ 弱（OOD 易崩） |
| 真机迁移 | ✅ 不依赖训练数据分布 | ❌ sim-to-real gap 大 |
| 长程效率 | ⚠️ FBE 探索次优 | ✅ RL 能学到高效路径 |

**实务结论**：
- **零样本必须用** frontier + LLM 路线（VLFM / OpenFMNav）
- **闭集 + 大量训练数据** RL 仍最优（HM3D 闭集 SR 70%+）
- **混合**：PIRLNav 先 IL 后 RL 是闭集 SOTA；VLFM + RL 微调是 2025 新方向

**易错**：不是"zero-shot 一定比 RL 强"——闭集设定下端到端 RL 仍 dominate；只是工业部署需要 zero-shot，所以前者更受关注。延迟敏感场景（如人形实时反应）frontier+LLM 路线仍有挑战。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q41</b> · PIRLNav 的 IL+RL 两阶段：IL warm-up 解决什么？RL fine-tune 又解决什么？</summary>

**答**：PIRLNav（Ramrakhya CVPR 2023）针对纯 RL 在 ObjectNav 上不收敛——**reward 稀疏**（只在终点给）+ 探索空间大。

**两阶段**：① **IL warm-up**：在 70k 人类演示上 behavior cloning，先学"基础行走 + 看到物体停下"；② **RL fine-tune**：PPO 在 IL 检查点继续 1B steps，reward = 成功 + 接近 + 探索。

**为什么必须两阶段**：纯 IL 模仿专家但 OOD 崩；纯 RL reward 太稀疏（500 步 episode 通常 0 reward）；IL 提供可执行 baseline，RL 扩张优化效率。

**结果**：HM3D **closed-set** ObjectNav SR 65%（IL+RL 工作 SOTA）。

**易错**：PIRLNav 是**闭集** SOTA，不是 zero-shot——和 VLFM / OpenFMNav 不在同一赛道。两路"SR 数字"不可直比（PIRLNav 训练见过 chair，VLFM 没见过）。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q42</b> · 零样本 ObjectNav 在真实机器人上的核心瓶颈是什么？sim-to-real 最大 gap 在哪？</summary>

**答**：三大瓶颈：① **深度感知**（仿真 GT depth vs 真实 RealSense 噪声 / 反光 / 黑色物体丢深度 → VLFM value map 投影直接受影响）；② **场景结构**（HM3D 真实扫描但**无动态物体**，真机场景常变，frontier 探索逻辑要重设计）；③ **语义检测**（Grounding DINO 仿真 mAP ~50，真机相机畸变/光照下掉 10-15pt，错传到上层选 frontier）。

**次级**：控制器 sim2real（PointNav 仿真 95% → 真机 70-80%）、指令歧义（"那个红色杯子"agent 不知道哪个房间）、长程可靠性（500 步真机 5-10 分钟，传感器漂移 / 电量 / 网络）。

**缓解**：VLM 真机相机微调（域适应）、保守 frontier 策略、failure 检测 + recover。

**易错**：不要以为"零样本"就 sim2real 友好——VLM 在网络图像训练，**与真机相机仍有 gap**；零样本只是不需要场景级训练，但 VLM 训练分布仍重要。

</details>



## §7 Embodied VLM（10 题）

> 用户研究方向第三核心节。VLM 是 zero-shot 导航 / 操作的"通用感知器官"——本节聚焦怎么在 embodied pipeline 里**正确调用 VLM**，避免幻觉、延迟、误检三大坑。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q43</b> · CLIP 和 SigLIP 在 embodied 里的角色？为什么 SigLIP 在小 batch 下更好？</summary>

**答**：

**CLIP**（Radford OpenAI 2021）：图文对比预训练 + softmax；4 亿图文对（OpenCLIP/LAION 系 2B+）；图文 encoder 独立。

**SigLIP**（Zhai Google 2023）：把 softmax 换成 **sigmoid 二分类**——每对图文独立判断 match，不再相互归一化。

**为什么 SigLIP 小 batch 更好**：CLIP softmax 的 partition function 在 batch 内归一化，需超大 batch（32k+）；SigLIP 每对独立计算，batch 影响小，小 batch 下 ImageNet 0-shot 高 CLIP 5-10pt。

**embodied 应用**：大批量预训练 → CLIP / OpenCLIP；实时 query / 端侧 → SigLIP（推理快、batch=1 稳）；多语长文 → SigLIP-So400m / SigLIP2。

**易错**：SigLIP **不是 CLIP 升级版**——大 batch 充足时 CLIP 仍持平；embodied 圈用 CLIP 多是因生态成熟，不是 SigLIP 更弱。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q44</b> · Grounding DINO 是怎么做开放词汇检测的？vs OWL-ViT 区别？</summary>

**答**：

**Grounding DINO**（Liu IDEA 2023，arXiv 2303.05499）：基于 DINO 检测器（Zhang ICLR 2023，与 Caron 自监督 DINO 同名不同），文本 prompt 经 BERT 编码后在 backbone + transformer 多层与图像做 **deep fusion**。

**OWL-ViT**（Minderer Google 2022）：ViT + DETR 头，**晚期融合**——CLIP 预训练 + detection 微调，编码器独立最后点积；OWL-v2 加 self-training 提升。

| | Grounding DINO | OWL-ViT/v2 |
|---|---|---|
| 融合 | 多层 cross-attention | 最后点积 |
| LVIS mAP | ~40 | ~36 |
| 速度 | 150-300 ms | 50-100 ms |

**实务**：高精度离线 → GD 1.5；实时机器人 → OWL-v2 / GD-Tiny。

**易错**：开放词汇检测 ≠ zero-shot 分类——前者要 bbox。Grounding DINO 也不是"任意 prompt 都行"，长 phrase / 抽象概念（"舒适的椅子"）效果差。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q45</b> · SAM 2 相比 SAM v1 的核心进步？视频 / 流式怎么做？</summary>

**答**：

**SAM v1**（Kirillov Meta 2023）：单图像分割，prompt（点/框/mask）→ mask；1B+ mask 训练。

**SAM 2**（Ravi Meta 2024，SA-V 数据集 5 万视频 + 60 万 masklets）加 **memory bank + streaming attention**：
- 前帧 mask + features 入 memory，下一帧 decoder 通过 cross-attention 访问 → **时序一致**
- 第 1 帧给 prompt，后续帧自动 propagate；单帧 ~30 ms（v1 ~100 ms）

**embodied 应用**：物体跟踪（teleop / VLN 长程记忆）、地面分割（locomotion controller 输入）。

**易错**：SAM 2 仍**需要 prompt**，不主动检测——要 open-vocab 自动分割得配 Grounding DINO（Grounded-SAM 2 组合）。memory bank 是隐式特征 cache，不是显式物体级 ID，多个相似物体易 mask 混淆。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q46</b> · VLMaps 是怎么构建的？landmark indexing 怎么做？</summary>

**答**：VLMaps（Huang ICRA 2023）核心：**把 LSeg/CLIP 的 pixel-level 特征反投影到 BEV 体素地图**。

**构建步骤**：
1. agent 走过环境采 RGB-D
2. 每帧 RGB 过 **LSeg**（CLIP 像素级版本）得 pixel-level 特征图
3. depth + camera pose 反投影 3D → 压到 BEV grid
4. 多帧累积：同一 cell 多帧特征加权平均（按视角置信度）

**landmark indexing**：给文本 query（"go to the sofa"），CLIP 文本编码 → BEV map 每个 cell 算 cos similarity → top-K cell 作 landmark，规划器走到 max 处。

**强项**：零样本开放词汇 + 空间结构化（BEV 可直接规划） + 跨任务通用（landmark / region / spatial relation 都能查）。

**易错**：VLMaps 特征聚合是**简单平均**，不分前/背景——天花板地板会污染同一 cell；ConceptFusion / OpenScene 用 mask-aware 聚合改进。LSeg 训练数据小，long-tail 物体仍弱。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q47</b> · ConceptFusion：open-set 3D map 怎么把多模态特征融合进去？</summary>

**答**：ConceptFusion（Jatavallabhula RSS 2023）思路：**多基础模型（CLIP / DINO / AudioCLIP）特征统一融合到 3D map**。

**做法**：走环境采 RGB-D → 每帧用 Mask2Former 分割，每个 instance 用 CLIP 抽特征 + 全图 CLIP global feature → pixel 特征 = global + per-mask 加权（防背景污染）→ 反投影 3D 体素，多帧累积。

**multimodal 查询**：text / image / audio / 3D click 全部映射到同一特征空间做 cos similarity。

**对比 VLMaps**：VLMaps BEV 2.5D，只 CLIP 文本；ConceptFusion 完整 3D + 多模态查询 + mask-aware 聚合。

**易错**：ConceptFusion **离线建图**——非实时增量，建图后查询很快但建图本身需要走全场景。multimodal 主要是**查询模态多样**，存储仍是 CLIP 特征；audio query 先转 CLIP-aligned vector 再查。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q48</b> · Grounded-SAM (DINO + SAM) 在 embodied pipeline 里典型用法？</summary>

**答**：**Grounded-SAM**（IDEA 2023）= Grounding DINO 给 bbox + SAM 给 mask 的级联：

```
RGB → Grounding DINO("the red cup") → bbox →
SAM (prompt=bbox) → pixel-precise mask
```

**embodied 典型用法**：
1. **抓取目标定位**：text → mask → mask 内点云均值作抓取中心
2. **导航语义地图**：每帧 detect + segment → 投影到 BEV semantic map
3. **遥操作辅助**：人指 "拿那个杯子"，自动框选 + 提示
4. **数据标注**：teleop 数据后处理时自动给物体 mask

**vs Grounded-SAM 2**：SAM v1 → SAM 2 升级后视频流式 mask 跟踪，对**连续操作 / VLN**更友好。

**易错**：Grounded-SAM 是**两次模型调用**，单帧延迟 300-800 ms（GPU），实时场景需要 model distillation 或 cache。Grounding DINO 给的 bbox 不准时，SAM 的 mask 也跟着错——上下游都有失效模式。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q49</b> · VLM 在 zero-shot nav 里的幻觉问题怎么处理？常见缓解策略</summary>

**答**：VLM 幻觉两类典型场景：① 目标不存在但 VLM 说存在（把茶几误判为 sofa → frontier 偏向假目标）；② 指令偏差（"绿色马克杯"被 VLM 关注"绿色"忽略"马克杯"找到绿色背包）。

**缓解策略**：
- **多帧验证**：同物体 K 帧一致检测才确认
- **几何先验**：sofa 高度 0.4-0.8 m，与 depth-derived 高度矛盾就拒
- **多 prompt ensemble**：`"a photo of a sofa"` / `"sofa in a living room"` 取平均
- **score 阈值**：cos sim < 0.25 直接不算
- **VLM 二次确认**：找到候选后让 GPT-4V "is this {goal}? yes/no"
- **闭环 verification**：靠近后再确认

**实务**：VLFM / OpenFMNav 默认用前 2-3 个策略；真机部署加入 VLM 二次确认。

**易错**：幻觉**不能完全消除**——VLM 内在限制；只能降低概率到可接受水平。调试时优先看 false positive（看到不存在的目标）而非 false negative，因 FP 直接导致任务失败。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q50</b> · VLM 推理延迟 vs 准确率 trade-off：cloud-edge 协同 / cache / batched query 怎么取舍？</summary>

**答**：**延迟**：Grounding DINO 单帧 150-300 ms（A100）；BLIP-2 / SigLIP 50-100 ms；GPT-4V API 1-3 s。

**策略**：
| 策略 | 收益 | 代价 |
|---|---|---|
| 小模型蒸馏 | 5-10× 速度 | 5-15pt 精度损失 |
| cloud-edge 协同 | 准 + 低端侧算力 | 网络抖动 / 隐私 |
| frame skip | 线性加速 | 错过快速变化 |
| VLM cache | 同场景同 query 重用 | 动态场景失效 |
| batched query | 多 frontier 一次过 | 跨帧不行 |
| Cascade 早退 | 简单帧用小模型 | 复杂度判断难 |

**实务架构**：高频感知（30 Hz）用蒸馏小 VLM；低频规划（1 Hz）用大 VLM / LLM；二者通过 semantic map 解耦。

**易错**：不是"VLM 越大越准"——小 VLM + 任务微调反而更稳。**真实瓶颈往往不是 VLM**：相机 IO（USB 30 FPS）、网络（云端 200+ ms）、控制频率上限。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q51</b> · prompt 工程在 zero-shot nav 中的重要性？chain-of-thought 真的有用吗？</summary>

**答**：**prompt 工程**对 zero-shot nav 收益**显著**：`"a photo of a {goal}"` 比裸 `"{goal}"` 在 VLFM 上 +5-10pt SR；`"the goal is in {room_type}"`（加 room context）+3-5pt；多 prompt ensemble +2-4pt。

**CoT 在导航里**：
- **有用**：高层任务规划——"bedroom 先看床、kitchen 先看橱柜"；NaVILA / NaviLLM 都用 CoT-style prompt
- **无用 / 有害**：低层动作选择——"前进 0.25 m 还是 0.5 m" 用 CoT 反而引入噪声 + 延迟
- **VLN-R1 / Nav-R1 (2025)**：把 CoT 蒸馏到 SFT 阶段，推理时无需显式 CoT 效果保留

**关键判断**：CoT 用于**语义推理**有效，用于**几何 / 控制**无效——LLM 不擅长空间数值推理。

**易错**：CoT 不是免费午餐——token 数翻倍 → 延迟翻倍；CoT 输出可能"圆滑而错"。生产代码 CoT 通常**离线评估**保留，在线折叠成 SFT 数据。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q52</b> · open-vocabulary 检测 vs 闭集检测在导航里：什么时候必须开放词汇？</summary>

**答**：

| | **闭集**（COCO 80 / LVIS 1200） | **open-vocabulary** |
|---|---|---|
| 类别 | 训练集固定 | 任意文本 prompt |
| 精度 | 高（per-class fine-tuned） | 略低 |
| 速度 | 快（YOLO 5-30 ms） | 慢（150-500 ms） |
| 训练 | 一次性 | 不需要 |

**必须开放词汇**：REVERIE/SOON 含未知物体（"我的眼镜"）；自由形式 ObjectNav；长 tail 操作（古董花瓶）。

**闭集够用**：家庭服务（类别预定义）；实时性硬约束（自驾 30 Hz）；嵌入式资源受限。

**实务混合**：高频通道闭集快速检测（YOLO），低频通道（1-2 Hz）开放词汇兜底新类别——zero-shot ObjectNav 常见工程方案。

**易错**：开放词汇**不等于"无所不知"**——Grounding DINO 在 long tail / 抽象概念上仍弱。"开放词汇 → 解决一切"是误区；闭集 + 大量训练在已知任务上仍最强。

</details>



## §8 端到端 VLN-VLA 与前沿判断（5 题）

> 2026 视角的前沿题——多是开放判断题，没有标准答案；面试官关心你能否**有依据地表达立场**。

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q53</b> · Habitat vs AI2-THOR vs iGibson 三大仿真器的区别？为什么 ObjectNav 主流用 Habitat？</summary>

**答**：

| | **Habitat 2/3** | **AI2-THOR** | **iGibson 2.0** |
|---|---|---|---|
| 场景 | MP3D / HM3D 真实扫描 | Unity 交互式资产场景（ProcTHOR 是程序化扩展） | 真实 + 程序混合 |
| 物理 / 交互 | 简化（Habitat 3 加 humanoid） | 符号化交互强 | PyBullet 完整物理 |
| 渲染速度 | **10000+ FPS** | 100-1000 FPS | 10-100 FPS |
| 主用 | ObjectNav / VLN-CE | 任务规划 / 长程交互 | 操作 + 导航联合 |

**为什么 ObjectNav 主流用 Habitat**：① 渲染速度（RL 需 1B+ frames，唯一可大规模）；② HM3D 1000+ 真实房屋扫描，规模/真实/标注最优；③ Habitat Challenge 每年办，CVPR Embodied AI 默认平台。

**易错**：Habitat 3 加 humanoid + 物理交互，已追赶 AI2-THOR；别按 2020 旧知识答"Habitat 不能交互"。NeurIPS 2024 / 2025 Embodied AI 主流仍 Habitat-Lab。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q54</b> · VLN 在真实机器人 vs 仿真的最大 gap 在哪？哪些 trick 能拉近？</summary>

**答**：四大 gap：① **视觉**（仿真渲染 vs 真实相机：曝光、白平衡、运动模糊、畸变）；② **空间结构**（仿真静态整洁，真实动态：人 / 宠物 / 物品移位）；③ **底层控制**（PointNav 仿真完美，真机有打滑、不平、轮速误差）；④ **指令分布**（R2R 指令风格统一，真实用户随意：停顿、口误、省略）。

**拉近 trick**：域随机化（光照/纹理/噪声）、真机 demo 微调 VLM、保守 controller（容差放宽到 1.5-2 m）、failure recovery、闭环 VLM verification、panorama → 单目蒸馏。

**实务**：2024-2025 真机 VLN 论文（NaVid 实机、Mobile-ALOHA + VLN）仿真 → 真机 SR 掉 20-40pt 是常态；OK-Robot / VoxPoser 做了更激进 zero-shot 真机方案。

**易错**：sim-to-real 不只是"加噪声重训"——要从**任务定义**层面缩 gap（成功阈值放宽、step 上限拉长）；不要假设"仿真 SR 60% 真机也 60%"。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q55</b> · VLMnav / Nav-CoT / RoboFlamingo：端到端 VLN-VLA 真的可行吗？2026 年的判断</summary>

**答**：**端到端 VLN-VLA**：图像 + 指令 → 大 VLM → 动作或路径，不用 frontier / waypoint / semantic map 中间表征。

**代表**：VLMnav (2024) GPT-4V zero-shot；Nav-CoT / Nav-R1 (2025) CoT 推理 + 动作；RoboFlamingo (ICLR 2024) OpenFlamingo 导航 + 操作；NaVid / Uni-NaVid 视频 VLM 端到端 VLN-CE。

**2026 判断**：
- **学术**：端到端是趋势，VLN-R1 / Uni-NaVid 在 R2R-CE 上接近 panorama 方法
- **工程离工业部署还差**：延迟（VLM 0.5-3 s）、可解释性差（黑盒 debug 难）、错误归因难（感知错还是规划错？）
- **混合架构仍是 2026 主流**：VLM 出 high-level 决策 + 经典 controller 落地（如 NaVILA）

**易错**：不要把"端到端 = 更强"绑定——算力 + 数据充裕时端到端能逆袭模块化，但当前阶段模块化在真机部署 dominate。RT-2 / OpenVLA 是 manipulation 端到端好例子，VLN 还差几步。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2 (破例)</span> <b>Q56</b> · VLFM vs OpenFMNav 选哪个作 zero-shot baseline？</summary>

**答**：实务推荐 **VLFM** 作首选 baseline，OpenFMNav 在以下情况下用：

| 选择 | 理由 |
|---|---|
| **VLFM 优先** | ① 单 VLM 调用（BLIP-2），延迟稳定；② 代码极简，复现快；③ HM3D zero-shot SR 52.5% 已经够 benchmark |
| **OpenFMNav 优先** | ① 自由形式指令（属性 + 关系）；② 罕见类别需要开放词汇检测；③ 不在意延迟（OpenFMNav 单步 1-3 s） |

**复现成本**：VLFM 在 Habitat 上 1-2 周可复现 → 论文 baseline 大部分都引用 VLFM。

**易错**：不要在文章里写"我们的 baseline 是 VLFM"然后用 OpenFMNav 设置 evaluate ——两者**任务定义和数据 split 都需对齐**才能公平比较。HM3D ObjectNav 与 HM3D-Sem **不是同一个评测**——有些工作混用。

**破例说明**：本题频次 2 但来自论文 baseline 选择实务问题，权威工作（VLFM ICRA 2024 / OpenFMNav NAACL 2024）双独立来源，主表保留。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2 (破例)</span> <b>Q57</b> · 未来 2 年（2026-2028）导航方向的 3 个预测：world model 介入？多机器人协作？VLA 一统？</summary>

**答**：3 个有依据的预测：

1. **World Model 进入导航**：当前 frontier / VLM 是**反应式**；未来 agent "在心里走一遍"（类似 Dreamer / V-JEPA 2）。Genie / 1X 世界模型 / GR00T 正向导航延伸，2026-2027 大概率出现"导航 dreamer"。

2. **多机协作 / 人机协同**：单机导航接近天花板；下阶段是无人机 + 地面机器人协作探索、人机协作 VLN。

3. **VLA 一统**：OpenVLA / π0 / Helix / GR00T-N2 追求"一模型做导航 + 操作"。短期 (2026) 不会完全统一（长程 vs 精控差异大）；中期 (2027-2028) 可能基于 video-VLM + 真机数据出现强统一模型。

**反预测**：经典栈（Nav2 + 闭集检测）**不会消亡**，与 VLM-driven 长期共存。

**易错**：开放题没标答；面试**有依据 + 自洽** > 猜对，引用论文 / 数据支持立场显得专业。

**破例说明**：开放预测题面试常见（特别是顶级 lab / startup），频次 2 但代表整类题型，主表保留。

</details>



## §9 低频备选（3 题）

> 频次 1-2，但来源权威或前沿话题；备查不强求。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2 (低频)</span> <b>N1</b> · NICE-SLAM / Gaussian-SLAM 把 NeRF/GS 当地图表征：实务上是否取代 ORB-SLAM3？</summary>

**答**：**短期不会取代**。

**NICE-SLAM / Co-SLAM / SplaTAM**：在线 NeRF/GS-SLAM，能产生**稠密 + 可渲染**地图，对比 ORB-SLAM3 稀疏点云优势在重建。

**短板**：
- 鲁棒性：ORB-SLAM3 在快速运动 / 弱纹理 / 长隧道经过 7 年迭代，工业稳定；NeRF/GS-SLAM 在挑战场景仍易崩
- 实时：SplaTAM ~10 Hz；ORB-SLAM3 30+ Hz
- 资源：NeRF/GS-SLAM 显存 10+ GB；ORB-SLAM3 1-2 GB
- 回环 / 多地图：传统 SLAM 工具链成熟，NeRF/GS-SLAM 还在演进

**适用场景**：
- 离线建图 + 重建 → NeRF/GS-SLAM 优势
- 实时定位 + 控制 → ORB-SLAM3 / VINS-Fusion

**易错**：Gaussian-SLAM 不等于 ORB + Gaussian map ——它把 GS 当**前端 odometry**而不只是后端 map；这种端到端神经 SLAM 仍是研究阶段。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2 (低频)</span> <b>N2</b> · ETPNav（R2R-CE SOTA）：拓扑图 + 局部 metric 混合表征怎么落地？</summary>

**答**：ETPNav（An et al. TPAMI 2024，arXiv 2304.03047）核心：**evolving topological + local panorama predictor**。

**架构**：① 拓扑图在线构建——agent 走过节点 + 边，每节点存 panorama embedding；② waypoint predictor 在 panorama 上预测 12 个候选 waypoint；③ 跨模态决策：指令 + obs + topology → transformer → "去哪个 waypoint" 或 "回退"；④ 回溯：走错可回到老节点重选。

**为什么强**：拓扑记忆避免重复探索；回溯能力处理复杂指令；混合粒度（拓扑粗规划 + 局部 waypoint 精控）。

**结果**：R2R-CE val-unseen SR 57%——2024 上半年 panorama-based 方法 SOTA。

**易错**：拓扑图**在线生长**——非先验地图，新场景从空图开始。回溯机制带来推理开销，**单步可达 0.5 s**，真机部署需要权衡。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2 (低频)</span> <b>N3</b> · DROID-SLAM / DPVO 端到端神经 SLAM 当前的工业可用度？</summary>

**答**：**DROID-SLAM**（Teed NeurIPS 2021）：端到端学习的稠密 BA SLAM，光流 + 几何整体网络化。优势：弱纹理 / 重复纹理 / 长基线显著优于 ORB-SLAM3。短板：显存 10+ GB、推理 < 10 Hz。

**DPVO**（Teed 2023）：DROID-SLAM 轻量后继，单 GPU 实时 30+ Hz、显存 < 4 GB。

**工业可用度**：学术 benchmark（TUM RGB-D / EuRoC）DPVO 持平甚至超过 ORB-SLAM3；真机部署仍少见——主要瓶颈是**无回环检测**（DPVO 只是 VO 不是 SLAM）+ GPU 依赖；特定场景（白墙、玻璃）有不可替代优势。

**预测**：2026-2027 神经 SLAM 在 Jetson Orin 等嵌入式 GPU 普及后会扩大份额；ORB-SLAM3 在 IMU + 多传感器融合方向仍领先。

**易错**：DROID/DPVO 是**视觉 odometry**，回环检测要外挂传统 BoW / NetVLAD；现代工业方案通常**神经 VO + 传统 loop closure** 混合。

</details>

---

## §H 手撕代码（7 题）

> 感知 / 导航 / 视觉 backbone 手撕段——与上文"概念+答案"题区分开。本节每题给"考察点 / 实现 / 易错"——附 ≤30 行 Python 实现。

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×11</span> <b>✍ H01</b> · 手撕 IoU（两个矩形框）</summary>

**考察点**：交并比；`max(0,·)` 防负宽高；axis-aligned 与 rotated 求法差异。

**实现**：

```python
import numpy as np

def iou(a, b):
    # a, b: [x1, y1, x2, y2]，axis-aligned
    # 交集左上角取 max、右下角取 min
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    # 不相交时 x2<x1 或 y2<y1，用 max(0,·) clip 防负面积
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)  # +eps 防除零

def iou_batch(boxes_a, boxes_b):
    # boxes: [N, 4]、[M, 4]；返回 [N, M]，向量化
    a, b = boxes_a[:, None], boxes_b[None]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-6)
```

**易错**：负宽/高没 clip 致 IoU>1；除零；边界半开/闭区间约定（+1 偏移）；旋转框直接套此式错（应 polygon 求交）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×11</span> <b>✍ H02</b> · 手撕 NMS（非极大值抑制）</summary>

**考察点**：排序 + 贪心 + IoU 过滤；soft-NMS 按 IoU 衰减而非删；多类应分类做。

**实现**：

```python
import numpy as np

def nms(boxes, scores, iou_thresh=0.5):
    # boxes: [N, 4] (x1,y1,x2,y2)；scores: [N]
    order = scores.argsort()[::-1]  # 按 score 降序索引
    keep = []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 当前框与剩余框做向量化 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        # 保留 iou <= thresh 的；+1 因为 order[0] 已剔除
        order = order[1:][iou <= iou_thresh]
    return keep
```

**易错**：未按类别分别 NMS（不同类不该互抑）；忘 score thresh 预过滤 → N 大 TLE；`iou_thresh` 取 1.0 等于不抑制。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>✍ H03</b> · 手撕 Conv2d 输出维度 / 感受野</summary>

**考察点**：`out=⌊(in+2P-D·(K-1)-1)/S⌋+1`；感受野是累加而非累乘；pooling 层也算 stride。

**实现**：

```python
def conv_out_dim(in_size, kernel, stride=1, padding=0, dilation=1):
    # PyTorch 标准公式（与 nn.Conv2d 一致）
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

def deconv_out_dim(in_size, kernel, stride, padding=0, out_pad=0, dilation=1):
    # 转置卷积输出尺寸
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + out_pad + 1

def receptive_field(layers):
    # layers: [(kernel, stride), ...] 顺序排列；pooling 也按 (k,s) 计入
    rf, jump = 1, 1
    for k, s in layers:
        # RF_{l} = RF_{l-1} + (k-1)·∏_{i<l} s_i；累加而非累乘
        rf += (k - 1) * jump
        jump *= s
    return rf
```

**易错**：忘 `dilation`；以为感受野累乘（实是累加）；忽略 pooling 层 stride 贡献；same padding 在 stride>1 时含义模糊。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>✍ H04</b> · 手撕 BFS / DFS 在 grid 上找最短路径</summary>

**考察点**：BFS 必给最短（无权图）；visited 入队即标（非出队）；DFS 求所有路径不能求最短。

**实现**：

```python
from collections import deque

def bfs_shortest(grid, start, goal):
    # grid: 2D list；0 通行、1 障碍；返回最短步数或 -1
    R, C = len(grid), len(grid[0])
    if grid[start[0]][start[1]] or grid[goal[0]][goal[1]]:
        return -1
    visited = [[False] * C for _ in range(R)]
    q = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True  # 入队即标，防重复入队 TLE
    dx, dy = (-1, 1, 0, 0), (0, 0, -1, 1)  # 四邻居
    while q:
        x, y, d = q.popleft()
        if (x, y) == goal:
            return d
        for k in range(4):
            nx, ny = x + dx[k], y + dy[k]
            # 边界 + 障碍 + visited 三道闸
            if 0 <= nx < R and 0 <= ny < C and not visited[nx][ny] and not grid[nx][ny]:
                visited[nx][ny] = True
                q.append((nx, ny, d + 1))
    return -1
```

**易错**：visited 出队才标 → 重复入队 TLE；用 DFS 求最短（应 BFS）；八邻居斜走代价应 √2 而非 1。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>✍ H05</b> · 手撕 A* / Dijkstra 路径规划</summary>

**考察点**：Dijkstra h=0、A* `f=g+h`；h 必须 admissible（≤真实代价）保证最优；用 `g_best` 替代 closed set。

**实现**：

```python
import heapq

def heuristic(p, goal):
    # 4 邻居用 Manhattan；8 邻居/自由方向用 Euclidean；都 admissible
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

def astar(grid, start, goal):
    R, C = len(grid), len(grid[0])
    g_best = {start: 0}
    pq = [(heuristic(start, goal), 0, start)]  # (f, g, pos)
    dx, dy = (-1, 1, 0, 0), (0, 0, -1, 1)
    while pq:
        f, g, p = heapq.heappop(pq)
        if p == goal:
            return g
        if g > g_best.get(p, float('inf')):  # 懒删除：过期条目跳过
            continue
        for k in range(4):
            nx, ny = p[0] + dx[k], p[1] + dy[k]
            if not (0 <= nx < R and 0 <= ny < C) or grid[nx][ny]:
                continue
            ng = g + 1
            np_ = (nx, ny)
            # 仅当新 g 更小才入队（替代 closed set，更简洁）
            if ng < g_best.get(np_, float('inf')):
                g_best[np_] = ng
                heapq.heappush(pq, (ng + heuristic(np_, goal), ng, np_))
    return -1
```

**易错**：`h` 不 admissible（高估）→ 解不最优；按 `g` 入堆而非 `f`；同节点重复入队忘懒删除致路径错。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>✍ H06</b> · 手撕 K-Means 聚类</summary>

**考察点**：E-step 分配 + M-step 更新；K-Means++ 初始化远胜随机；空簇要重选远点。

**实现**：

```python
import numpy as np

def kmeans_pp_init(X, k, rng):
    # K-Means++：第一个中心随机；后续按距已选中心最近距离的平方加权采样
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum()
        centers.append(X[rng.choice(n, p=probs)])
    return np.array(centers)

def kmeans(X, k, max_iter=100, tol=1e-4, rng=None):
    rng = rng or np.random.default_rng(0)
    centers = kmeans_pp_init(X, k, rng)
    for _ in range(max_iter):
        # E-step：每点找最近中心（argmin |x - μ_k|²）
        dist = ((X[:, None] - centers[None]) ** 2).sum(-1)  # [N, k]
        labels = dist.argmin(axis=1)
        # M-step：更新中心；空簇用全局最远点替换
        new_centers = np.array([
            X[labels == j].mean(0) if (labels == j).any()
            else X[dist.min(1).argmax()]  # 空簇 → 取目前最难拟合的点
            for j in range(k)
        ])
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, labels
```

**易错**：用随机初始化局部最优；空簇未处理；用 `==` 比 `float` 距离判收敛；k 选错（用肘部法 / silhouette）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>✍ H07</b> · 手撕 ViT patch embedding</summary>

**考察点**：用 `Conv2d` 一步切+映射；patch 大小决定序列长度 `N=(H/p)·(W/p)`；CLS token + learnable PE。

**实现**：

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, in_c=3, dim=768):
        super().__init__()
        assert img_size % patch == 0, "img_size 必须能被 patch 整除"
        self.n_patch = (img_size // patch) ** 2
        # Conv2d 同时实现切 patch + 线性投影：kernel=stride=patch
        self.proj = nn.Conv2d(in_c, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # PE 长度 = n_patch+1（含 CLS）；换分辨率需 interpolate
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patch + 1, dim))

    def forward(self, x):  # x: [B, 3, H, W]
        B = x.size(0)
        # Conv 输出 [B, dim, H/p, W/p] → 展平为 [B, N, dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # 拼 CLS token 到序列首
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed
```

**易错**：H/W 不被 patch 整除（要 pad/crop）；忘加 CLS token；PE 长度与 patch 数不一致致换分辨率失败。

</details>

