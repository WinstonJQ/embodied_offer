# 卷四 · 世界模型 / Sim2Real · 高频面试题

> 中文具身智能秋招高频面试题库 · **第四卷**
> 题源：牛客 / 知乎 / CSDN / GitHub 公开面经 / arxiv 论文讨论 / 具身智能算法岗面试汇编
> 同义题合并后 **31 题**（主表 31 题：频次 ≥3 + Q31 破例入选）

**难度分布**：L1（必会） **9** · L2（进阶） **18** · L3（顶级难度） **4**

**使用方式**：题目默认折叠，点开看答案。建议按 **L1 → L2 → L3** 顺序刷；同级内按频次从高到低。手机端原生支持。

---

## §1 世界模型基础（5 题）

> 从 PlaNet 到 Dreamer：RSSM 核心 + latent imagination + model-based RL 的 sample efficiency 优势。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q01</b> · RSSM（循环状态空间模型）是什么？Dreamer 系列如何用它做"想象训练"？</summary>

**答**：RSSM（Recurrent State-Space Model，Hafner 2019 PlaNet）包含两条路径：
- **确定性路径** $h_t$（GRU 递归记忆）：保留长序列的历史信息
- **随机路径** $z_t$（随机隐变量：V1 用 Gaussian，V2/V3 用 Categorical 离散）：表达单步不确定性

两路拼接得到 latent state $(h_t, z_t)$。在训练时，$z_t$ 从观测中后验推断；在"想象"时，$z_t$ 从 RSSM 的先验中直接采样，无需真实环境交互。

**Dreamer 的想象训练**：世界模型在 latent 空间中前向 rollout 若干步，actor-critic 从这些"想象轨迹"中通过 BPTT（截断梯度）更新策略——样本效率比 model-free 高 5-20 倍。

**易错**：RSSM 不是 VAE，它的 deterministic path（h_t）是必须的，使得模型能记忆历史；纯 VAE 缺少这条路径会导致长程记忆消失。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q02</b> · PlaNet 是什么？和 Dreamer 的核心区别？RSSM 中"确定性路径 + 随机路径"分别起什么作用？</summary>

**答**：**PlaNet**（Hafner et al., 2019）是第一个用 RSSM 从像素做规划的模型。策略是**基于模型的规划**（CEM/随机射击搜索），在 latent 空间内采样动作序列、评估价值，选最优序列的第一步执行。

**与 Dreamer 的区别**：PlaNet 不学独立 actor，每步都要在线规划（慢）。Dreamer 额外学了 actor-critic，规划由 actor 网络"内化"，推理时直接前向过 actor，无需每步优化。

**两路的作用**：
- **确定性路径 $h_t$**（GRU）：跨步传递历史，提供给随机路径作上下文
- **随机路径 $z_t$**（随机隐变量，防止后验崩塌）：建模单步观测的随机性/不确定性

**易错**：PlaNet 已经用 RSSM，但策略仍是 MPC 式在线规划，不是学习到的 actor；Dreamer 才是"梦中学 actor"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q03</b> · 什么是 latent imagination？Dreamer 如何在隐空间里学策略而不需要真实交互？</summary>

**答**：**Latent imagination**：在 RSSM 学好后，从任意 latent state 出发，纯在模型内部（latent 空间）前向模拟 H 步，得到"想象轨迹"。这些轨迹不需要渲染成图像、不需要真实物理引擎，只在低维隐向量间运算，极速高效。

**Dreamer 的策略学习步骤**：
1. 用真实数据训练 RSSM（重建损失 + KL 正则）
2. 从 replay buffer 中的已见 latent 出发，通过 RSSM 前向展开 H=15 步想象轨迹
3. actor（策略网络）用 BPTT 在这些想象轨迹上最大化折扣奖励
4. critic（价值网络）拟合回报，给 actor 提供低方差梯度

**优势**：真实数据采集量少（用来训 RSSM），策略迭代在"梦中"完成，可做数百次 gradient step 对应一次真实 rollout。

**易错**：Dreamer 的 imagination 不是视频生成（不解码到像素），而是在 latent 向量空间中的前向传播；解码器只用于 RSSM 训练的重建损失。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q04</b> · model-based RL（Dreamer 类）和 model-free RL（PPO/SAC 类）在 sample efficiency 上的对比？</summary>

**答**：
| | Model-Free（PPO/SAC） | Model-Based（Dreamer/TD-MPC） |
|---|---|---|
| 环境步数需求 | 高（百万级） | 低（十万级） |
| 算法复杂度 | 低 | 高（需训世界模型） |
| 计算成本/步 | 低 | 高（想象 rollout） |
| 真机部署适合度 | 低（样本贵） | 高（用数据想象学策略） |
| 模型偏差风险 | 无 | 有（world model error 累积） |

**实务选择**：
- 真机直接训练（无仿真）→ **model-based 首选**，1 小时内学四足行走（DayDreamer）
- 大规模仿真 + GPU 并行 → model-free（PPO）足够，parallelism 弥补了样本效率劣势
- 数据稀缺的精细操作 → model-based + offline RL 混合

**易错**：model-based 不等于更快收敛（wall-clock time），因为每步需要训世界模型并做想象 rollout；但**环境交互次数**（环境 step 数）确实减少。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q05</b> · MuZero 和 Dreamer 的核心区别？各自适合什么场景？</summary>

**答**：
| | Dreamer / PlaNet | MuZero |
|---|---|---|
| 观测重建 | **有**（解码器重建图像/状态） | **无**（decoder-free，隐式模型） |
| 规划方式 | 想象 rollout + actor-critic | MCTS（树搜索）+ 动力学网络 |
| 动作空间 | 连续为主 | 离散为主（board games / Atari） |
| 真机适用 | 强（连续控制，低采样数） | 弱（MCTS 开销大，实时性差） |
| 世界模型目标 | 最小化重建 + KL | 最小化 value/policy/reward 预测误差（价值等价原则） |

**关键差异**：MuZero 的世界模型不需要还原真实观测，只需要能做价值等价的预测（Schrittwieser 2020）；Dreamer 用重建损失保证世界模型学到有意义的 latent。

**适合场景**：MuZero → 离散决策、棋类、Atari；Dreamer → 连续控制、视觉机器人、仿真和真机。

**易错**：MuZero 不是"更先进的 Dreamer"，两者是不同范式；MuZero 的规划开销在实时连续控制中通常无法接受。

</details>

---

## §2 Dreamer 系列进化（5 题）

> DreamerV1 → V2 → V3 的三次跃升 + TD-MPC2 的 decoder-free 路线 + DayDreamer 真机验证。

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×7</span> <b>Q06</b> · DreamerV3 相比 DreamerV1/V2 做了哪些改进？symlog 和 KL balancing 分别解决什么问题？</summary>

**答**：
- **V1**（2020 ICLR）：RSSM + 连续 latent + actor-critic 想象训练，首次像素级机器人控制
- **V2**（2021 ICLR）：离散 latent（Categorical 分布）+ KL balancing（α=0.8 偏先验），Atari 超人类
- **V3**（2023 Hafner，Nature 2025）：symlog 归一化 + free bits + 固定超参，7 类 domain 无需调参

**symlog 解决什么**：奖励尺度跨任务差异巨大，直接学习梯度不稳定。symlog(x) = sign(x)·ln(|x|+1) 压缩大值、线性化小值，稳定 reward/value 预测。

**KL balancing（α=0.8）解决什么**：若 posterior 和 prior 同等更新，prior 容易学得太简单。α=0.8 使更多梯度流向 prior，令其主动贴近 posterior，提升动力学预测质量。

**易错**：free bits（KL 低于阈值不惩罚）是 V1 就有的技巧，不是 V3 新增；V3 的创新是 symlog + percentage normalization + 规模化固定超参。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q07</b> · DreamerV3 为什么能"一套超参跨 7 大 domain"？固定超参的背后是什么设计？</summary>

**答**：DreamerV3 的"固定超参"核心在于消除了**奖励尺度依赖**和**任务特定归一化**：

1. **symlog 统一奖励尺度**：不同任务奖励量级不一，symlog 压缩后 critic 和 actor 的学习率无需逐任务调整
2. **percentage-based value normalization**：用百分位数（5th / 95th）归一化 advantage，消除绝对数值的影响
3. **固定 KL free bits = 1 nat**：使 latent 既有足够信息量又不过度正则化，无需按任务调 β
4. **规模化鲁棒性**：模型够大（200M 参数），容量充足以应对不同 domain 的复杂性

**跨越的 7 类 domain**：Atari（离散）/ DMControl（连续）/ MineCraft（3D 探索）/ Crafter（多任务）/ DMLab（记忆） / BSuite（系统性测试）/ proprio state（低维）。

**易错**：固定超参不代表"不需要训练更多步"，不同任务环境步数仍有差异；固定的是**算法超参**，不是训练步数。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q08</b> · DayDreamer 实验说明了什么？世界模型是否能在真机上在线训练？</summary>

**答**：**DayDreamer**（Wu et al., CoRL 2022，UC Berkeley）是 Dreamer 的真机版本，关键结论：

- **四足行走从零学会**：Unitree A1 四足机器人从随机初始化出发，在**真机不重置**的情况下，约 **1 小时**内学会站立和行走（无需仿真、无需演示）
- **在线学习可行**：世界模型实时更新（边交互边训），策略在 latent imagination 中迭代，真机数据需求比 model-free 少 5-10×
- **真机 adaptation**：学会走后 10 分钟内适应扰动（推动/地形变化）

**意义**：证明世界模型在真机上在线训练是可行的，不是只适合仿真；为无需 sim2real 的真机模型学习路线提供了可行 baseline。

**限制**：四足行走任务相对简单（无精细操作），高维操作任务在真机上的实时世界模型学习仍是开放问题。

**易错**：DayDreamer 不是"不需要 sim2real"的方案，而是"不依赖仿真"的替代方向；两条路线（sim2real 和真机在线学习）各有适用场景。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q09</b> · TD-MPC2 的"隐式世界模型"是什么概念？它和 DreamerV3 的 RSSM 有什么区别？</summary>

**答**：**TD-MPC2**（Hansen et al., ICLR 2024）的隐式世界模型（implicit world model）是 **decoder-free** 设计：

| | DreamerV3 | TD-MPC2 |
|---|---|---|
| 世界模型类型 | 显式（explicit）：有解码器，重建观测 | 隐式（implicit）：无解码器，不还原观测 |
| 训练信号 | 重建损失 + KL + reward 预测 | TD 误差 + reward + consistency loss |
| 规划方式 | actor-critic 想象（BPTT） | MPC + CEM/MPPI（latent 空间规划） |
| 适用规模 | 较大模型（200M+） | 较小模型（1M~317M，多任务单权重） |
| 任务多样性 | 跨 7 大 domain | 104 任务，单套超参，包含操作和运动 |

**隐式的好处**：不需要重建观测，节省容量；用 SimNorm 归一化 latent，防止 latent 坍缩；decoder-free 使模型更轻量。

**易错**：TD-MPC2 的"world model"不学重建，所以无法"看"世界模型的想象；它的 latent 只对 value/reward 预测有意义，不是可解释的图像 latent。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q10</b> · 世界模型用于 planning（MCTS/CEM/MPPI）的思路？MuZero 怎么做 planning？</summary>

**答**：**世界模型 + Planning**：在 latent 空间中搜索最优动作序列，而无需真实执行：

- **CEM**：采样动作序列，前向展开评估价值，选 top-k 迭代更新分布（TD-MPC 采用）
- **MPPI**：重要性采样加权多条 rollout，适合连续控制
- **MCTS**（MuZero）：树搜索 + 展开/模拟/回传，适合离散任务

**MuZero 的 Planning**：① representation（obs → latent）② dynamics（latent + action → next latent + reward）③ prediction（latent → policy + value）。MCTS 在这三个网络构成的虚拟树中搜索，policy/value 指导方向，无需真实环境交互。

**易错**：CEM/MPPI 是无梯度采样优化；MCTS 开销大（每步数百次前向），在连续控制实时场景中通常太慢，不要认为 MuZero 也适合机器人控制。

</details>

---

## §3 Sim2Real 基础（6 题）

> reality gap 来源 + Domain Randomization + ADR + Digital Twin + 迁移方式选择。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q11</b> · Sim2Real 中的 reality gap 主要来源有哪些？如何系统性解决？</summary>

**答**：**Reality gap 的主要来源**：

1. **物理参数不精确**：摩擦系数、质量惯性张量、关节阻尼——仿真默认值与真机不符
2. **接触/碰撞模型误差**：真实接触面非点接触，仿真的 point-contact / spring-damper 模型过于简化
3. **执行器（actuator）模型误差**：电机动态、传动比、延迟、反向间隙（backlash）在仿真中通常被忽略
4. **传感器噪声差异**：IMU 漂移、相机曝光变化、深度传感器噪声模式
5. **视觉外观差异**：光照、纹理、材质反射率不同（渲染真实感不足）

**系统解决策略**：
- **物理 gap** → Domain Randomization（DR）或 System Identification
- **执行器 gap** → actuator delay 建模 + torque/PD 控制 matching
- **视觉 gap** → Photo-realistic rendering + 视觉 DR（颜色/纹理/光照随机化）
- **自适应** → Teacher-Student + online adaptation（RMA）

**易错**：并非所有 gap 都能靠 DR 覆盖；contact model 误差是"硬性"误差，DR 只是用分布包住，无法从根本消除接触力学的模拟误差。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q12</b> · Domain Randomization（域随机化）是什么？为什么能缩小 reality gap？</summary>

**答**：**Domain Randomization（DR）**：在仿真训练时，将物理参数（摩擦、质量、关节刚度）和视觉参数（纹理、光照、相机位置）在预设范围内**随机采样**，迫使策略对参数变化保持鲁棒。

**为什么有效**：若真实世界参数在训练时的随机化范围内（或接近），策略已见过该参数值，可直接 zero-shot 迁移。核心假设是"只要随机化范围足够宽，真实世界参数必然被覆盖"。

**两类 DR**：
- **物理 DR**：摩擦系数、关节阻尼、质量 → 应对动力学 gap
- **视觉 DR**：纹理贴图、光照强度、颜色扰动 → 应对渲染 gap（OpenAI dexterous hand 经典案例）

**关键超参**：随机化范围需要调——太窄覆盖不了真实，太宽任务太难导致训练不收敛。

**易错**：DR 不等于"随机化越大越好"；过大的随机范围会使任务变得无法学习（policy 一直在应对极端情况），需要 curriculum 逐步扩大范围。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q13</b> · Automatic Domain Randomization（ADR）是什么？比手动 DR 好在哪里？</summary>

**答**：**ADR**（Akkaya et al., 2019 OpenAI，Rubik's Cube 论文 arXiv 1910.07113）：自动调节 DR 参数分布范围的方法。核心机制：

1. 每个随机化参数都有一个当前分布（如摩擦 [0.5, 2.0]）
2. 根据策略在**边界条件**（分布上/下端采样）下的成功率动态调整：若成功率 > 上阈值，扩宽分布范围；若 < 下阈值，收窄
3. 分布自动渐进扩展，只要策略能跟上就持续变难

**比手动 DR 好的地方**：
- 无需逐参数手调范围，省去繁琐 hyperparameter tuning
- 自适应课程：跟策略学习进度对齐，不会"一步跳太难"导致训练崩溃
- 可扩展到几十个随机化维度，手动调 50+ 参数不现实

**局限**：ADR 假设成功率是好的反馈信号，稀疏奖励任务中难以判断"是否在进步"；另外计算开销比固定分布 DR 大。

**易错**：ADR 不是一种新的随机化类型，而是一种**自动调整 DR 参数范围**的调度策略。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q14</b> · Domain Adaptation 和 Domain Randomization 是什么关系？Sim2Real 中各自的应用场景？</summary>

**答**：

| | Domain Randomization（DR） | Domain Adaptation（DA） |
|---|---|---|
| 时机 | 训练期间 | 测试时（有少量真实数据） |
| 思路 | 扩大 source domain 覆盖 target | 对齐 source 和 target 的分布 |
| 真实数据需求 | 不需要真实数据 | 需要少量真实数据（labeled or not） |
| 典型方法 | 随机化仿真参数 | DANN、MMD、GAN-based alignment |
| 部署场景 | zero-shot 迁移 | 有一定 real world 收集预算 |

**具身中的组合使用**：
- 先 DR 训练 → zero-shot 迁移（效果 60-80%）
- 再用少量真实数据 fine-tune（DA）→ 效果提升到 90%+
- 或用 sim-to-real 感知模块（视觉特征对齐）+ DR 策略组合

**易错**：DA 不是"比 DR 更好的替代"，两者互补；没有真实数据时 DR 是唯一选择，有少量真实数据时 DA 是高效的补充。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q15</b> · Digital Twin 在 sim2real 中的作用？与传统 DR 方法有什么区别？</summary>

**答**：**Digital Twin（数字孪生）**：对特定真实场景/机器人进行精确建模（几何、动力学、外观），使仿真尽量还原真实个体，目标是"仿真精度高到不需要大范围 DR"。

**和 DR 的对比**：

| | Digital Twin | Domain Randomization |
|---|---|---|
| 策略 | "精确仿真" → 减小 gap | "宽泛仿真" → 覆盖 gap |
| 构建成本 | 高（3D 扫描、系统辨识） | 低（随机化参数设范围） |
| 泛化性 | 绑定特定环境 | 泛化到更多场景 |
| 适合场景 | 工厂固定工位、已知机器人型号 | 多样化部署场景 |

**前沿融合**：用 3D Gaussian Splatting 快速构建数字孪生（分钟级），与 RL 训练流程结合；ManiSkill3 的数字孪生 RL 接口仍在开发中（WIP），是研究前沿方向。

**易错**：Digital Twin 不是"替代 DR"的方案，在动态场景（人群、随机光照）中精确孪生不可能，仍需 DR 补充。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q16</b> · 仿真到真机的"zero-shot transfer"和"fine-tuning transfer"有何区别？哪种更常用？</summary>

**答**：
- **Zero-shot transfer**：仿真训练后**直接部署**到真机，无需真实数据；依赖 DR 或精确仿真覆盖真实分布。典型：ANYmal 四足行走（Lee 2020）、OpenAI dexterous hand（Andrychowicz 2020）。
- **Fine-tuning transfer**：仿真训练后用少量真实数据 fine-tune；真实环境数据几十到几千条即可显著提升成功率。典型：机械臂精细操作（夹夹手、拧瓶盖）。

**哪种更常用**：
- 运动控制（四足/双足步态）→ **zero-shot 更主流**，因为策略对精确接触力不敏感，DR 覆盖足够
- 灵巧操作（抓取/插针/翻转）→ **fine-tuning 更必要**，接触力精度要求高，仿真永远有残差

**实务建议**：先尝试 zero-shot，若成功率 < 70% 再收集少量真实数据 fine-tune；节省数据采集成本。

**易错**：fine-tuning 不等于从头在真机上训练（那是 online RL，成本高）；fine-tuning 的基础是仿真预训练，只需少量真实 step 微调。

</details>

---

## §4 Teacher-Student / 自适应方法（4 题）

> 特权观测 + RMA + Curriculum + System ID：把仿真优势最大化转化为真机能力。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q17</b> · Teacher-Student 框架在 sim2real 中怎么用？privileged observation 是什么？</summary>

**答**：**Teacher-Student 框架**：仿真中训练两个策略：
- **Teacher**（教师策略）：可访问仿真中的**特权观测**（privileged observations）——真实机器人无法获取的信息，如精确地形高度图、摩擦系数、外力值等
- **Student**（学生策略）：只能访问真机上的传感器数据（IMU、关节角、相机图像）

训练流程：
1. 仿真中训练 Teacher 达到高性能（有特权信息，容易）
2. Student 用 DAgger / BC 蒸馏 Teacher：给定相同状态，让 Student 模仿 Teacher 的动作
3. Student 部署到真机（无需特权信息）

**为什么有效**：Teacher 利用特权信息找到了好策略，Student 蒸馏保留了策略质量，真机只需要 Student 可获取的传感器——打通了"仿真特权"和"真机局限"之间的桥梁。

**易错**：Teacher-Student 不是 knowledge distillation 的一般版本；特权观测是仿真专属的，不是"从大模型蒸馏小模型"；若 Student 观测与 Teacher 动作之间信息量差距太大，蒸馏也会失败。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q18</b> · RMA（Rapid Motor Adaptation）的核心思路是什么？为什么比纯 DR 更鲁棒？</summary>

**答**：**RMA**（Kumar et al., RSS 2021）= Teacher-Student + 在线自适应模块，两阶段训练：

- **Phase 1（仿真）**：训练 base policy（输入 proprio + extrinsics 向量 $e$，$e$ 编码地形/摩擦等特权信息）+ extrinsics encoder
- **Phase 2**：冻结 base policy，训 adaptation module：用历史本体感觉序列（IMU/关节角）预测 $\hat{e} \approx e$

**真机部署**：adaptation module 实时估计 $\hat{e}$，base policy 用 $\hat{e}$ 适应未知环境，无需外感传感器。

**比纯 DR 更鲁棒**：DR 是"被动覆盖"，策略不知当前处于哪种环境；RMA 是"主动适应"，根据历史本体感觉动态估计当前参数，策略能"感知"到地形变化。

**易错**：adaptation module 只用 IMU/关节角（真机都有），不依赖外感；估计误差 $\hat{e} \ne e$ 由 module 的鲁棒性吸收，不要求完美匹配。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q19</b> · curriculum learning 在 sim2real 中怎么用？常见 curriculum 设计策略有哪些？</summary>

**答**：**Curriculum Learning（课程学习）**：从简单任务/环境逐步加难，让策略稳定收敛，避免直接在"全难度"下训练失败。

**在 Sim2Real 中的常见 Curriculum 策略**：

1. **地形 Curriculum**（运动控制）：从平地开始 → 加坡度 → 加台阶 → 加随机不规则地形。Isaac Lab 标配地形课程。
2. **随机化范围 Curriculum**（DR Curriculum）：DR 范围从窄到宽逐步扩展；ADR 本质上是自动化的 DR curriculum
3. **任务难度 Curriculum**（操作）：从接近成功位置开始 → 逐步增大初始距离；从固定目标 → 随机目标位置
4. **奖励塑造 Curriculum**（dense → sparse）：初期用稠密奖励引导，后期切换稀疏奖励增强泛化

**何时用**：策略在完整难度下无法启动时（cold start）；或 DR 范围过大导致训练崩溃时。

**易错**：Curriculum 不一定总能提升最终性能——有时直接暴力训 full random 效果一样甚至更好（任务简单时）；Curriculum 主要解决"训练启动难"问题，不是性能上界问题。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q20</b> · 什么是 system identification（系统辨识）？如何与 DR 结合？</summary>

**答**：**System Identification（系统辨识，SysID）**：用真实机器人数据（如施加已知力矩后测量关节角度轨迹）估计物理模型参数（质量、惯量、摩擦、阻尼），使仿真参数尽量接近真实值。

**与 DR 的结合（标准实践）**：
1. 用 SysID 找到"最优估计"参数 $\theta^*$（均值）
2. 以 $\theta^*$ 为中心，在 ±uncertainty 范围内做 DR
3. 这样 DR 的覆盖是**有根据的分布**，不是盲猜范围

**好处**：DR 范围精准，不会过宽（浪费训练容量）也不会过窄（漏掉真实参数）。

**常见 SysID 方法**：最小二乘辨识（线性系统）/ 贝叶斯辨识（给出后验分布，直接指导 DR 范围）/ RMA 风格的在线辨识。

**易错**：SysID 是"一次性标定"，真机磨损/环境变化后需要重新辨识；RMA 等在线自适应方法是 SysID 的动态替代，无需离线标定步骤。

</details>

---

## §5 仿真器选型与工程（6 题）

> Isaac Lab / MuJoCo / Genesis / ManiSkill：GPU 并行 + 接触模型 + 格式选择。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q21</b> · Isaac Gym 和 Isaac Lab 的关系？与 MuJoCo 在仿真精度/速度上的对比？</summary>

**答**：
- **Isaac Gym**（2021 NVIDIA）：第一款 GPU 加速 RL 仿真器，基于 PhysX，支持数千并行 env，已于 2024 年**正式 deprecated**，推荐迁移到 Isaac Lab
- **Isaac Lab**（2024 NVIDIA）：Isaac Gym 的官方继任，基于 Isaac Sim（Omniverse 平台）+ PhysX，提供模块化 RL 环境接口，支持 USD 场景格式，功能更完整（sensor 插件、物理材质 API）

| | Isaac Lab（PhysX） | MuJoCo（Newton/CG/PGS） |
|---|---|---|
| 速度（GPU 并行） | 极快（100K+ FPS，千级 env） | 较快（MJX GPU 后端，但单核慢） |
| 接触精度 | PhysX：速度优先，精度中等 | Newton/CG/PGS：接触精细，适合精细操作 |
| 开源 | Isaac Lab 开源，Isaac Sim 闭源依赖 | 完全开源 |
| 常用场景 | 大规模 RL 训练、四足/人形步态 | 控制研究、操作任务、学术基准 |

**易错**：Isaac Gym 已 deprecated，不要在新项目中使用；Isaac Lab 需要 Isaac Sim 作为底层，而 Isaac Sim 是 NVIDIA 闭源商业软件（学术免费）。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q22</b> · MuJoCo、PyBullet、SAPIEN、Genesis 的特点对比？各自适合什么场景？</summary>

**答**：

| 仿真器 | 优势 | 局限 | 适合场景 |
|---|---|---|---|
| **MuJoCo** | 接触精细（Newton/CG/PGS 求解器），学术标配，DMControl 基准 | 慢（CPU 主）；MJX 是 GPU 后端但较新 | 控制研究、精细操作、学术对比 |
| **PyBullet** | 开源免费，易上手，社区大 | 接触精度差，性能低，已逐渐式微 | 快速原型，教学 |
| **SAPIEN** | 关节/铰链精确，articulated 对象建模好 | 并行性有限，生态较小 | 多关节操作（抽屉、门）；ManiSkill 基础 |
| **Genesis** | 号称最快（43M FPS），多物理后端 | 成熟度低，纯物理比 ManiSkill 慢 3-10x（基准测试） | 超大规模 RL 探索，数据生成 |

**MJX（MuJoCo with JAX）**：MuJoCo 的 GPU 并行后端，2023 发布，接触精度保持，支持 JAX 自动微分，逐步成为精细操作研究的 GPU 方案。

**易错**：Genesis 宣称"最快"基于特定场景；在物理仿真精度可比的条件下，ManiSkill3 的 GPU 并行效率（3.5GB vs 14.1GB，128 env）比 Isaac Lab 更高效。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q23</b> · GPU 并行仿真（tensorized envs）的原理和关键优势？</summary>

**答**：**GPU 并行仿真**：将 N 个独立仿真环境的物理状态张量化，通过 GPU 的 SIMD（单指令多数据）并行架构同时计算 N 个环境的物理步。

**核心原理**：
- 所有 env 的物理状态（关节角、速度、接触力）打包为 [N, D] 的 GPU 张量
- 物理引擎（PhysX / JAX MuJoCo）在 GPU 上一次前向计算 N 个 env
- RL 的 policy 网络也在 GPU 上批量推断，无 CPU-GPU 数据传输瓶颈

**关键优势**：
- 吞吐量：Isaac Lab 单卡 4090 可跑 4096 个 ANYmal env，效果相当于 4096 块 CPU 核
- 数据去相关：并行 env 采样多样轨迹，减少 on-policy 算法的 variance
- 训练加速：PPO 在 4096 env 下 wall-clock 时间比 1 env 快近 100×

**注意**：GPU env 数量不能无限增大——内存受限（128 env 约 3.5 GB GPU 内存，ManiSkill3 测试）；N 太大时 batch too large 导致策略更新步过大。

**易错**：GPU 并行仿真不等于仿真精度更高，速度和精度是不同维度；PhysX 为了速度做了精度妥协，精细接触任务仍需 MuJoCo。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q24</b> · URDF / MJCF / USD 三种机器人描述格式的区别？各仿真器对它们的支持情况？</summary>

**答**：

| 格式 | 来源 | 核心特点 | 适用场景 |
|---|---|---|---|
| **URDF** | ROS | XML 树形链结构；不支持 loop closure；无物理材质细节 | ROS / PyBullet / 快速原型 |
| **MJCF** | MuJoCo | 功能丰富（接触参数、肌腱、equality 约束），支持闭链 | MuJoCo 精细仿真、控制研究 |
| **USD** | Pixar / NVIDIA | 场景层级管理，支持传感器/光照/动态物体，工业级扩展 | Isaac Sim / Isaac Lab / 大规模渲染 |

**转换**：URDF → USD（Isaac Lab 官方工具）；URDF → MJCF（dm_control 工具）；信息量不同，转换有损失。

**选型**：学术研究 → URDF；精细操作 → MJCF；大规模 GPU 训练/渲染 → USD。

**易错**：URDF 不支持闭链（loop closure），平行四杆等机构必须用 MJCF 的 equality 约束；直接用 URDF 会出现关节约束错误。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q25</b> · 仿真器的接触/碰撞模型（contact model）对 sim2real 的影响？为什么抓取任务难做？</summary>

**答**：**接触模型的核心问题**：真实接触是面接触（有面积、有变形、有摩擦锥），但主流仿真器用点接触近似（point contact）+ 弹簧阻尼模型（spring-damper），引入两类误差：

1. **接触力方向误差**：点接触无法模拟接触面的力矩分布（如抓取时手指包裹物体的摩擦力）
2. **接触刚度误差**：仿真的"软体接触"系数很难与真实物体材质匹配

**为什么抓取任务特别难**：
- 抓取稳定性取决于接触点分布和摩擦系数，仿真误差直接影响是否抓住
- 物体几何细节（倒角、表面粗糙度）在 CAD 模型中往往简化，真实接触面与仿真不同
- 插针（peg-in-hole）等精密操作误差仅 1mm，仿真接触力方向偏差就可能导致失败

**缓解策略**：使用 MuJoCo（Newton/CG/PGS 精细接触）+ 合适摩擦参数 / 收集少量真实数据 fine-tune / 用力传感器反馈替代精确接触建模。

**易错**：PhysX（Isaac Lab）速度快但接触精度低于 MuJoCo，精细抓取研究应优先选 MuJoCo；不要因为 Isaac Lab 并行速度快就用它做高精度接触任务。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q26</b> · ManiSkill 和 Isaac Lab 的 GPU 并行能力对比？哪个适合操作任务训练？</summary>

**答**：

| | ManiSkill3（SAPIEN/Vulkan） | Isaac Lab（Isaac Sim/PhysX） |
|---|---|---|
| 物理引擎 | SAPIEN（基于 PhysX 5 + Vulkan） | PhysX（NVIDIA 官方） |
| 128 env GPU 内存 | **3.5 GB** | 14.1 GB |
| 并行渲染 | 原生支持，速度极快 | 支持但资源占用高 |
| 操作任务丰富度 | 12+ 类操作域，human-artist 设计场景 | 操作任务少，步态类为主 |
| 学习框架集成 | RL + IL，直接支持视觉 obs RL | RL 为主 |
| 开源程度 | 完全开源 | Isaac Lab 开源，依赖闭源 Isaac Sim |

**结论**：
- 操作任务（抓取/灵巧操作/移动操作）→ **ManiSkill3 更合适**（内存效率高、操作任务丰富、可从视觉训练）
- 步态/人形运动 → **Isaac Lab 更合适**（四足/双足步态生态更完整，官方 ANYmal/G1 模板）

**易错**：Genesis 号称最快，但在物理精度可比的基准下，ManiSkill3 GPU 内存效率更高；Genesis 2024 年仍在成熟中，生产环境慎用。

</details>

---

## §6 具身 Sim2Real 实战（5 题）

> 人形 / 四足 / 操作任务的 sim2real 挑战差异 + 视频世界模型与 JEPA 的应用前景。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q27</b> · 人形机器人 sim2real 面临哪些特有挑战？和四足机器人 sim2real 有何不同？</summary>

**答**：

| 维度 | 四足（ANYmal / Unitree Go2） | 人形（Unitree G1 / Figure） |
|---|---|---|
| DoF | 12-16 | 30-50+（含双臂） |
| 稳定性 | 4 接触点，裕度大 | 2 接触点，balance 极敏感 |
| 主要 gap | 地形接触、地面摩擦 | 全身动力学、髋关节柔顺性 |
| 仿真工具 | Isaac Lab（模板齐全） | Isaac Lab / MuJoCo（模板快速增加） |

**人形特有挑战**：
1. COM 高，关节误差 1cm 即可导致摔倒；
2. 30+ 维 joint torques 输出，执行器模型误差被成倍放大；
3. 双臂 + 行走协同，sim2real 难度非线性增长。

**缓解**：Teacher-Student（特权观测含接触状态）+ RMA 在线适应 + 保守 DR 范围（不能太宽）。

**易错**：人形不是"四足加两条腿"——双足平衡对 contact model 误差极敏感，需比四足更精确的动力学建模。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q28</b> · 操作任务的 sim2real（抓取/插针/双臂）和运动控制的 sim2real 有何核心区别？</summary>

**答**：

| 维度 | 运动控制 sim2real | 操作任务 sim2real |
|---|---|---|
| 主要传感器 | IMU + 关节编码器（proprio） | 相机 + 深度 + 力传感器（多模态） |
| 关键 gap 来源 | 地形接触、动力学参数 | 视觉外观、物体接触/摩擦、末端精度 |
| 成功指标 | 稳定行走（鲁棒） | 任务完成（精确） |
| DR 有效性 | 高（地形/摩擦 DR 有效） | 中（接触模型误差难以 DR 覆盖） |
| 常用仿真器 | Isaac Lab / Isaac Gym | MuJoCo / ManiSkill / SAPIEN |
| 真机 fine-tune 需求 | 低（zero-shot 较常见） | 中-高（精细操作几乎总需要 fine-tune） |

**操作的特殊难点**：末端执行器精度要求 < 5mm（插针 < 1mm），仿真的接触误差直接影响任务成败；物体的 CAD 模型与真实物体几何存在偏差，导致视觉特征也有 gap。

**关键差异总结**：运动控制看"稳定鲁棒"，sim2real 靠 DR 基本解决；操作任务看"精确完成"，sim2real 需要数据 + fine-tune + 传感器融合。

**易错**：不能把运动控制的 zero-shot sim2real 成功经验直接套用到操作任务——两者的 gap 来源和解决方案都不同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q29</b> · Genie / 视频生成世界模型和具身智能的关联？它能否直接作为机器人 policy？</summary>

**答**：**Genie**（Google DeepMind，2024 / Genie 3 2025）：从视频自监督学习的交互式世界模型，给定初帧 + 动作指令生成后续帧；Genie 3 实现 24 FPS 实时 3D 世界生成。

**与具身的关联**：① 从无标注视频学物理先验，减少真机数据需求；② 生成多样场景替代传统仿真；③ 在生成视频中 rollout 动作序列评估。

**能否直接作为 policy**：目前不能。控制粒度粗（不是 joint torques）、推理延迟高（无法达到 50Hz）、缺少精确物理约束（接触力/惯量）。

**未来方向**：Video World Model + Action Expert 两阶段（类 π0）——视频模型负责感知规划，轻量 action head 输出控制量。

**易错**：视频世界模型（像素空间生成，高保真但慢）≠ Dreamer RSSM（latent 空间预测，快但不可视化）；前者是"看得见的世界"，后者是"隐式的动力学"。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q30</b> · JEPA（Joint-Embedding Predictive Architecture）和 Dreamer 世界模型的差异？</summary>

**答**：**JEPA**（LeCun 提出，Meta V-JEPA / V-JEPA 2）：在 embedding 空间预测目标特征，不重建像素。

| | JEPA（V-JEPA） | Dreamer（RSSM） |
|---|---|---|
| 预测目标 | 特征向量（feature space） | 像素重建 + latent 预测 |
| 重建损失 | 无 | 有（解码器） |
| 训练数据 | 互联网视频（无动作标签） | 带动作的 agent 交互数据 |
| 可用于控制 | 间接（高质量表征 → 下游 RL） | 直接（imagination rollout 训 actor） |

**V-JEPA 2**（Meta，2025，100 万小时视频预训练）：可做短期物理预测，作为 backbone 提升操作策略效果；类似 VLM 之于 VLA 的作用。

**易错**：V-JEPA 2 已有 action-conditioned 版本（V-JEPA 2-AC），支持短程规划和机器人控制；但它没有 Dreamer 那样的 long-horizon imagination rollout 训 actor 的成熟流程，两者适用场景不同。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×4</span> <b>Q31</b> · TD-MPC 的 temporal difference + MPC 是如何结合的？和纯 MPC 有何区别？（**破例**：TD-MPC2 ICLR 2024 顶会，面试中正在新增）</summary>

**答**：**TD-MPC**（Hansen et al., 2022；TD-MPC2 ICLR 2024）结合 TD 学 value 和 MPC 做规划：

**TD 部分**：学 latent 动力学 $z_{t+1}=f(z_t,a_t)$、Q 函数（TD 误差监督）、reward 预测；latent 用 SimNorm 防坍缩。

**MPC 部分**：每步用 CEM 在 latent 空间搜索 H 步最优序列，序列末端价值用学到的 Q 评估（解决纯 MPC 只看 H 步的近视问题）；选第一步执行，下步重规划。

**与纯 MPC 的区别**：纯 MPC 依赖精确物理模型、无 value function；TD-MPC 用 TD 提供无限 horizon 的价值估计，精度和泛化性更强。

**易错**：TD-MPC2 是 decoder-free（不重建观测），latent 只对 TD/reward 预测有意义，不是可视化状态；和 Dreamer 的重建式 latent 本质不同。

</details>

---

## §7 低频备选（未入主表）

> 以下 4 题频次 1-2，保留供参考。面试中偶发，不作为必刷优先级。

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2</span> <b>N2</b> · V-JEPA 2 的预测目标是什么？和 Dreamer 的 reconstruction loss 有何区别？</summary>

**答**：**V-JEPA 2**（Meta，2025，100 万小时视频预训练）的预测目标是**遮挡区域的 patch 特征向量**，而非像素值：给定上下文帧（部分可见），预测被遮挡 patch 的 encoder 输出（在 feature space 中预测，而非 pixel space）。

**与 Dreamer 重建损失的区别**：
- Dreamer：decoder 将 latent 解码回像素，优化像素级 MSE；学的是"能还原图像"的 latent
- V-JEPA 2：predictor 在特征空间预测目标特征，无像素解码器；学的是"能预测遮挡部分特征"的表征

**V-JEPA 2 的优势**：避免了像素级重建的高频细节（噪声）干扰，表征质量更高；可直接迁移到 action-conditioned 场景进行下游操作策略学习。

**易错**：V-JEPA 不等于 JEPA（后者是架构原则），V-JEPA 是视频版本；V-JEPA 2 是 2025 年的更新版，加入了更多 embodied 相关任务评估。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N3</b> · 仿真器中的 actuator 模型如何影响 sim2real？PD/torque 控制的选择？</summary>

**答**：**Actuator 模型**是 sim2real 中常被忽视的 gap 来源。仿真默认理想执行器（无延迟、无摩擦、线性响应），但真实电机有：电机动态延迟（约 3-10ms）、齿轮传动间隙（backlash）、热效应、电流饱和。

**PD vs Torque 控制的选择**：
- **PD 控制**（仿真主流）：给定目标关节角，仿真内置 PD 控制器输出力矩。隐藏了低层电机动态，sim2real gap 小，但精细控制受限
- **Torque 控制**（精细操作）：直接输出关节力矩，完全暴露电机动态，gap 大但控制精度高

**实务建议**：步态控制（四足/人形）优选 PD 控制（配合适当 Kp/Kd 随机化）；精细操作优选 torque 控制 + 执行器延迟建模。

**易错**：不要在仿真中用过大的 PD 增益（刚性仿真），真机上无法重现；降低 PD 增益或加柔顺控制往往使 sim2real 更好。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×2</span> <b>N4</b> · Habitat 3.0 / SAPIEN 的应用场景？和 Isaac Lab 的定位有什么不同？</summary>

**答**：
- **Habitat 3.0**（Meta AI）：室内场景导航和人机协作仿真，强调**高真实感室内环境**（PhotoAugmented 渲染）；特色是社交导航（机器人与人协作）和语义导航任务；GPU 并行有限，主打感知-导航研究。
- **SAPIEN**（清华/斯坦福）：基于 PhysX 的关节物体仿真，提供大量**可关节化物体数据集**（PartNet-Mobility）；ManiSkill 基于 SAPIEN 构建；擅长门、抽屉、开关等关节操作任务。
- **Isaac Lab**：NVIDIA Omniverse 生态，专注**大规模 GPU 并行 RL**，四足/人形步态完整工具链；场景真实感高（USD/Omniverse 渲染）；操作任务相对较少。

**定位总结**：Habitat → 室内导航/人机协作；SAPIEN/ManiSkill → 关节物体操作；Isaac Lab → 大规模步态/运动 RL；MuJoCo → 学术控制基准。

**易错**：SAPIEN 和 ManiSkill 是不同层级——SAPIEN 是物理引擎层，ManiSkill3 是在 SAPIEN 上构建的任务基准框架，不要混淆。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N5</b> · 世界模型的"model error accumulation"（模型误差累积）如何缓解？</summary>

**答**：**模型误差累积**：世界模型在 latent 空间的 H 步前向展开中，每步误差叠加，H 步后 latent 偏离真实状态越来越远，导致 imagination 训出的策略在真实环境中失效。

**缓解策略**：
1. **短 imagination horizon**：Dreamer 实务 H=15，不宜太长（H>20 误差明显放大）
2. **不确定性感知**：对模型输出的不确定性加权，高不确定区域不做更新或降低学习率
3. **frequent model refresh**：每 N 个真实 step 后重新训世界模型，对齐当前策略的分布
4. **ensemble**：多个世界模型取平均，降低单模型 bias
5. **latent 正则化（SimNorm / free bits）**：防止 latent 随时间漂移

**与 offline RL 对比**：offline RL 中的 OOD 问题类似——策略迭代后访问的状态超出数据分布，解法都是"保守更新 + 不确定性罚项"。

**易错**：加大 H 不能提升 imagination 质量，只会加剧误差累积；增大 H 需要配合更好的世界模型质量（更多真实数据）。

</details>

---

*题库维护：可查 [具身智能面试题库主页](../index.html) 访问其他分册。*
