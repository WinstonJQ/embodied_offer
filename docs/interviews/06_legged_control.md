# 卷六 · 腿足机器人运动控制 / 全身控制 / 遥操作 · 高频面试题

> 中文具身智能秋招高频面试题库 · **第六卷**
> 题源：牛客 / 知乎 / CSDN 公开面经 / OpenLoong 项目文档 / MIT-Cheetah 论文 / 各大人形项目 arXiv 论文 / 公司公开招聘 JD
> 同义题合并后 **48 题**（主表 45 题：频次 ≥3 + 低频备选 3 题）

**难度分布**：L1（必会） **5** · L2（进阶） **36** · L3（顶级/前沿） **4** （低频备选另计：L2 ×2 / L3 ×1）

**使用方式**：题目默认折叠，点开看答案。建议按 **L1 → L2 → L3** 顺序刷；同级内按频次从高到低。手机端原生支持。

**与其他卷的关系**：本卷与卷二（RL 算法）在 PPO / curriculum / domain randomization 上有意保留少量重叠——视角不同：卷二讲算法本身，本卷讲算法**怎么用在腿足上**。

---

## §1 浮动基座 / 动力学 / 接触建模（7 题）

> 腿足机器人的物理基础：浮动基座意味着无法用固定机械臂的那套运动学/动力学公式，必须显式建模接触。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q01</b> · 什么是浮动基座（floating base）？腿足机器人为什么不能直接用固定基座机器人的运动学 / 动力学公式？</summary>

**答**：**浮动基座**：躯干在世界坐标系无固定连接，姿态由 6 自由度自由变量描述（3 平动 + 3 旋转），靠脚-地接触约束被动维持位置。

**与固定基座区别**：
- 固定机械臂：广义坐标 $q \in \mathbb{R}^n$；动力学 $M\ddot q + C\dot q + g = \tau$
- 浮动基座：$q = [q_b, q_j]$，前 6 维基座**无主动驱动**，需接触力"驱动"基座

**实务建模**：base 作 6-DoF 虚拟关节挂到 world，用 Featherstone 浮动基 RNEA / ABA 求解——Pinocchio / Drake / RBDL 都这么做。

**易错**：不要把"浮动基座"和"欠驱动"混为一谈——浮动基座说的是建模方式，欠驱动说的是控制特性。是否欠驱动取决于接触模型 / 约束秩：飞行相、点足接触、动态步态通常欠驱动；平足刚性支撑可近似为受约束全驱动。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q02</b> · 雅可比矩阵在腿足机器人里的作用？接触雅可比和任务空间雅可比的区别？</summary>

**答**：雅可比 $J(q)$ 把广义速度映射到笛卡尔速度。腿足里两类雅可比：

| | 接触雅可比 $J_c$ | 任务雅可比 $J_t$ |
|---|---|---|
| 作用 | 把接触力反向映射到广义力 $\tau_c = J_c^\top \lambda$ | 描述足端 / 质心 / 躯干姿态速度 |
| 用途 | 浮动基座动力学约束 / MPC 摩擦锥 | WBC / OSC 跟踪目标 |

**关键约束**：脚不滑时 $J_c \dot{q} = 0$；微分得 $J_c \ddot{q} + \dot{J}_c \dot{q} = 0$，是 WBC 等式约束的标准形式。

**易错**：浮动基座的雅可比形状是 $J \in \mathbb{R}^{m \times (6+n_j)}$，前 6 列对应 base——不要只对关节求导；两类雅可比都必须包含 base 部分才能在浮动基座下成立。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q03</b> · 摩擦锥 friction cone 是什么？为什么在 MPC / WBC 里通常做成线性化金字塔约束？</summary>

**答**：**摩擦锥**：地面接触力的物理可行域，约束法向力 $f_n \ge 0$（不能拉脚）和切向力 $|f_t| \le \mu f_n$（库伦摩擦不滑）。几何上是一个以法向为轴、半角 $\arctan\mu$ 的圆锥。

**为什么线性化**：原始摩擦锥是二阶圆锥约束（SOCP），求解器慢；常用 4 边或 8 边金字塔近似，把约束变成**4 / 8 条线性不等式**，整个优化变成 QP，OSQP / qpOASES 这类 QP 求解器可以 < 1 ms 解。

**线性化代价**：4 边**内接**金字塔保守（切平面面积比圆锥小约 36%）但**安全**——不会让机器人违反真实摩擦；8 边内接更贴近圆锥；**外接矩形**（$|f_x|, |f_y| \le \mu f_z$）反而**不保守**，角点会超出圆锥。生产里 4 / 8 边内接是常见取舍。

**易错**：摩擦锥约束 $|f_t| \le \mu f_n$ 是非线性的，不是简单 $|f_t| \le \mu f_{n,\max}$；不要在写 QP 时把 $f_n$ 当常数。MIT Cheetah Convex MPC 通过同时把法向力和切向力作为决策变量、配合金字塔近似，把整个问题变 QP。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q04</b> · 质心动力学（centroidal dynamics）和单刚体模型（SRBM）有什么区别？为什么 MPC 常用 SRBM？</summary>

**答**：两者都是**简化模型**，但抽象层次不同：

| | 单刚体模型 SRBM | 质心动力学 |
|---|---|---|
| 假设 | 把整机当一个刚体，惯量矩阵常数 | 用整机的质心动量 $h_G$ 描述系统状态 |
| 状态 | 质心位置 / 姿态 / 速度 / 角速度 | 线动量 + 角动量 + 质心位置 |
| 输入 | 各支撑足的接触力 $f_i$ | 同上（接触力直接产生动量变化率） |
| 用途 | MIT Cheetah 凸 MPC、Convex MPC | 更精细的 NMPC / WBC 内的质心任务 |

**为什么 MPC 偏爱 SRBM**：① 假设腿质量小 + 惯量近似常数后整个动力学**变线性**；② 加上摩擦锥金字塔近似后整个问题是 QP，可 100-500 Hz 解；③ 实测在四足上效果已经够好。

**易错**：SRBM **忽略了腿的惯量**——快速摆腿时会有几个百分点的建模误差；高动态任务（跳跃、奔跑）需要全身动力学或质心动力学补偿。质心动力学不等同于 SRBM，前者是更通用的降阶表达。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q05</b> · 什么是 ZMP（零力矩点）？和 CoP、Capture Point、DCM 是什么关系？</summary>

**答**：**ZMP（Zero Moment Point）**：地面上一个虚拟点，绕该点地面反力对水平面的力矩为零。**当 ZMP 严格落在支撑多边形内**，机器人不会绕支撑边沿翻倒；落到边沿则机器人即将倒下。

**几个概念对比**：
- **CoP（压力中心）**：实际地面反力的"加权重心"。**ZMP = CoP 当 ZMP 落在支撑多边形内**；落出去 ZMP 不再有物理意义，CoP 仍在边界上
- **Capture Point**：机器人当前状态下，若立即把脚踩到该点上，就能在不再迈步的情况下回到静止
- **DCM（Divergent Component of Motion）**：质心动量的发散分量，是 Capture Point 的通用化连续形式；DCM 轨迹规划可以反推质心 + 落脚点

**应用**：传统人形（ASIMO/HRP）用 ZMP 离线规划质心轨迹；现代人形多用 DCM / Capture Point 做在线规划，对扰动响应更快。

**易错**：ZMP 是静态稳定性判据，对快速动态（如跑步飞行相）失效——飞行相没有 ZMP，需要质心动量守恒分析。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q06</b> · Pinocchio / Drake / RBDL 这几个动力学库各自的定位？为什么腿足机器人圈 Pinocchio 最常用？</summary>

**答**：

| 库 | 定位 | 优势 | 劣势 |
|---|---|---|---|
| **Pinocchio** | Featherstone 体系 C++ / Python | 浮动基座原生支持；ABA / RNEA / CRBA 全有；**解析微分**；快（约 1 µs 操作臂 / 3 µs 腿足） | 学习曲线陡 |
| **Drake** | TRI / MIT 大库 | 内建优化（MathematicalProgram）/ 仿真 / 控制一体 | 体量大、编译复杂 |
| **RBDL** | 轻量 C++ 实现 | 与 Featherstone 书完全对齐，易学 | 工具链生态弱 |

**为什么 Pinocchio 主流**：① 速度最快（HPP 项目优化十年）；② **解析雅可比 + 解析微分** 让上层 MPC / WBC 的代价梯度可以闭式得到，比数值差分稳定；③ TSID / Crocoddyl 等主流腿足栈都以 Pinocchio 为基；④ Python 绑定方便快速原型。

**易错**：KDL（ROS 自带）**不支持浮动基座**——只能做固定基机械臂，所以 ROS 默认动力学库在腿足圈基本用不上。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q07</b> · RNEA、CRBA、ABA 三种动力学算法各自解决什么问题？复杂度怎么样？</summary>

**答**：刚体动力学三大经典算法（Featherstone）：

| 算法 | 问题 | 复杂度 | 用途 |
|---|---|---|---|
| **RNEA** | 逆动力学 $\tau = f(q, \dot{q}, \ddot{q})$ | $O(n)$ | WBC 控制律 |
| **CRBA** | 求质量矩阵 $M(q)$ | $O(n^2)$ | KKT 系统 / 解析微分 |
| **ABA** | 正动力学 $\ddot{q} = f(q, \dot{q}, \tau)$ | $O(n)$ | 仿真器内核 |

**三件套关系**：WBC 需要 $M$（CRBA）+ 逆动力学算力矩（RNEA）；仿真器要正推状态（ABA）。

**易错**：RNEA 是**逆动力学**——给加速度算力矩，不是反过来；常见误用。复杂度 $n$ 是连杆数；腿足 30 关节场景下三者实测都在 µs 级。

</details>

## §2 MPC 系列（7 题）

> 模型预测控制是腿足圈最经典的控制框架，从 MIT Cheetah 的 Convex MPC 开始统治了四足近 7 年。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q08</b> · MIT Cheetah 提出的 Convex MPC 怎么把腿足运动控制问题做成凸优化？</summary>

**答**：MIT Carlo & Wensing 2018 的 Convex MPC（应用于 Cheetah 3）做了几个关键近似把非凸 NMPC 化简到 QP：

1. **单刚体模型 SRBM**：忽略腿质量，整机当一个固定惯量的刚体
2. **小角度假设**：在 hover frame 里把 roll/pitch 当小量，旋转动力学 $I\dot{\omega} = \tau - \omega \times I\omega$ 中的 $\omega \times I\omega$ 项忽略（线性化）
3. **接触序列预先给定**：哪只脚什么时间落地由 gait scheduler 输出，不作为决策变量，避免组合优化
4. **摩擦锥金字塔近似**：4-边线性化（见 Q03）

化简后状态线性、约束线性、代价二次，是标准 QP；用 dense QP 求解器约 1 ms 内可解，30 Hz 跑实时，时域 0.5 s / 10-16 步。

**易错**：原论文 30 Hz 是 MPC 重规划频率，**底层关节力矩控制器跑 1 kHz 甚至更高**——别把 MPC 频率和电机控制频率混为一谈。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q09</b> · MPC 的预测时域 / 控制时域怎么选？为什么腿足 MPC 一般 0.3-0.5 s、10-20 个时步？</summary>

**答**：选时域的两个权衡：

- **太短**：看不到未来一个完整步态周期，控制器决策短视（脚还没落地就结束）
- **太长**：① 模型误差累积；② QP 规模 $\propto$ 时步数，10 步 vs 30 步求解时间差 5-10 倍

**腿足实务**：典型步态周期 0.3-0.5 s，覆盖 1-2 个完整周期最佳，于是 10-20 步、每步 25-50 ms。MIT Cheetah 论文用 0.5 s / 16 步 / 30 Hz，是一个长期被沿用的配方。

**控制时域 vs 预测时域**：MPC 标准定义中"控制时域 ≤ 预测时域"——预测时域看远，控制时域内的控制变量做决策变量；腿足里通常**两者相等**简化实现。

**易错**：不要假设"时域越长越好"——长时域要求模型在 0.5 s 内仍准确，超过这个时间 SRBM 误差会显著上升。NMPC（非线性 MPC）可以用更长时域（1 s+），但算力代价大。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q10</b> · MPC 求解器对比：OSQP vs qpOASES vs HPIPM，腿足实时控制怎么选？</summary>

**答**：

| 求解器 | 算法 | 优势 | 劣势 |
|---|---|---|---|
| **OSQP** | ADMM | 稀疏问题快、warm-start 好、BSD 开源 | 一阶，迭代多 |
| **qpOASES** | active-set | 中小问题快、解精确 | 不适合超大稀疏 |
| **HPIPM** | 内点 + Riccati | 对 MPC 时序结构最优 | 接口较复杂 |

**腿足典型选择**：四足 Convex MPC → OSQP / qpOASES 都 < 1 ms；人形 30 自由度 WBC → HPIPM / ProxQP 适配 MPC 时序；嵌入式 → 自研 ADMM。

**实务**：先用 OSQP（开源、接口干净），延迟不达再换 HPIPM。

**易错**：OSQP 是一阶方法，对 ill-conditioned 问题收敛慢——MPC 代价矩阵尺度差异大时要先做预处理 / 缩放。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q11</b> · LQR、PID、MPC 在腿足控制里各自的定位？为什么四足主流都是 MPC？</summary>

**答**：

| | PID | LQR | MPC |
|---|---|---|---|
| 模型 | 不需要 | 线性，需 $A$、$B$ | 线性/非线性都行 |
| 约束 | 不支持 | 不支持 | **原生支持**（摩擦锥、力上下限） |
| 计算 | 几乎免费 | 离线一次解 Riccati | 每周期解 QP |
| 时域 | 单点反馈 | 无限时域 | 有限滚动时域 |

**腿足偏爱 MPC 的原因**：① 接触约束是**硬约束**（摩擦锥、不可拉脚），LQR / PID 没法直接处理；② 步态切换时接触序列变，需要重新优化，MPC 滚动天然适合；③ 现代算力 1 ms 解 QP 可行。

**简单系统例外**：质心动力学接近线性倒立摆、接触不切换的系统（如轮式平衡），LQR 一次离线解 Riccati 就够；详见低频备选 N3。

**易错**：PID 在腿足里**仍然存在**——但只用在最底层关节伺服（电流环、位置环），MPC / WBC 是上层。说"腿足只用 MPC"是不准确的。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q12</b> · 接触时序（contact schedule / gait scheduler）在 MPC 里是怎么编码进去的？热切换怎么避免抖动？</summary>

**答**：**接触时序**：一个 0/1 矩阵 $s_i^k$，表示第 $k$ 个 MPC 时步、第 $i$ 只脚是否接触地面。Convex MPC 把它**预先给定**（gait scheduler 输出），不作为优化变量，这才能保持凸性。

**怎么用**：
- $s_i^k = 1$（支撑相）：摩擦锥 + 力上限约束生效；摆动相约束变为 $f_i^k = 0$
- 与 swing-leg 控制器协作：摆动期 $f_i^k = 0$，足端轨迹由独立的 swing leg 控制器（如五次多项式插值）执行

**避免抖动**：① 落地 / 离地切换瞬间，QP 解会有跳变 → **力的差分项**加入代价（$\| f^{k+1} - f^k \|^2$）；② 接触序列由 finite state machine 维护，不允许中途修改；③ 接触检测用低通滤波 + 滞回。

**易错**：步态切换时（如 trot → bound）整张时序表会重写，热切换需要把"未来 0.3 s"的接触模式一次性更新，否则 MPC 看着旧时序解出错误力。生产代码里通常做"phase 锁定"，切换在某只脚刚落地时执行。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q13</b> · 摆动腿（swing leg）轨迹一般怎么设计？为什么常用五次多项式或 Bezier？</summary>

**答**：摆动腿轨迹要求：① 起飞点与落地点匹配；② 起飞 / 落地速度 = 0（避免冲击）；③ 中间有抬腿高度（避障）。

**为什么五次多项式（quintic / minimum-jerk）**：六个边界条件（起 / 落各一对 位置 / 速度 / 加速度）刚好对应五次多项式的 6 个系数，解析可求；jerk 最小 ⇒ 加速度平滑 ⇒ 电机受力舒缓。MIT Cheetah 用五次 + 可调最高点（apex height）。

**Bezier 曲线**：通过控制点直观调形，常用于希望"前快后慢"或"避开高障碍"的轨迹；ANYmal 系列也常用。

**落脚点怎么算**：常用 Raibert heuristic $p_{foot} = p_{hip} + \frac{v T_{st}}{2} + k(v - v_{des})$，其中 $T_{st}$ 是支撑相时长、$k$ 是反馈增益。RL 训练的策略可以学到更精细的落脚点选择。

**易错**：摆动期不能假设质心匀速——MPC 给的质心轨迹是变速的，落脚点要用 MPC 输出的预测质心位置而不是当前位置。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q14</b> · NMPC（非线性 MPC）和 Convex MPC 比，哪些场景值得为额外算力换更优解？</summary>

**答**：NMPC 不做小角度近似、不做 SRBM，保留全身动力学和圆锥摩擦锥；用 SQP / DDP / iLQR 求解，单次 5-20 ms。

**值得换 NMPC 的场景**：
- 大姿态机动（深蹲、跳跃落地、滚翻）—— 小角度假设崩
- loco-manipulation（推门、扛物）—— SRBM 忽略的腿惯量影响显著
- 高动态奔跑 / 飞行相 —— 线性化误差大
- contact-implicit MPC（接触时序也作决策变量）

**Convex MPC 够用**：四足 trot / walk、人形低速行走；Unitree / ANYmal 主线产品仍用 Convex。

**前沿**：Crocoddyl（INRIA）DDP-NMPC、IIT contact-implicit MPC 在 ANYmal 跳跃；RL policy 也在抢这个生态位。

**易错**：NMPC ≠ "更好"——SQP 易陷局部最优，warm-start 不好比 Convex MPC 还慢；产线优先 Convex MPC，Convex 失败才考虑 NMPC。

</details>

## §3 全身控制 WBC（6 题）

> WBC 把多任务（质心轨迹、足端力、姿态、关节限位）写成分层 QP，是人形不可绕开的中间层。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q15</b> · WBC（whole-body controller）的核心思想是什么？为什么需要任务优先级？</summary>

**答**：**WBC**：在每个控制周期（典型 1 kHz）解一个**约束二次规划**，同时满足多个相互冲突的运动学/动力学任务（质心位置、躯干姿态、足端力、关节限位等），输出关节力矩。

**为什么要优先级**：多任务常冲突。例如人形要"跟踪末端轨迹"+"维持平衡"+"避免关节超限"，冲突时**平衡必须赢**。两种实现：

| 实现 | 写法 | 行为 |
|---|---|---|
| **加权 QP** | 所有任务加权进单一代价 | 调权重难 |
| **严格 HQP** | 高优先级先解，零空间留给低优先级 | 严格不冲突，但难实现 |

**实务**：人形产线多用加权 QP；学术界爱严格 HQP。TSID 是 Task-Space Inverse Dynamics 框架（开源 stack-of-tasks/tsid 库内部用 HQP solver 做严格层级，同层内可用权重软化任务）。

**易错**：WBC 不能替代 MPC——WBC 只看当前一拍（无预测），平衡 / 落脚等长期规划必须靠 MPC / 步态规划器；典型架构是 **MPC（10-100 Hz）输出参考 → WBC（1 kHz）跟踪 + 力分配**。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q16</b> · TSID 和 HQP（hierarchical QP）的区别？weighted QP 和 strict hierarchy 各自的取舍？</summary>

**答**：TSID 是 Task-Space Inverse Dynamics 框架（Saab 2013 + stack-of-tasks/tsid 库），内部 solver 实际做 HQP 严格分层，但**同一层可用权重软化**多个任务；HQP 特指严格层级。常用对比 weighted vs strict hierarchy：

| | 加权 QP（单层） | 严格 HQP（多层） |
|---|---|---|
| 结构 | 任务加权，单 QP | 多层，高层零空间传给低层 |
| 调参 | 权重，**任务间偷量** | 严格不冲突，但难实现 |
| 计算 | 单 QP（毫秒） | k 层 QP（更慢） |
| 代表实现 | OpenSoT 加权 / Pink IK QP | tsid SolverHQP、OpenSoT 严格 |

**取舍**：加权简单快、调参直观，但极端配置下低优先级会拽偏高优先级；严格 HQP 理论严格但实现复杂。

**实务**：产线多用加权 QP；研究界爱严格 HQP。

**易错**：一个"任务"通常是一组等式 + 不等式（关节限位、摩擦锥），不是单一约束；层级指这一组的优先级。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q17</b> · OSC（operational space control）vs TSID：本质区别在哪？</summary>

**答**：两者都把任务空间映射到关节力矩，但范式不同：

| | OSC（Khatib 1987） | TSID（Saab 2013） |
|---|---|---|
| 形式 | 闭式 $\tau = J^\top (\Lambda \ddot{x}_{des} + \mu + p)$ | QP 优化 |
| 不等式约束 | 不自然支持 | 原生支持（摩擦锥 / 关节限位） |
| 速度 | 矩阵运算，超快 | QP 求解，几毫秒 |
| 浮动基座 | 需额外处理 | 原生支持 |

**OSC 本质**：把动力学投影到任务空间 $\Lambda \ddot{x} = F + \mu + p$ 反解力矩；固定基机械臂上单层任务优于 TSID。

**TSID 优势**：统一处理浮动基座 + 接触约束 + 多任务，是腿足/人形的天然选择。

**实务**：腿足圈 TSID 主流（stack-of-tasks/tsid 是事实标准）；OSC 思想仍以"任务空间投影"融入 WBC。

**易错**：OSC 不是"过时"——单任务 + 已知接触状态时仍更快；TSID 的优势在多任务 + 不等式约束。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q18</b> · WBC 和 MPC 一般怎么配合？层次频率怎么分？</summary>

**答**：典型分层（OpenLoong / Cheetah / ANYmal 通用）：

- **高层规划 1-10 Hz**：任务编排 / VLA / 视觉，给目标
- **MPC 30-100 Hz**：输出未来 0.5 s 质心轨迹 + 接触力
- **WBC 500-1000 Hz**：跟踪 MPC 输出 + 全身动力学 + 力分配，出关节力矩
- **关节伺服 > 2 kHz**：电流环 / 力矩闭环

**为什么分层**：MPC 单次 QP 约 1-5 ms 不能 1 kHz；WBC QP 规模小可 1 kHz；电机伺服需 > 1 kHz 稳定。

**通信**：MPC → WBC 通过共享内存 / ROS2；传递的是**力 + 加速度参考**，不是关节力矩。

**易错**：MPC 直接生成关节力矩跳过 WBC 在简单四足上可行，但**复杂人形不行**——MPC 模型粗（SRBM）不能直接出 30+ 关节力矩；WBC 是必经中间层。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q19</b> · 零空间投影（null-space projection）在 WBC 里怎么用？为什么这样能保任务优先级？</summary>

**答**：**零空间投影**：把低优先级任务的控制信号投影到高优先级任务的零空间（不会改变高优先级任务的输出），从而实现"低优先级不干扰高优先级"。

**数学形式**：若高优先级任务雅可比 $J_1$，则零空间投影矩阵 $N_1 = I - J_1^+ J_1$。低优先级 $\ddot{q}_2$ 控制律变为：
$$\ddot{q} = J_1^+ \ddot{x}_{1,des} + N_1 \ddot{q}_2$$
左边任意 $\ddot{q}_2$ 都不影响 $\ddot{x}_1$，因为 $J_1 N_1 = 0$。

**HQP 中的用法**：每一层解完后，把它的零空间作为下一层的"可用动作空间"，递归到底。等价于多层 QP，每层在前层零空间内优化。

**易错**：**零空间维度可能不够**——高优先级任务用了 $k$ 个自由度，只剩 $n-k$ 个给低优先级；机械臂 7 关节时只剩 1 维，几乎没法干次要任务。这是 redundancy（冗余度）不足的根本限制，加任务前要算清楚自由度账。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q20</b> · 阻抗控制、导纳控制、力位混合控制在腿足里的应用场景对比？</summary>

**答**：三者都是含柔顺性的控制策略，关键区别在输入-输出因果方向：

| | 阻抗 Impedance | 导纳 Admittance | 力位混合 Hybrid |
|---|---|---|---|
| 输入→输出 | 运动偏差 → 力 | 力偏差 → 运动 | 分方向：力 / 位置正交 |
| 适用机器人 | 力矩控制电机 | 位置控制电机 + 外置 F/T | 两者皆可 |
| 接触稳定性 | 接触刚硬环境也稳 | 刚硬环境易振 | 力方向稳，位置方向精确 |

**腿足应用**：
- **支撑腿** = 阻抗控制：虚拟弹簧-阻尼吸收着地冲击；摆动腿 = 位置控制
- **loco-manipulation** = 阻抗 + 力位混合：抓物方向位置，接触方向力
- **早期人形（ASIMO 等位置控电机）** = 导纳控制：F/T 外环算位置参考、内环 PID 跟踪

**易错**：PD 控制 = 简化阻抗（无虚拟惯量项），仍属于阻抗范式（位移 → 力），不是导纳；导纳的因果方向反过来（力 → 位移），需要 F/T 传感器读力做反馈。

</details>

## §4 RL-based locomotion（8 题）

> 2021 年 RMA 之后，RL-based locomotion 已成主流——尤其在感知（perceptive）、动态地形、人形上身扰动等场景下打败传统 MPC。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q21</b> · 用 PPO 训腿足 locomotion，为什么 IsaacGym / IsaacLab 这种 GPU 并行仿真是事实标准？</summary>

**答**：腿足 RL 的样本需求极大（百万级 step），CPU 仿真吞吐太低。

**GPU 并行三大优势**：① 数千环境并行（IsaacGym 典型 4096 envs，加速 100-1000 倍）；② 观测/动作驻留显存，无 CPU↔GPU 拷贝；③ PPO 与仿真同卡，端到端吞吐百万 step/分钟。

**典型配置**：4096 envs、PPO、horizon 32-50、几千万 step 几小时收敛（Humanoid-Gym / legged_gym）。

**仿真器选择**：
- **IsaacGym**（已停更）：legged_gym 等经典仓库依赖
- **IsaacLab**：NVIDIA 接班，物理后端可换 Newton
- **MJX / MuJoCo Playground**：JAX + MuJoCo，开源跨厂

**易错**：envs 不是越多越好——超过 4096 batch 内方差减小、PPO 优势估计噪声小反而**收敛更慢**（Brax 论文实测）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q22</b> · 腿足 locomotion 的 reward 一般怎么设计？为什么常拆 task / style / regularization 三类？</summary>

**答**：legged_gym / Humanoid-Gym 等主流框架通常拆三类：

| 类别 | 例子 | 作用 |
|---|---|---|
| **Task** | 跟踪速度命令、保持高度 | 完成任务 |
| **Style** | 步态周期、抬腿高度、对称 | 让运动像走路 |
| **Regularization** | 关节力矩平方、动作平滑、限位惩罚 | 节能、防过载 |

**为什么必须拆**：只设 task reward 会出现"姿态崩但完成任务"的解；style + regularization 才能引导到自然可部署步态。论文里 reward 项常达 15-25 个。

**常见坑**：① 系数失衡——style 太重不学动、reg 太重不动；② Reward hacking——策略找漏洞（来回小步刷 air time），要加约束。

**易错**：**不要直接奖励"前进距离"**——会鼓励冲刺然后摔倒；用速度命令跟踪误差作为 task reward 更稳。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q23</b> · 课程学习（curriculum）在腿足 RL 里怎么搭？常见做法有哪些？</summary>

**答**：直接训复杂地形 / 高速 / 强扰动会让 PPO 早期完全摔倒——curriculum 是事实标配。

**常见类型**：
1. **速度命令**：从 ±0.5 m/s 开始按成功率扩到 ±2 m/s
2. **地形**：平地 → 斜坡 → 阶梯 → 碎石，game-style 关卡进阶
3. **扰动**：早期不推、后期加大随机外力 / 质量随机化
4. **机器人姿态**（人形特有）：上身先固定、下身学走，再放开上身

**实务**：按 episode return 或成功率触发难度提升；停滞太久回退一级。IsaacLab / Newton 内置 progress-based terrain curriculum 模板。

**易错**：curriculum 不是越激进越好——过快进阶会催化策略遗忘早期能力（catastrophic forgetting）；通常保留旧难度环境做"经验回放"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q24</b> · RMA（Rapid Motor Adaptation）的核心思想是什么？两阶段训练分别学什么？</summary>

**答**：RMA（Kumar et al, RSS 2021, UC Berkeley + CMU + FAIR）解决"训练时有特权信息（摩擦、质量、扰动力），部署时只有本体感觉"的 gap。

**两阶段**：
- **阶段一**：策略输入 = 本体感觉 + privileged info 的 latent $z$，PPO 训到收敛
- **阶段二 · Adaptation Module**：从近 N 步本体感觉历史**监督预测** $z$；部署时 adaptation module 替代特权编码器

**部署**：adaptation module 约 100 Hz，几百毫秒内适应新环境（不同摩擦、负载）。原论文 A1 在沙地、草地、徒步路径上零 fine-tune 走过。

**为什么 work**：本体感觉历史**隐含**环境信息（打滑、扭矩异常对应低摩擦），encoder 学的就是从隐含信号恢复 latent。

**易错**：RMA 不是 sim2real 银弹——处理的是"训练分布内的物理参数变化"，对训练时未见过的极端情况（雪地）仍可能崩；ASAP 等用残差 action model 进一步弥补。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q25</b> · 教师-学生（teacher-student）蒸馏在腿足 RL 里为什么几乎是标配？privileged info 一般有哪些？</summary>

**答**：腿足部署时只有本体感觉（关节编码器、IMU），但仿真训练时可以"作弊"看到更多信息——教师-学生蒸馏让仿真效率最大化。

**典型 privileged info（教师独占）**：
- 摩擦系数、地面材质、阻尼
- 机器人当前质量、惯量
- 外力大小方向（随机推力）
- 完整地形高度图（足下 ± 0.5 m）
- 关节摩擦、电机响应延迟

**学生（部署用）输入**：
- 本体感觉历史（关节角、关节速度、IMU）
- 命令向量（速度命令、步态参数）
- 可选：视觉（depth / RGB）—— RMA / PIM / Walk These Ways 多用本体感觉为主，视觉是后期补加

**蒸馏方法**：① DAgger（学生在仿真里跑、教师在线给标）；② BC（教师轨迹存离线集学生 BC）；③ 用教师 latent 监督学生 latent。

**易错**：教师不能太强—— privileged info 让教师拿到"超人类"信号，学生本体感觉就是恢复不出来时，蒸馏永远失败；要保证 privileged info 与本体感觉有可学习的映射关系。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q26</b> · Walk These Ways 提出的 MoB（Multiplicity of Behavior）想解决什么问题？</summary>

**答**：传统 locomotion 策略一个模型只学一种步态/速度风格，调速度命令仍是同一步态——**新任务（爬阶、推门）就要重新训**。

**MoB 的洞察**：让单个策略学一个**结构化的行为族**（family），通过外部参数（步态周期、抬腿高、躯干姿态）切换行为；新任务时**不重训，只调参数**。

**做法**：
- Reward 里加入 gait-encoding 项，强制策略在不同周期/相位下都能跟踪
- 训练时随机抽样所有行为参数（速度、步态周期、抬腿高、躯干俯仰）
- 部署时人 / 高层 / 脚本动态指定参数

**结果**：Margolis & Agrawal 在 Unitree Go1 上展示一个策略可以做：跑、跳、爬阶、蹲下、舞蹈、抗推——**全是同一个网络权重**。CoRL 2022 (arXiv 2212.03238) 开源 walk-these-ways 仓库被广泛复用。

**易错**：MoB 不是"无限通用"——策略仍只能在训练时随机化过的参数范围内泛化；完全没见过的动作（如倒立行走）仍需要重新训。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×4</span> <b>Q27</b> · ASAP 通过什么机制对齐仿真和真机？delta action model 起什么作用？</summary>

**答**：ASAP（CMU + NVIDIA, 2025-02, arXiv 2502.01143）解决人形 sim-to-real 剩余 gap——IsaacGym 训得好的策略到 Unitree G1 上仍会摔。

**两阶段**：① 仿真预训练，retargeted 人体动作 + PPO 训运动跟踪；② 实机收集 (state, sim_action, real_outcome)，训 delta action model $\Delta a = a_{real} - a_{sim}$，装进仿真器后 fine-tune 原策略。

**为什么是 action**：相比改 dynamics（actuator net），改 action 更简单——少量真机数据、小 MLP 即可。

**关键贡献**：IsaacGym → IsaacSim / Genesis / Unitree G1 三套迁移都验证；与 RMA / actuator net **互补**可叠加。

**易错**：ASAP 不是纯零样本 sim2real——需真机数据；零样本场景仍要回到 actuator network + 域随机化。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q28</b> · HIM（Hybrid Internal Model）/ belief encoder 在 RL locomotion 里起什么作用？</summary>

**答**：把本体感觉历史 + 视觉编码成 latent state，让策略"看懂"环境隐含信息，是 RMA 思路的进化版。

**HIM（Hybrid Internal Model）核心**：用本体感觉历史预测"下一步本体感觉 + 速度"，迫使 encoder 学环境动力学的内部模型；与 RMA 不同**不需要 privileged info**——完全自监督。

**belief encoder（感知 locomotion）**：输入本体感觉历史 + 高度图 / depth；输出 latent（融合本体动力学 + 地形）；用 attention 或 GRU 编码。

**典型应用**：PIM（arXiv 2411.14386）把 LiDAR 高度图 + 本体感觉送进 belief encoder，让 humanoid 走阶梯。

**实务好处**：① 训练不依赖 privileged info；② sim2real gap 小；③ 对地形泛化好。

**易错**：belief encoder 比纯 MLP 慢，需 latency 权衡——典型做法 encoder 50 Hz、策略 200 Hz，encoder latent 喂多次。

</details>

## §5 人形遥操作 Teleop（6 题）

> 数据飞轮的入口：teleop 决定了模仿学习/VLA 训练数据的质量。Apple Vision Pro + 上身 IK 是 2024-2026 的事实标准之一。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q29</b> · OPEN-TeleVision 这套系统的核心组件是什么？为什么要用 Apple Vision Pro？</summary>

**答**：OPEN-TeleVision（Cheng et al, arXiv 2407.01512, UCSD + MIT）是 2024-2026 人形 teleop 事实标准之一。

**核心组件**：① VR 头盔（Apple Vision Pro / Quest）捕获头 / 手腕 / 手指 pose；② 机器人头部双目相机流回 VR 做 immersive 反馈；③ 服务器做 human-to-robot retargeting（IK）；④ 底层控制器把末端目标转关节力矩。

**为什么 Apple Vision Pro**：① 手部跟踪毫米级，SDK 直出 6-DoF；② 眼动 + 视野追踪，机器人头部跟随，沉浸感高；③ 延迟 < 50 ms，比开源 OpenVR 稳。

**典型成果**：MIT 操作员远程控制 UCSD 的 Unitree H1 抓物——推动了一波 humanoid teleop 工作。

**易错**：主要解决"上身"teleop——腿部仍由 RL/MPC 自主控制，操作员不直接控腿；这是与 Mobile ALOHA "全身遥操"的关键差别。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q30</b> · 人形 teleop 的 motion retargeting 怎么做？SMPL 到机器人骨架的尺寸校准为什么麻烦？</summary>

**答**：**Motion retargeting**：把人体动作映射到机器人关节空间，输出可执行关节目标。

**两条主流路线**：① **几何/IK 法**：机器人骨架做 IK 匹配末端位置，SEW-Mimic 用闭式几何解；② **学习法**：神经网络从人体姿态回归机器人关节角，GAN/VAE 都用过。

**SMPL 校准的麻烦**：
- SMPL 是 24 关节 + 形状参数 $\beta$（10 维），骨长由 $\beta$ 决定
- 机器人骨长固定，关节定义也不同（多数没腰 yaw / 锁骨）
- 直接复制关节角不行——同样肩角，人抬 90° 机器人可能 70°（限位 + 骨长）
- 标准做法：用 12 个对应关节优化 $\beta$ 让 SMPL 体型最接近机器人，再复制关节角

**易错**：忽略骨长差异直接 retarget 会让"摸鼻子"变成"机器人手在面前 30 cm"——这是 H2O / HumanPlus 强调先做体型对齐的原因。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q31</b> · Mobile ALOHA / ALOHA 2 的 leader-follower 双臂遥操方案为什么比 VR 抓握主流？</summary>

**答**：**Leader-follower**：操作员手握两支小臂（leader），实时同步映射到机器人本体大臂（follower）；50 Hz 流 14-DoF 关节角。

**主流原因**（桌面操作场景）：① **0 retargeting**：leader / follower 关节同构，直接复制角，无需 IK；② **天然力反馈**：leader 用同款电机，能感受 follower 接触阻力；③ **数据质量高**：精细操作（穿针、扣纽扣）远超 VR 手势；④ **成本低**：套件约 $32K，比工业方案便宜一个数量级。

**Mobile ALOHA**：底盘装轮 + 全身 teleop（推车 + 手控双臂）；50 demos 协同训练，扫地 / 炒菜成功率 80%+。

**VR 抓握的位置**：人形上身（OPEN-TeleVision）；leader-follower 适合双臂工作站。两者不冲突，是不同 form factor 的最佳实践。

**易错**：Mobile ALOHA "低成本"是相对工业级——开源套件仍约 $32K，不是"几千块业余项目"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q32</b> · H2O / OmniH2O 的训练流程：人体动作怎么变成机器人能执行的可行轨迹？</summary>

**答**：H2O / OmniH2O（CMU LeCAR Lab, arXiv 2403.04436）从大规模人类动作（AMASS / Motion-X）训人形模仿任意动作的策略。

**流程**：① Retargeting：SMPL → 机器人骨架；② 可行性过滤：privileged imitation policy（教师）跟踪每条动作，跟不上的丢弃（剔除约 20-30%）；③ 教师-学生蒸馏：cleaned 数据集训学生，输入 SMPL keypoints + 本体感觉，输出关节力矩。

**部署**：实时输入操作员 SMPL keypoints（VR / 视觉 pose），策略出关节动作。

**OmniH2O 升级**（CoRL 2024, arXiv 2406.08858）：通用接口（VR / RGB / 语言）+ dexterous 全身 teleop + 自主策略 + OmniH2O-6 数据集。

**易错**：H2O 不是零样本 retargeting——先要大量人类动作训策略；罕见动作仍需重训。与 Mobile ALOHA 的 demo→BC 路径完全不同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q33</b> · teleop 数据采集为什么对模仿学习这么关键？数据飞轮里它处于什么位置？</summary>

**答**：模仿学习 / VLA 完全靠演示数据，质量直接决定策略上限——teleop 是大规模收集"高质量带因果性"演示的事实标准。

**为什么 teleop > 视频 / 动捕**：
- **视频**：缺力交互信号（不知接触瞬间力）
- **动捕**：缺机器人本体感觉（人关节 ≠ 机器人关节）
- **teleop**：直接在目标机器人上演示，观测+动作天然对齐

**数据飞轮位置**：teleop demo → 训 BC / Diffusion Policy → 真机 rollout → 失败接管 → 追加 demo → 再训。teleop 是飞轮的种子 + 燃料。

**典型成本**：单 task 200-500 demos 起步；每 demo 1-3 分钟；π0 / GR00T 一次训需 1000+ 小时多任务 teleop。

**实务**：质量 > 数量——10 完美 demos > 100 抖动 demos；操作员熟手再录。Mobile ALOHA / UMI / DROID 是著名数据集。

**易错**：teleop 不能涵盖"机器人自遇失败状态"——DAgger / 真机 RL 在飞轮后期仍不可或缺。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q34</b> · 闭环 teleop 中的延迟（视觉反馈 + 控制下行）一般多少？怎么补偿？</summary>

**答**：典型闭环延迟（OPEN-TeleVision 类系统实测）：

| 环节 | 延迟 |
|---|---|
| 相机采集 + 编码 | 15-30 ms |
| 网络到 VR | 20-50 ms（WiFi）/ 10-20 ms（有线） |
| VR 渲染 | 10-20 ms |
| 操作员反应 | 50-200 ms |
| VR → 服务器 → 机器人 | 10-30 ms |
| 控制器响应 | 5-10 ms |
| **总闭环** | **~150-300 ms** |

**补偿手段**：① 机器人本体 1 kHz 自主（WBC、平衡），人只负责高层目标；② 预测补偿——按命令推断 100 ms 后位置，用预测态做控制；③ 视频压缩（H.265 低延迟 profile）降到 30 ms 以下；④ 同地部署，避免广域网。

**易错**：延迟不能完全消除——人反应时间是物理极限；好的设计是让人**感觉不到延迟**（VR 浸入 + 机器人本体局部自主），不是真把延迟降到 0。

</details>

## §6 Loco-manipulation / 上下身解耦（5 题）

> 移动 + 抓取的耦合任务：腿的动会扰动臂，怎么处理是 2025-2026 人形论文的热点。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q35</b> · HumanPlus 的 shadowing + imitation 两阶段方案是怎么走的？</summary>

**答**：HumanPlus（Stanford, arXiv 2406.10454）目标是"机器人实时跟一个人的动作，再从这些 shadowing 数据学自主执行任务"。

**两阶段**：① **Shadowing**：操作员做动作 → RGB pose estimation → retarget → 机器人模仿（10 Hz）。底层 Humanoid Shadowing Transformer 接 32 维姿态出 19 维关节命令；② **Imitation**：shadowing 演示当 demo，训 HIT（类 ACT）出动作 chunk，机器人自主执行。

**为什么这套**：① shadowing 只要 RGB 门槛低；② 同时拿到"人体动作 + 机器人本体感觉"对齐数据；③ 跑跳拳击抓物都验证过。

**真机**：Unitree H1，几十 demo 后自主穿鞋、写字、抓物。

**易错**：shadowing 不等同 OPEN-TeleVision——前者"机器人跟人模仿"（无 VR），后者"沉浸式控机器人"（有 VR）；用途不同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q36</b> · ExBody 为什么解耦上下身控制？上身跟模仿、下身只跟踪速度有什么取舍？</summary>

**答**：ExBody（UC San Diego, arXiv 2402.16796）观察到：整体模仿人体动作太难——下身要平衡、上身要做表达动作，目标冲突。

**解耦方案**：上身跟踪 retargeted 人体动作（精确模仿），下身只跟踪速度命令（forward / yaw），允许步态自适应。

**为什么 work**：上身模仿对**平衡稳定性要求低**——不极端可容错；下身平衡对**末端动作精度要求低**——走对方向就行。两者独立优化，难度大幅下降。

**取舍**：
- 优势：训练简单、sim2real 稳、表现力强（边走边跳舞）
- 局限：上下身完全独立无法做全身高动态动作（如蹲下抓地物）；后续 ExBody2 / ASAP / KungfuBot 引入耦合改进

**易错**：解耦不等于上下身没耦合——下身仍承受上身反作用力；只是控制器不直接同时优化两者。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q37</b> · Loco-manipulation 任务里，腿的运动会扰动臂的精度——主流怎么处理这种耦合？</summary>

**答**：人形 / 移动机械臂的 loco-manipulation 中，腿（locomotion）和臂（manipulation）有强耦合：
- 走路时躯干周期性晃动 → 末端轨迹有几厘米抖动
- 臂伸出时重心前移 → 腿要补偿

**主流处理思路**：
1. **分层 + 通信**（最常见）：locomotion 控制器把"未来质心轨迹"广播给 manipulation 控制器，manipulation 用这个轨迹做 feed-forward 补偿。如 ANYmal 的 multi-task WBC
2. **联合 QP**：把臂末端跟踪和腿落脚同写进一个 WBC QP，让优化器自动找折中（计算开销大，但精度高）
3. **学习联合策略**：用 RL 学一个端到端"全身策略"，省掉显式建模耦合。如 OmniH2O、HumanPlus
4. **解耦但加补偿**：上身用 IK 时把 IMU 测到的躯干姿态变化做实时补偿

**实务**：产线多用方案 1+4 组合（成熟、可调试）；研究多用方案 3（端到端、效果好）。

**易错**：不要假设手臂用 PD 跟踪就万事大吉——走路时关节震动会激起手臂二阶振动，需要主动补偿，否则末端轨迹完全飞掉。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q38</b> · 人形 stand-up 起立 policy（HoST）和 walking policy 为什么分开训？</summary>

**答**：HoST（SJTU + Shanghai AI Lab + HKU, arXiv 2502.08378）是首个让人形在多姿态下自主起立的工作（Unitree G1 上验证）。

**为什么不和 walking 一起训**：
1. **状态空间不同**：起立从 lying / kneeling / leaning 等极端姿态，walking 从直立——一个策略两边都做难度爆炸
2. **奖励冲突**：起立要"低→高"渐进奖励，walking 要维持高姿态，混训不收敛
3. **课程顺序**：先训 walking（基础），起立训练时 walking 已稳，可以当结束条件用

**HoST 核心技巧**：
- 把起立分阶段（lying → kneeling → squat → stand），各阶段单独奖励 + 自动 curriculum
- 训练时随机化跌倒姿态（不只平躺）
- 部署时跌倒检测 → 切起立 policy → 站起后切回 walking

**易错**：产线"自动起立"（如宇树 G1）多用两段式 + FSM 调度——不是单一全能策略；面试说"一体化全能策略"不准确。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q39</b> · 端到端 VLA 接管人形的上下身控制可行吗？目前主流为什么还是分层（高层语义 + 底层 RL/MPC）？</summary>

**答**：理论上 VLA 可端到端控人形关节力矩，但 2026 年**几乎没有产品这么做**——主流仍是分层。

**端到端难点**：① **频率不匹配**：VLA 5-10 Hz vs 关节力矩 ≥ 200 Hz；② **数据稀**：OpenX / DROID 主要是机械臂，腿足全身 teleop 数据规模小；③ **稳定性兜底**：VLA 错一步可能让人形摔倒（成本几十万），机械臂错只是抓不到，容错差巨大；④ **物理可行性**：VLA 不内置浮动基座动力学，输出可能违反摩擦锥 / 关节限位。

**主流分层**：VLA / LLM（1-5 Hz）出语义目标 → locomotion policy（50-200 Hz）出关节命令 → WBC + 力矩控制（1 kHz）跟踪 + 物理保护。

**前沿**：GR00T-N1 / Figure Helix（快慢双系统）把 VLA 与底层 control 分离，语义层不直接接关节；ASAP / HoST 把特定技能做成可调用 RL 模块。

**易错**："VLA 完全替代传统控制"是 2024-2025 媒体说法；2026 年实际仍分层——VLA 决定"做什么"，腿足底层执行"怎么做"。

</details>

## §7 Sim-to-Real for legged（6 题）

> 仿真训出来不能上真机是腿足 RL 最大的坑：actuator network、domain randomization、sim-to-sim 验证是三件套。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q40</b> · 域随机化 domain randomization 在腿足 sim2real 里怎么用？哪些参数必须随机？</summary>

**答**：**域随机化（DR）**：训练时随机化仿真器参数，让策略学到对参数分布鲁棒的特征——部署到真机时把真机当作"分布中的一个样本"。

**腿足必须随机化的参数（高优先级）**：
1. **质量 / 惯量**：±20-30%（电池电量、负载变化）
2. **摩擦系数**：脚 - 地接触摩擦 0.4-1.2（不同地面）
3. **电机参数**：扭矩限制、PD 增益（actuator delay）
4. **观测噪声**：IMU 漂移、关节编码器抖动
5. **外部扰动力**：随机推力 50-100 N，持续 0.1-0.3 s

**次优先级**：
- 关节摩擦 / 阻尼、连杆质心偏移
- 地面接触刚度（软泥 / 硬地）
- 地形高度噪声（凹凸不平）

**实务范围设定**：从真机标称值 ±20% 起步，policy 不收敛就降到 ±10%；可部署后再扩展。

**易错**：DR 不是越宽越好——randomization 过大会让 policy 学不出明确策略（变保守的 "走得最慢但永远不摔"）；与 curriculum 配合，从窄到宽逐步扩展是惯例。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q41</b> · Hwangbo 等人的 actuator network 解决了什么问题？为什么单纯随机化电机参数不够？</summary>

**答**：Hwangbo et al, Science Robotics 2019（ANYmal 上验证），是腿足 sim2real 奠基工作之一。

**问题**：仿真器电机简化为"指令 → 力矩"线性模型，真实电机有：① 谐波减速器非线性摩擦 / 间隙；② 电流环动态（ms 级低通）；③ 编码器延迟；④ 高速扭矩-速度耦合。单纯随机化 PD 增益 + 力矩限制抓不到这些——sim 中走得好的策略到真机抖。

**Actuator Network**：在真机记录大量 (位置/速度/指令 → 实际力矩) 数据，训 MLP 拟合，然后**换掉仿真器的电机模型**。

**效果**：ANYmal 学到敏捷动态技能（如跑跳、从倒地起身），sim 训练后直接零样本部署到真机——2019 年的标杆结果之一。

**与 DR 关系**：互补。actuator net 抓**确定性**非线性，DR 抓**随机性**变化（样机差异、温度漂移）；两者并用是腿足 sim2real 黄金组合。

**易错**：actuator network 每代硬件要重训——换电机或减速器后旧网络失效，是 product life cycle 的 hidden cost。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q42</b> · IsaacGym → IsaacLab → MJX / MuJoCo Playground 这条仿真器演进路线，每一代解决了什么痛点？</summary>

**答**：

| 仿真器 | 解决了什么 |
|---|---|
| **IsaacGym**（2021-2024 EOL） | GPU 并行（数千 envs），RL 训腿足从天降到小时 |
| **IsaacLab**（2024-） | 接班 IsaacGym；物理后端可换 Newton；统一 IsaacSim 渲染 |
| **MJX / MuJoCo Playground**（2024-） | JAX-MuJoCo，**全 GPU + 全开源**，不绑 NVIDIA |

**痛点演进**：IsaacGym 解算力但视觉弱 + 接触粗糙 + 私有 PhysX；IsaacLab 解视觉 + 资产但仍绑 NVIDIA；MJX 解开源但工业级视觉仍弱。

**实务选择**：腿足本体感觉 RL → MJX；视觉 + 多机器人 + 工业部署 → IsaacLab；老项目 → IsaacGym（新项目不建议，已 EOL）。

**易错**：legged_gym 等老仓库迁到 IsaacLab 不是无痛，资产格式 / API 都有 breaking changes。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q43</b> · 仿真到真机一般会先做 sim-to-sim（如 IsaacGym → MuJoCo）验证，为什么这一步很关键？</summary>

**答**：sim-to-sim（在第二个不同仿真器里跑同一策略）是 sim2real 前的"廉价金丝雀"。

**为什么有用**：① 接触模型差异——IsaacGym（PhysX）和 MuJoCo（软接触）对接触建模不同，策略只在一个仿真器 work 说明对仿真细节过拟合；② 零成本验证，先发现问题；③ 跨仿真器表现是 sim2real readiness 的代理指标。

**Humanoid-Gym 等流程**：训练 IsaacGym（4096 envs 快）→ 验证 MuJoCo（接触模型不同）→ 真机 Unitree H1/G1。

**典型失败模式**：sim1 跑得好 → sim2 摔 → 必然 sim2real 失败。常见原因：接触延迟假设不一致；学到特定接触模型的 trick（利用 PhysX 不准做能量回收）；关节限位边界处理不同。

**易错**：sim-to-sim 不能替代 actuator network / DR——它只验证"策略对仿真细节的鲁棒性"，不能补偿真机的非建模动力学。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q44</b> · 腿足机器人感知 locomotion（perceptive）里 height map / elevation map / depth camera 各自的取舍？</summary>

**答**：

| 表示 | 优势 | 劣势 |
|---|---|---|
| **Height map** | 紧凑（16×16）、维度低 | 离散化误差、遮挡需 inpaint |
| **Elevation map** | 历史融合、抗短时遮挡 | 计算 / 内存开销大 |
| **Depth 原始** | 最丰富、保留 3D | 数据量大、样本需求高 |

**主流选择**：ANYmal / Unitree → LiDAR + height map（PIM）；Boston Dynamics Spot → 双目 + elevation map；GA-PHL → 下方 depth + U-Net 实时重建 height map。

**为什么不用 RGB**：腿足策略对几何（高度）敏感，对颜色不敏感——depth 信息密度匹配需求。

**易错**：height map 不是单帧 depth——而是投影到机器人下方网格后的图；坐标变换 + 遮挡处理是工程难点，新手常忽略。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q45</b> · 触地接触瞬间的非连续性怎么处理？仿真器里软接触 vs 硬接触的取舍？</summary>

**答**：脚接触瞬间速度 / 力跳变——仿真和真机最容易不一致的地方。

**两种接触模型**：

| | 硬接触 / 冲量 | 软接触 / 罚函数 |
|---|---|---|
| 形式 | LCP / NCP | $f = k\delta + b\dot\delta$ |
| 代表 | Bullet、ODE、PhysX | MuJoCo |
| 优势 | 无穿透、物理正确 | 快、可微 |
| 劣势 | 解器迭代多、冲量模型速度非连续 | 有可感知穿透 |

**MuJoCo 特殊做法**：soft constraint with stiff parameters——可调到接近刚体但仍数学连续，求解快且稳；这是 MuJoCo 成 RL 主流的关键。

**腿足 sim2real 实务**：① 软接触穿透会让策略学"利用穿透做能量回收"trick → 真机失败；② 接触刚度调到 1e5-1e7 N/m；③ 加 actuator network 补差异。

**易错**：MuJoCo 默认软接触对快速跑跳偏弹——训练后策略会"利用弹力"；要主动调硬或加阻尼，否则真机零样本失败。

</details>

---

## 低频备选（频次 1-2，未入主表）

> 以下题目出现频次低（1-2 次），供参考，未纳入主表题号。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N1</b> · 谐波减速器 vs 行星减速器：人形哪些关节用哪种？为什么？</summary>

**答**：人形关节的两类主流减速器：

| | 谐波减速器（harmonic） | 行星减速器（planetary） |
|---|---|---|
| 减速比 | 高（30-200:1） | 低-中（3-30:1） |
| 体积 / 重量 | 紧凑、轻 | 体积大 |
| 反向间隙 | 几乎无 | 有（约 5-15 弧分） |
| 精度 | 高 | 中 |
| 成本 | 高（哈默纳科 / 绿的等） | 低 |
| 寿命 | 5000-10000 小时 | 较短，依赖润滑 |

**人形典型选型**：
- **小臂 / 腕 / 手**：谐波减速器（精度高、轻量）—— 特斯拉 Optimus 旋转关节用谐波 + 框架电机
- **大腿 / 髋部**：常用大功率行星 + QDD（quasi-direct drive）方案，要承载高动态冲击
- **膝盖**：两种都见——力大用行星，对精度要求高用谐波

**易错**：QDD（直驱风格的小减速比电机）正在取代部分关节的谐波——它牺牲精度换抗冲击和透明力控；Mini-Cheetah、宇树系列大量使用 QDD。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N2</b> · 串联弹性致动器（SEA）和直驱（QDD）在腿足里各有什么优劣？</summary>

**答**：

| | SEA（串联弹性） | QDD（准直驱） |
|---|---|---|
| 结构 | 电机 + 减速器 + 弹簧 | 大扭矩电机 + 小减速比（5-10:1） |
| 力控 | 测弹簧形变得力矩 | 反推电流估力矩 |
| 抗冲击 | 弹簧吸收 | 电机本身被动柔顺 |
| 控制带宽 | 受弹簧动态限制 | 高带宽（直接控） |
| 代表 | NASA Valkyrie、IIT COMAN、ANYmal 腿部 | MIT Mini-Cheetah、宇树 Go / G1、Solo |

**SEA 历史**：2010-2020 是 NASA Valkyrie / IIT COMAN 等电驱人形的主流（ATLAS 液压、ASIMO 位置伺服皆非 SEA）。

**QDD 接管原因**：① 永磁同步电机 + 高磁链让小减速比也能给足扭矩；② 无串联动力学、控制直接、带宽高；③ Mini-Cheetah 2018 证明跑跳可行；④ 维护简单。

**易错**：SEA 不是过时——KUKA iiwa 类协作机械臂内部仍用 SEA 做柔顺；QDD 是腿足主流，但不是机器人界唯一答案。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2</span> <b>N3</b> · 为什么轮足机器人控制器主流是 LQR，而四足是 MPC？两者的本质区别是什么？</summary>

**答**：轮足 = 轮 + 双腿（Ascento 等），本体是连续接触的倒立摆。

**轮足偏 LQR**：① 轮子持续接触，**无接触模式切换**，动力学连续；② 简化后接近线性倒立摆，LQR 一次离线解 Riccati 就够；③ 实时算力极低。

**四足偏 MPC**：① 接触模式频繁切换（trot / bound），LQR 跨切换无能；② 摩擦锥、力上限是**硬约束**，LQR 无法处理；③ MPC 滚动重规划可处理 contact schedule。

**本质区别**：轮足 = 状态空间稳定问题（经典控制）；四足 / 人形 = 约束 + 切换的混合系统（必须 MPC / 学习）。

**前沿**：人形 RL policy 越来越像"广义 MPC"（隐式优化 + 网络拟合），LQR / MPC / RL 边界在融合（RL-augmented MPC）。

**易错**：不能说"LQR 落后"——简单系统下 LQR 解析 + 可调 + 可证明稳定性仍是工业首选；MPC 在复杂系统不可替代但不是万能更优解。

</details>

---
