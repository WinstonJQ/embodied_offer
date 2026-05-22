# 卷二 · RL 算法 · 高频面试题

> 中文具身智能秋招高频面试题库 · **第二卷**
> 题源：牛客 / 知乎 / CSDN / GitHub 公开面经 / 强化学习算法岗面试汇编
> 同义题合并后 **43 题**（主表 40 题频次 ≥3 + 低频备选 3 题）

**难度分布**：L1（必会） **14** · L2（进阶） **22** · L3（顶级难度） **7**

**使用方式**：题目默认折叠，点开看答案。建议按 **L1 → L2 → L3** 顺序刷；同级内按频次从高到低。手机端原生支持。

---

## §1 策略优化基础（8 题）

> on-policy / off-policy 分野 · 策略梯度定理 · Actor-Critic 框架 · 优势函数。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×15</span> <b>Q01</b> · on-policy 和 off-policy 的区别？各自有哪些代表算法？</summary>

**答**：
- **on-policy**：用**当前策略** $\pi_\theta$ 收集的数据训练，数据用完即丢；策略每次更新后必须重新采样。代表：REINFORCE / A2C / A3C / **PPO**（严格说 PPO 通过重要性采样做 mini off-policy，但仍属 on-policy 范畴）。
- **off-policy**：可以用**任意策略**（如旧策略、人类演示）收集的数据训练，借助经验回放重复利用；探索策略与优化策略可分离。代表：Q-learning / DQN / DDPG / **TD3** / **SAC**。

**关键区别**：on-policy 样本效率低但稳定；off-policy 样本效率高但收敛难（Q 值过估计、分布漂移）。

**易错**：SAC 是 off-policy，PPO 虽然用了重要性采样，但用旧数据的 epoch 数很少（通常 3-10 epoch），学术上仍归 on-policy；不要说"PPO 是 off-policy"。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q02</b> · 策略梯度定理是什么？为什么要减去 baseline？</summary>

**答**：**策略梯度定理**：$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$，其中 $G_t$ 是从时刻 $t$ 起的累积回报。

**为什么减 baseline**：$G_t$ 方差极大（随机轨迹可达 $\pm$几百），导致梯度噪声大、收敛慢。减去与动作无关的 baseline $b(s_t)$（常取 $V^\pi(s_t)$）后：
- **不影响期望**（$b$ 与动作无关，减去后梯度期望不变）
- **方差显著降低**：$G_t - V(s_t) = A_t$（优势函数），只保留"这个动作比平均好多少"的信号

**易错**：baseline 必须与动作 $a_t$ 无关，否则会引入 bias；用 $V(s_t)$ 作 baseline 是最常见选择。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q03</b> · Actor-Critic 框架是什么？与纯 Policy Gradient 的区别？</summary>

**答**：**Actor-Critic (AC)** 同时维护两个网络：
- **Actor**（策略网络）：$\pi_\theta(a|s)$，负责选动作；
- **Critic**（值函数网络）：$V_\phi(s)$ 或 $Q_\phi(s,a)$，负责评估动作好坏。

Actor 用 Critic 的输出（advantage 或 TD 误差）代替 MC 累积回报 $G_t$ 来更新，**把 baseline 从 V 变成实时 TD 估计**，大幅降低方差。

**vs 纯 PG（REINFORCE）**：
- REINFORCE 需等轨迹结束才能计算 $G_t$，高方差、慢收敛；
- AC 可在线（每步）更新，方差更低，但 Critic 引入 bias（若 $V_\phi$ 估不准）。

**易错**：AC 的"两网络"说法是概念层面；实际实现中 Actor 和 Critic 常共享前几层特征，只在输出头分叉。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q04</b> · GAE（Generalized Advantage Estimation）是什么？λ 怎么控制 bias-variance 权衡？</summary>

**答**：**GAE** 在 MC 高方差 ($\lambda=1$) 与 TD 低方差但有 bias ($\lambda=0$) 之间用指数加权插值：

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\lambda \to 1$：接近 MC，方差大、bias 小；
- $\lambda \to 0$：接近单步 TD，方差小、bias 大；
- 实务甜区：**$\lambda \approx 0.95$, $\gamma \approx 0.99$**。

**PPO 与 GAE**：PPO 的策略梯度目标 $\mathbb{E}[\min(r_t\hat{A}_t, \text{clip}(r_t,1\pm\varepsilon)\hat{A}_t)]$ 需要 $\hat{A}_t$，GAE 提供低方差的 $\hat{A}_t$ 估计，是 PPO 稳定收敛的关键。

**易错**：GAE 不是 PPO 专属，TRPO/A3C 也用；但 PPO + GAE 是当前最常见组合。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q05</b> · A3C 和 A2C 的区别？为什么 A3C 提出时被认为是突破？</summary>

**答**：
- **A3C（Asynchronous Advantage Actor-Critic, 2016）**：多个**异步** worker 各自与独立环境交互，将梯度推送给全局网络，不需要经验回放；去相关化的异步更新相当于天然正则。DeepMind 原作在 CPU 集群上效果极好。
- **A2C**：**同步**版本，等所有 worker 完成一批再统一更新；实现更简单，在 GPU 上 batch 更友好，实际性能常与 A3C 持平或略强。

**A3C 突破点**：① 证明异步并行可替代经验回放去相关化；② 无 replay buffer 降低内存；③ on-policy 不需要重要性采样。

**当前地位**：A3C/A2C 已基本被 PPO 取代，但仍是理解 Actor-Critic 并行训练的经典范例。

**易错**：A3C 不是 off-policy，仍是 on-policy；它的"异步"指的是数据收集异步，不是说能复用旧数据。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q06</b> · TD 误差（TD error）是什么？和 Bellman 方程有什么关系？</summary>

**答**：**TD 误差**（时序差分误差）：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，表示"实际得到的 bootstrap 估计"与"当前估计"之差。

**与 Bellman 方程的关系**：Bellman 期望方程 $V^\pi(s) = \mathbb{E}[r + \gamma V^\pi(s')]$，若 $V$ 满足 Bellman 方程，则 $\mathbb{E}[\delta_t] = 0$；TD 学习就是用随机梯度把 $\delta_t$ 向 0 推。

**核心作用**：① 提供单步可计算的 critic 更新信号（不需等轨迹结束）；② 等于 GAE 公式里的 $\delta_{t+l}$ 基本单元；③ Prioritized Replay 用 $|\delta_t|$ 衡量样本重要性。

**易错**：$\delta_t$ 是带符号的误差，正值表示"实际比预期好"、负值表示"实际比预期差"；直接用 $\delta_t^2$ 做损失就是 critic 的 MSE loss。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q07</b> · 重要性采样（Importance Sampling）在 RL 中的作用？PPO 怎么用的？</summary>

**答**：**重要性采样（IS）**：当数据分布 $q$ 与目标分布 $p$ 不一致时，用比值 $w = p(x)/q(x)$ 纠正：$\mathbb{E}_p[f(x)] = \mathbb{E}_q[w \cdot f(x)]$。

**RL 中的用途**：① 允许 on-policy 算法用旧策略数据做多轮 mini-batch 更新（提升数据效率）；② off-policy 评估（用行为策略数据评估目标策略）。

**PPO 怎么用**：定义比值 $r_t(\theta) = \pi_\theta(a|s) / \pi_{\theta_{\text{old}}}(a|s)$，梯度目标变为：
$$L^{\text{CLIP}} = \mathbb{E}_t\!\left[\min\!\left(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$
**Clip 机制**不是 IS 本身，而是限制 $r_t$ 偏离 1 太远（$\varepsilon$ 通常 0.2），防止比值失控、方差爆炸。

**易错**：PPO 的 clip 只影响梯度计算，不是完整 IS 校正；当 $r_t$ 被 clip 后梯度等效为 0，策略不会继续沿该方向走远。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q08</b> · 多步回报（n-step return）是什么？和 TD(0) / MC 的关系？</summary>

**答**：**n-step return**：$G_t^{(n)} = r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})$，即向前展 n 步后接 bootstrap。

**关系**：
- $n=1$：单步 TD（TD(0)）；低方差、高 bias（V 估计不准影响大）
- $n \to \infty$：蒙特卡洛（MC）；高方差、零 bias（用真实 G）
- 中间值：bias-variance 权衡；$\lambda$-return（GAE 是其指数加权版本）

**实务选 n 的经验**：短任务（episode < 50 步）可用较大 n；长任务（> 500 步）n 过大会导致方差爆炸，通常 n=3-10 + GAE。

**易错**：TD(λ) 和 GAE 的关系——GAE 是从 offline 轨迹计算的 λ-return advantage；TD(λ) 是 online 更新规则；计算方法不同，但在正向累积 δ 的公式上等价。

</details>

---

## §2 PPO / TRPO（5 题）

> on-policy 主流，RLHF/VLA 标配。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×18</span> <b>Q09</b> · PPO 的 Clip 机制是什么？为什么 ε 通常取 0.2？</summary>

**答**：**PPO（Proximal Policy Optimization, Schulman 2017）**的核心是 Clip 目标：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t\hat{A}_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

其中 $r_t = \pi_\theta / \pi_{\theta_\text{old}}$。

**直觉**：advantage 为正时想升高 $r_t$，但若 $r_t > 1+\varepsilon$ 已涨太多，clip 截断梯度，防止过激更新；advantage 为负时同理，$r_t < 1-\varepsilon$ 则截断。"先设上下限、再取 min"确保保守更新。

**ε = 0.2**：Schulman 原论文实验甜区；太小更新保守、样本浪费；太大接近无约束 PG。RLHF 常用 0.1-0.2。

**易错**：PPO-Clip ≠ PPO-KL；后者在损失里显式加 KL penalty；前者更常用且稳定。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×14</span> <b>Q10</b> · TRPO 和 PPO 的联系与区别？为什么 PPO 更流行？</summary>

**答**：**TRPO（Trust Region PO, Schulman 2015）**：用二阶优化（共轭梯度 + Hessian-vector 乘积）在**KL 约束** $D_\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \le \delta$（旧策略到新策略的 KL）下求最优更新；理论保证单调策略改进。

**PPO**：TRPO 的轻量化替代，用 clip 机制隐式限制策略变化，**不需要共轭梯度**，一阶优化（Adam）即可；代码量从数百行降到数十行。

| | TRPO | PPO-Clip |
|---|---|---|
| 约束方式 | 显式 KL 约束 | Clip ratio |
| 优化器 | 二阶（CG+线搜） | 一阶（Adam） |
| 代码复杂度 | 高 | 低 |
| 效果 | 理论保证单调 | 实务近似等效 |
| 分布式 | 难 | 容易 |

**PPO 更流行的原因**：① 实现简单；② GPU batch 友好；③ 效果接近 TRPO；④ 与 LLM RLHF 对齐流程无缝集成（VLA 微调主流仍是 BC/flow matching）。

**易错**：TRPO 的"单调改进"在神经网络参数化下只是近似保证（Fisher 矩阵近似），并非绝对。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q11</b> · PPO 的 entropy bonus 是什么？什么时候需要加？</summary>

**答**：**Entropy bonus**：在 PPO 目标里加 $\beta \cdot \mathcal{H}[\pi_\theta(\cdot|s)]$（策略熵），鼓励策略保持随机性。

**直觉**：若策略过早收敛（熵趋 0），actor 只会走一条路，探索停止——这在稀疏奖励 / 多峰任务中灾难性。熵 bonus 提供额外梯度驱动策略"多尝试几种动作"。

**什么时候用**：① 奖励稀疏（初期没信号，需探索维持）；② 离散动作空间（熵易计算 $-\sum p \log p$）；③ 任务有多条最优路径（不想早期锁死一条）。

**什么时候不加**：连续控制精细任务（机械臂插针）——过多探索引入噪声反而有害；SAC 已内置熵最大化，不需要额外加。

**超参 $\beta$ 建议**：通常 0.01-0.05；RLHF 场景有时用 0.001 以下（微调阶段需保守）。

**易错**：entropy bonus 是可选项，PPO 原论文的联合目标中包含该项，但实验中系数可设为 0；SAC 的熵机制来自 max-entropy RL 框架（正则化目标），与 PPO 的 entropy bonus（探索辅助）不要混淆。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q12</b> · PPO 训练的完整流程是什么（采样 → 优势估计 → 多轮更新 → 截断）？</summary>

**答**：**PPO 一个迭代周期**：

1. **Rollout**：用 $\pi_{\theta_\text{old}}$ 交互 T 步，存 $(s_t, a_t, r_t, s_{t+1}, \log\pi_\text{old})$。
2. **GAE**：计算 $\hat{A}_t$ + target value $\hat{V}_t = \hat{A}_t + V_\phi(s_t)$。
3. **多 epoch 更新**：mini-batch 采样 K 次（K=3-10），计算 Actor $L^{\text{CLIP}}$ + Critic MSE + entropy bonus（可选）。
4. **Early stopping**：KL 超阈值时提前退出 epoch。
5. 更新 $\theta_\text{old} \leftarrow \theta$，重复。

**关键超参**：T（256-2048）、batch（32-256）、K（3-10）、ε（0.2）、λ（0.95）。

**易错**：多 epoch 是 PPO 核心优势；但 K 太大 → IS ratio 失控；K 通常不超过 10。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q13</b> · PPO 在机器人连续控制中常见的失败模式有哪些？如何诊断？</summary>

**答**：
**常见失败模式**：

1. **Reward hacking / mode collapse**：奖励函数有漏洞，agent 找到"不物理"的高分动作（如原地振荡换高速度奖励）。诊断：reward 高但实际任务失败；对策：奖励工程审查 + 额外约束项。

2. **训练不稳定（loss 剧烈震荡）**：学习率或 clip ε 太大；Critic 估计跟不上 Actor 更新节奏。诊断：entropy 先升后崩、KL 突增；对策：降 lr、减 epoch 数、共享 encoder 冻层。

3. **过早收敛到次优策略**：entropy 过快降至接近 0，探索停止。对策：增大 entropy bonus $\beta$ 或使用 curriculum。

4. **仿真 sim2real gap**：仿真里 PPO 高分，真机泛化差。诊断：Domain Randomization 覆盖不足；对策：增大 DR 范围 + 感知扰动。

5. **稀疏奖励不收敛**：长期无信号，梯度消失。对策：奖励 shaping（如势能函数）+ HER（Hindsight Experience Replay）。

**易错**：PPO 失败首先看 entropy 曲线——熵崩塌是早期预警信号；不要先调 clip ε，先检查奖励函数和数据分布。

</details>

---

## §3 SAC / TD3 / DDPG（5 题）

> off-policy 连续控制三巨头；机器人 sample efficiency 首选。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×14</span> <b>Q14</b> · SAC 是什么？max-entropy 框架的直觉是什么？temperature α 怎么自动调？</summary>

**答**：**SAC（Soft Actor-Critic, Haarnoja 2018）**：最大熵 RL 框架下的 off-policy Actor-Critic，目标不是最大化期望回报，而是：

$$J(\pi) = \mathbb{E}\!\left[\sum_t r(s_t, a_t) + \alpha \cdot \mathcal{H}[\pi(\cdot|s_t)]\right]$$

**Max-entropy 直觉**：最大化奖励 + 策略熵 → 策略在等效动作间均匀分布。好处：探索强、多峰奖励自然处理、扰动后鲁棒恢复。

**Temperature α 自动调节**（SAC v2）：设目标熵 $\bar{\mathcal{H}} = -|\mathcal{A}|$，优化 $\min_\alpha \mathbb{E}[-\alpha(\log\pi_\theta + \bar{\mathcal{H}})]$。熵低 → α 升（多探索）；熵高 → α 降（多利用）。初始 α ≈ 0.2，之后自适应。

**易错**：α 不是"探索率"，是正则权重；SAC 无 ε-greedy，策略随机性即探索来源；自动调 α 远优于手调，不要固定 α。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q15</b> · TD3 相比 DDPG 做了哪三点改进？分别解决什么问题？</summary>

**答**：**TD3（Twin Delayed DDPG, Fujimoto 2018）**针对 DDPG，做了 3 点改进：

1. **双 Q 网络**：维护 $Q_{\phi_1}, Q_{\phi_2}$，取 min：$y = r + \gamma \min_i Q_{\phi_i}(s', \pi(s'))$ → 解决过估计。

2. **Actor 延迟更新**：Actor 每 d 步更新一次（d=2），Critic 每步更新 → 防止 Actor 过快追高估 Q 值。

3. **Target Policy Smoothing**：target action 加裁剪高斯噪声 $\tilde{a} = \pi(s') + \text{clip}(\mathcal{N}(0,\sigma), -c, c)$ → 平滑 Q 面，防止在 Q 峰值上过拟合。

**易错**：TD3 是确定性策略，SAC 是随机策略；3 个改进必须**全部组合**才达到论文效果，缺一不可。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×9</span> <b>Q16</b> · DDPG 是什么？为什么在实务中不稳定？</summary>

**答**：**DDPG（Deep Deterministic Policy Gradient, Lillicrap 2016）**：off-policy Actor-Critic，专为连续动作设计：
- Actor 输出**确定性动作** $\mu_\theta(s)$（不是概率分布）
- Critic 用 Q-learning 更新：$y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$
- 使用**经验回放 + target 网络**（从 DQN 借来）

**实务不稳定的原因**：
1. **Q 值过估计**：actor 持续追最大 Q，Q 值单调高估，最终 NaN；
2. **超参极度敏感**：batch size / lr / 噪声 σ 调错就崩；
3. **探索困难**：确定性策略靠人为加 Ornstein-Uhlenbeck 噪声，效果有限；
4. **Critic 和 Actor 互相污染**：Q 估计不准 → 策略更新方向错 → Q 更不准（恶性循环）。

**TD3 / SAC 的出现**基本淘汰了裸 DDPG 的实际使用。

**易错**：DDPG 的"确定性"是指策略不是概率分布——推理时输出是单个动作；探索靠加外部噪声，不是策略本身的随机性。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q17</b> · replay buffer（经验回放）的作用？uniform 采样有什么缺陷？PER 怎么改进？</summary>

**答**：**Replay buffer**：存储历史 $(s, a, r, s', \text{done})$ 元组，off-policy 算法随机采样训练。

**作用**：① **去相关**：相邻时间步高度相关，随机采样打破 temporal correlation；② **数据复用**：每条数据可被多次学习，提升 sample efficiency；③ **稳定分布**：避免 online 数据分布漂移太快导致不稳定。

**均匀采样缺陷**：TD 误差大的（难学、重要的）样本和 TD 误差小的（已学好的）样本同等概率采到 → 训练低效。

**PER（Prioritized Experience Replay, Schaul 2016）**：用 $|\delta_t|^\alpha$（TD 误差绝对值的 α 次方）设优先级，高 TD 误差样本更频繁采样；同时用**重要性采样权重** $w_i = (1/N \cdot 1/p_i)^\beta$ 纠正采样 bias（$\beta$ 从 0.4 退火到 1.0）。

**易错**：PER 不是"总采最难的样本"——是概率高，不是必然采；重要性权重不是可选项，缺失会引入 bias 导致价值函数偏移。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q18</b> · target network 是什么？为什么不直接用当前网络做 TD 目标？</summary>

**答**：**Target network**：维护一份主网络的**软拷贝** $\phi'$，用于计算 TD 目标 $y = r + \gamma Q_{\phi'}(s', a')$；通过指数移动平均（EMA）缓慢更新：$\phi' \leftarrow \tau \phi + (1-\tau)\phi'$，通常 $\tau = 0.005$。

**为什么需要**：若 TD 目标用当前网络 $Q_\phi$ 计算，损失为：
$$L = (Q_\phi(s,a) - (r + \gamma Q_\phi(s',a')))^2$$
被拟合的目标 $r + \gamma Q_\phi$ **随着 $\phi$ 更新而移动**，形成"追逐运动目标"的循环——网络调一步，目标跟着变，容易发散（"自举问题"）。

**EMA 更新 vs 硬拷贝**：DQN 原版每 C 步硬拷贝；DDPG/TD3/SAC 用 EMA，更平滑、实务更稳定。

**易错**：$\tau$ 越小目标越稳，但学习越慢；$\tau = 1$ 等价于没有 target 网络（不稳定）。target 网络不是 Critic 专属，部分算法 Actor 也有 target 网络。

</details>

---

## §4 DQN 与离散动作（3 题）

> 虽然机器人用连续 RL 为主，DQN 系仍是高频考点（理解值函数根基）。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q19</b> · DQN 在 Q-learning 基础上做了哪两大改进？分别解决什么问题？</summary>

**答**：**DQN（Deep Q-Network, Mnih 2015）**在朴素 Q-learning（表格/线性）基础上：

**改进 1：经验回放（Experience Replay）**
- 问题：online 样本时间相关 → 神经网络训练不稳定
- 解法：随机从 buffer 采样，打破 temporal correlation + 重复使用数据

**改进 2：Target Network**
- 问题：TD 目标 $r + \gamma \max_{a'} Q_\phi(s',a')$ 与被拟合的 $Q_\phi(s,a)$ 用同一网络，移动目标 → 发散
- 解法：单独维护 $\phi'$（每 C 步从 $\phi$ 硬拷贝），让目标在短期内固定

**关键数字**：Atari 实验中 replay buffer 大小 10^6，target 更新频率 10^4 步，batch 32，epsilon-greedy 从 1.0 线性退火到 0.1。

**易错**：DQN 的神经网络替代的是 Q-table，不是策略；DQN 输出所有动作的 Q 值（离散动作），不输出动作概率；不能直接用于连续动作空间（DDPG 才能）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q20</b> · Double DQN / Dueling DQN / Noisy Nets：各自解决什么问题？</summary>

**答**：
- **Double DQN（van Hasselt 2016）**：解决 Q 值**过估计**（max 操作对估计误差有正偏差）。改法：选动作用当前网络 $\phi$，计算 Q 值用 target 网络 $\phi'$：$y = r + \gamma Q_{\phi'}(s', \arg\max_{a'} Q_\phi(s', a'))$。解耦"选哪个动作"与"评估这个动作"。

- **Dueling DQN（Wang 2016）**：解决 Q 值的**状态无关冗余**。把 Q 分解为：$Q(s,a) = V(s) + A(s,a)$（state value + advantage）。好处：能同时更新所有动作的 V 值（即使该 step 没采到某个动作），更新效率高。

- **Noisy Nets（Fortunato 2018）**：解决 $\varepsilon$-greedy **探索低效**问题（均匀随机探索浪费）。把网络权重换成带可学噪声的参数 $\mu + \sigma \cdot \varepsilon$（参数化探索），探索自适应地集中在不确定动作上。

**易错**：三者可叠加（Rainbow DQN = 六种改进叠加）；Dueling 的标准归一化为减**均值**：$Q(s,a) = V(s) + A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a')$，保证 V/A 分解唯一且梯度覆盖所有动作；不减任何量 → V/A 分解不唯一、不稳定。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q21</b> · HER（Hindsight Experience Replay）是什么？为什么对稀疏奖励机器人任务特别有效？</summary>

**答**：**HER（Andrychowicz 2017）**：将失败的轨迹"回头重写目标"为实际达到的结果，从而产生有效学习信号。

**流程**：机械臂试图把方块推到位置 $g$ 但失败（最终停在 $g'$），HER 将该 episode 的目标 $g$ 替换为 $g'$，**逐 transition 重新计算奖励**（$r = \mathbb{1}[\|s - g'\| < \delta]$，仅最后几步为 +1）→ 这段失败轨迹变成"成功到达 $g'$"的有效经验加入 buffer。

**为什么有效**：① 稀疏奖励下 agent 几乎得不到任何正信号，HER 把每次失败都变成"成功到达某个地方"的经验，信号密度大幅提升；② 不需要改奖励函数；③ 与任何 off-policy 算法（DDPG/TD3/SAC）无缝组合。

**适用场景**：目标条件策略（Goal-conditioned RL），如机器人 pick-and-place、推物体到目标位置等。

**易错**：HER 用的不是任意随机目标，而是从同 episode 实际达到的状态中重标（strategy = "future" / "final" / "episode"），语义上是"已经发生的真实结果"；若奖励不是 goal-conditioned（如 $r = \mathbb{1}[s=g]$），HER 无法应用。

</details>

---

## §5 Offline RL（6 题）

> 具身岗高频：为什么不能直接 off-policy？OOD 问题、CQL/IQL/BCQ/DT 对比。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×11</span> <b>Q22</b> · Offline RL 和 Online RL 的核心区别？为什么不能直接用 off-policy 算法跑 offline 数据？</summary>

**答**：**Online RL**：与环境实时交互，策略可以探索未知状态。**Offline RL（Batch RL）**：仅用预先收集的固定数据集 $\mathcal{D}$，无法与环境再交互。

**为什么普通 off-policy 不能直接用**：关键问题是**分布外（Out-of-Distribution, OOD）动作过估计**：
- Q-learning 的 Bellman 更新中有 $\max_a Q(s', a)$（或策略网络选出的 $a'$）
- 若 $a'$ 不在数据集 $\mathcal{D}$ 中（OOD），神经网络对 $Q(s', a')$ 的外推是任意高（函数近似没有约束）
- Actor 迭代地找最大 Q → 走向 OOD 区域 → Q 值正反馈爆炸 → 策略崩溃

**类比**：用餐厅菜单的数据学会"最好吃的菜"，但模型会说"菜单上没有的食材做的菜肯定最好吃"——它对未知区域外推是任意的。

**易错**：Offline RL 不等于 Behavior Cloning（BC）——BC 直接模仿数据动作，不显式学 Q；Offline RL 尝试从数据中推断最优策略（可能比数据 policy 更好），难度更高。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q23</b> · CQL（Conservative Q-Learning）的核心思想是什么？如何防止 OOD 过估计？</summary>

**答**：**CQL（Kumar 2020）**在 Q 值目标里加一个**保守惩罚**：

$$\min_\phi \mathbb{E}_{(s,a)\sim\mathcal{D}}[(Q-y)^2] + \alpha\!\left(\mathbb{E}_{s\sim\mathcal{D}, a\sim\mu(s)}\!\left[Q(s,a)\right] - \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[Q(s,a)\right]\right)$$

**直觉**：第二项**压低** OOD（随机/当前策略 μ 采样）的 Q；第三项**抬高**数据集中动作的 Q → 数据内高、数据外低 → 策略不走 OOD。

**理论保证**：CQL 的 $Q^\pi$ 是真实 $Q^\pi$ 的保守下界，在下界上优化策略是安全的。

**超参 α**：α 大 → 过保守（性能跌破 BC）；α 小 → 退化为普通 Q-learning（OOD 复现）。实务用 Lagrangian 自动调。

**易错**：CQL 不是纯 BC——它仍尝试找比 behavior policy 更好的策略，只是用保守 Q 防 OOD。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q24</b> · IQL（Implicit Q-Learning）如何在不查询 OOD 动作的情况下做 offline RL？</summary>

**答**：**IQL（Kostrikov 2021）**的关键创新是**完全避免**评估 OOD 动作：

**三个组件**：
1. **V(s)**：expectile 回归（τ≈0.9）近似最优 V 值，完全在数据动作上计算；
2. **Q(s,a)**：用 $Q = r + \gamma V(s')$ TD 更新，**不查询 $\max_{a'}$**；
3. **策略提取**：advantage-weighted 回归：$\exp(\beta A(s,a))\cdot\log\pi_\theta(a|s)$

**优势**：无 OOD 动作查询，训练稳定；在 locomotion 基准上优于或接近 CQL。

**易错**：expectile 回归 ≠ quantile 回归——前者是不对称加权期望，后者是精确分位点估计；τ 越大越激进（越接近最优 Q），不是"百分位"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q25</b> · BCQ（Batch-Constrained Q-learning）和行为克隆的区别？为什么叫"batch-constrained"？</summary>

**答**：**BCQ（Fujimoto 2019）**：在做 Q 最大化时，**只在与数据集分布相近的动作上做 max**，而不是全动作空间 max。

**核心机制**：训练一个生成模型 $G_\omega$（通常 VAE）来建模数据集的动作分布 $\beta(a|s)$，推理时：
- 从 $G_\omega$ 采样 N 个候选动作 $a_1, \ldots, a_N$
- 每个动作通过扰动网络 $\xi_\phi$ 微调（有限范围内）
- 选 $Q$ 值最大的候选 → 这些候选都在数据支持范围内

**与 BC 的区别**：BC 只克隆 $\pi_{\text{data}}$（学 $P(a|s)$），不优化 Q；BCQ 仍在优化 Q 值，只是把 max 操作限制在数据支持集内，**可以超越 behavior policy 的性能**（若 behavior policy 本身次优）。

**"Batch-Constrained"含义**：策略受到 batch 数据的约束——选动作受限于 batch 中见过的动作分布。

**易错**：BCQ 的 VAE 是建模数据分布，不是生成目标动作；扰动网络的范围用 $\Phi$ 控制，$\Phi=0$ 退化为纯 BC，$\Phi=\infty$ 退化为无约束 Q-learning。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q26</b> · Decision Transformer（DT）如何把 Offline RL 变成序列建模？有什么局限？</summary>

**答**：**DT（Chen 2021）**：把 Offline RL 重构为**条件序列预测**，用 GPT 风格 Transformer 自回归建模 $(R^g_1, s_1, a_1, R^g_2, s_2, a_2, \ldots)$，条件是 Return-to-Go（RTG）$R^g_t = \sum_{t'\geq t} r_{t'}$。

**训练**：给定历史 RTG + 状态，预测当前动作（监督学习）。**推理**：设定期望 RTG（如"我想得 1000 分"），模型自回归生成动作序列。

**优势**：① 不需要 Bellman 方程，无 OOD 过估计问题；② 直接复用 Transformer 预训练能力；③ 长程依赖建模好。

**局限**：
1. **依赖 RTG 设置**：测试时必须知道"目标 RTG"，稀疏/未知回报场景难设置；
2. **不能 stitch**：无法组合数据集中不同片段找出最优路径（Bellman 方程做的事）；
3. 在简单 D4RL benchmark 上不如 IQL/CQL，但在长序列 / 自然语言条件任务上有优势。

**易错**：DT 不做 Q 值优化，是纯监督学习；不能从次优数据提升到远超 behavior policy 的性能（这是值函数方法的优势）。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×5</span> <b>Q27</b> · AWAC / IQL offline-to-online 的流程是什么？为什么在具身机器人里受关注？</summary>

**答**：**Offline-to-Online RL**：先用大量离线数据（人工演示 / 历史轨迹）做 offline 预训练，再通过少量 online 交互精调，避免从零开始 online 探索的高昂成本。

**AWAC（Nair 2021）**：actor 更新用 $\exp(\lambda A)$ 权重（advantage-weighted 回归），offline 预训练后 online 阶段直接续跑 off-policy，无需特殊切换。

**IQL offline-to-online**：offline 用 IQL 训 V/Q/policy；online 直接继续采集 + IQL 更新；无 OOD 查询，稳定性优于 AWAC。

**具身场景价值**：真机探索代价高；遥操作演示数据丰富；offline 预训练 + 少量 online 精调是主流路线（代表：RLPD / Cal-QL）。

**易错**：不是"先 BC 后 RL"；CQL 预训练的 Q 值过保守时，online 阶段需 warm-up 解禁。

</details>

---

## §6 RLHF / DPO / GRPO（5 题）

> 具身算法岗 + 大模型岗双线高频；VLA 微调必考。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×13</span> <b>Q28</b> · RLHF 的三阶段是什么？每阶段目标是什么？</summary>

**答**：**RLHF（Reinforcement Learning from Human Feedback）**是 LLM 对齐的标准流程（VLA 微调主流仍是 BC/flow matching，RLHF 在 VLA 属新兴探索方向）：

**阶段 1：监督微调（SFT）**
- 目标：在高质量示范数据上做监督学习，让模型有基础能力
- 输出：SFT 模型（同时作为 PPO 阶段的参考模型）

**阶段 2：奖励模型训练（Reward Model）**
- 目标：学一个 $R_\phi(s, a)$，使人类对"输出 A 比 B 好"的偏好得分一致
- 数据：同一提示下多个输出 → 人类排序 → Bradley-Terry 模型训练 RM
- 关键：RM 是对齐信号的代理，质量决定 PPO 阶段的方向

**阶段 3：PPO 精调（RL fine-tuning）**
- 目标：最大化 RM 打分同时限制与 SFT 模型的 KL 散度：$r' = R_\phi(s, a) - \beta \cdot D_\text{KL}(\pi_\theta \| \pi_\text{SFT})$
- 四模型：Actor（被训）、Critic（价值估计）、Reference（SFT，冻结）、RM（冻结）

**易错**：RLHF 的 PPO 阶段不是普通 RL——Reward 来自 RM 而非环境；KL 项是必须的，不加会导致模型"奖励黑客"（对 RM 钻漏洞）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q29</b> · RLHF 中 KL 散度约束的作用是什么？β 过大 / 过小各有什么问题？</summary>

**答**：**KL 约束目标**：$r' = R_\phi(x, y) - \beta \cdot D_\text{KL}(\pi_\theta(y|x) \| \pi_\text{ref}(y|x))$

KL 项的作用：防止 actor 策略偏离参考模型（SFT 初始化）太远，避免"奖励黑客"（reward hacking）——模型找到 RM 的漏洞，产生高分但无意义 / 有害输出。

**β 过大（过度约束）**：
- KL 项主导，策略几乎不动，等效于仍是 SFT 模型
- RM 信号对训练影响极小，对齐效果差
- 训练效率低下

**β 过小（约束不足）**：
- 策略过度优化 RM，产生 reward hacking（如输出全是特定格式的高分废话）
- 与 SFT 模型偏离太远，语言质量下降（重复 / 无意义）
- 模型通用能力退化（catastrophic forgetting）

**实务 β 选择**：InstructGPT 原文 β = 0.02-0.04；实务中通常从 0.1 开始，监控 KL divergence 曲线决定调整方向。

**易错**：KL 不是 PPO clip 机制——clip 是限制策略比值 $r_t$，KL 约束是限制策略输出分布距离；两者都是防止策略更新过大，但机制不同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q30</b> · DPO（Direct Preference Optimization）和 RLHF+PPO 的核心区别？DPO 为什么不需要显式 RM？</summary>

**答**：**DPO（Rafailov 2023）**：通过数学推导将 RLHF 的 RL 目标化简为纯监督学习，**绕过了显式 RM 训练和 PPO**。

**推导**：RLHF 最优策略满足 $\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(R/\beta)$；代回 Bradley-Terry 模型后 R 消掉：

$$L_\text{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

$y_w$ 偏好输出，$y_l$ 非偏好输出。RM 被消掉，所有信号来自偏好对 + 参考模型。

**DPO vs PPO**：DPO 简单、无在线采样、稳定；PPO 可在线探索、处理新场景。DPO 用固定偏好数据，无法自适应更新。

**易错**：DPO 仍源于 RL 推导，不是"去掉了 RL"；它是 RLHF 最优解的解析等价，不是启发式简化。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q31</b> · GRPO（Group Relative Policy Optimization）是什么？与 PPO 有何不同？</summary>

**答**：**GRPO（DeepSeekMath / DeepSeek-R1 使用的算法）**：PPO 的变体，**消去了 Critic 网络**，用组内相对奖励估计 advantage。

**核心机制**：同一问题采样 G 个输出，用组内奖励计算 advantage：
$$A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$$
更新 loss（最小化）：$L = -L_\text{CLIP} + \beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})$（KL 直接加入 loss，而非加入奖励）

| | PPO | GRPO |
|---|---|---|
| Critic | 需要（GAE） | 无 |
| Advantage | GAE(TD) | 组内相对奖励 |
| 显存 | 4 模型 | 3 模型 |

**DeepSeek-R1 用 GRPO**：数学题可验证对错，组内对比清晰，无需 Critic；节省约 25% 显存。

**易错**：GRPO 依赖**可验证奖励**；纯 RM 打分场景组内方差大，advantage 估计噪声增加。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×5</span> <b>Q32</b> · VLA 微调中用 PPO 还是 BC？RLHF 和行为克隆的结合方式？</summary>

**答**：**当前主流**：VLA 微调以**行为克隆（BC）/ flow matching 为主**，而非 RLHF+PPO——原因是机器人奖励函数难以自动量化、RM 训练需要大量偏好标注。

**BC 微调路线**：在演示数据上最小化 $-\log\pi_\theta(a|s, l)$（连续用 MSE/flow 损失，离散用 CE）；OpenVLA / Octo / GR00T 均走此路线。

**PPO / RLHF 在 VLA 的尝试**：
- 任务成功率作为奖励（0/1），用 PPO 微调 VLA 策略；
- π0.6 的 RECAP 近似：把演示数据当 offline RL，advantage-conditioned 采样，强化"好动作"权重；
- 问题：① 真机 rollout 代价高；② 连续动作空间 policy 的概率比值计算不稳定；③ 收敛更慢。

**混合路线**（新兴）：SFT（BC 预训练）→ 少量 RLHF 精调（成功 / 失败 episode 作偏好对） → 接近 DPO 形式。

**易错**：VLA 动作预测通常用 flow matching / diffusion 而非 softmax token，PPO 的 clip ratio 需改造为连续策略形式；不要直接套 LLM RLHF 代码到机器人 VLA。

</details>

---

## §7 工程 tricks 与机器人 RL（8 题）

> 面试必考"怎么做"：稀疏奖励、Reward Shaping、Curriculum、仿真训练。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q33</b> · 稀疏奖励（Sparse Reward）有什么危害？有哪些解法？</summary>

**答**：**危害**：agent 几乎无法在合理时间内碰到正奖励，梯度信号几乎为零，策略无法学习（尤其在机械臂 pick-and-place 等任务中）。

**主要解法**：

1. **Reward Shaping**：加中间奖励（如 $-\|p_\text{ee} - p_\text{goal}\|$）；用势能函数形式 $F = \gamma\Phi(s') - \Phi(s)$ 保证最优策略不变。
2. **HER**：goal-conditioned 任务首选（见 Q21）。
3. **Curriculum**：从易到难，避免 agent 从零面对全难版本。
4. **探索增强**：RND（随机网络蒸馏）给新奇状态额外内在奖励。
5. **Demos + RL**：BC 预训练初始化 → RL 精调（AWAC/RLPD）。

**易错**：Reward Shaping 非势能函数会改变最优策略；必须验证诱导最优与原始目标一致。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q34</b> · Curriculum Learning 在机器人 RL 中怎么设计？有哪些维度可以 curriculate？</summary>

**答**：**课程学习（Curriculum Learning）**：将任务难度按序排列，从易到难训练，类似人类学习过程。

**机器人 RL 中可 curriculate 的维度**：

1. **初始状态分布**：从目标附近随机化（easy）→ 逐步扩大到全状态空间（hard）；GoalGAN / ALP-GMM 自动生成适当难度的初始状态。

2. **目标距离**：先训练短程（目标近）→ 扩展到远距离；或 HER + 课程。

3. **物体属性**：形状简单（正方形）→ 复杂（细长棒）；摩擦系数从高（好抓）到低（滑）。

4. **任务子目标分解**：把长任务分解为子目标链（抓起 → 移动 → 放下），分别学再组合。

5. **Domain Randomization 强度**：从低扰动（接近真实）→ 高扰动（鲁棒 sim2real）。

**自动课程（ACL）**：不人工设计难度，用 competence-progress 信号（最近成功率变化量）自动选下一批任务；ALP-GMM 是经典实现。

**易错**：课程不能"难度跳跃"——若简单任务能力尚未泛化就进入难任务，容易 catastrophic forgetting；Critic 从 easy 到 hard 的价值估计分布会剧变，需要特殊处理。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q35</b> · Domain Randomization（DR）是什么？为什么有助于 Sim2Real 迁移？</summary>

**答**：**Domain Randomization（DR）**：在仿真训练中随机化物理参数（摩擦系数、质量、关节阻尼、视觉参数如光照/纹理），迫使策略在宽范围条件下仍有效。

**为什么有助于 Sim2Real**：真实环境的精确参数未知，但只要真实参数**落在仿真随机化范围内**，策略就能泛化。训练出来的策略对参数变化鲁棒，本质上是一种多任务训练。

**两类 DR**：
- **视觉 DR**：光照/颜色/纹理/相机参数 → 让视觉策略不对仿真纹理过拟合
- **物理 DR**：质量/摩擦/弹性/延迟 → 让控制策略对动力学不确定性鲁棒

**OpenAI Dactyl (2019)**：纯仿真训练（+ 大量 DR）直接迁移到真实五指手完成魔方，是 DR 的里程碑。

**易错**：DR 不是"随机化越大越好"——过度随机化使任务太难学（策略在过于宽泛的参数空间无法收敛）；需要配合 adaptive DR 或 curriculum DR（逐步扩大范围）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q36</b> · Reward Shaping 的势能函数条件是什么？为什么违反会导致最优策略偏移？</summary>

**答**：**Ng 等 1999 提出**：shaped reward $r'(s,a,s') = r(s,a,s') + F(s,s')$，若 $F$ 为**势能函数**形式：
$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
则原 MDP 的最优策略 $= $ shaped MDP 的最优策略（策略不变性）。

**为什么违反导致偏移**：若 $F$ 不是势能函数（如加一个任意中间奖励），等效于改变了原始奖励结构，agent 会被引导走向"中间奖励高"的路径，而非原始目标最优路径。

**典型违反例子**：在迷宫中给"靠近墙壁"加 +1（因为人觉得这样探索多），结果 agent 学会在墙壁反复徘徊而不去目标。

**实务建议**：取 $\Phi(s) = -\|s - g\|$（负距离），则 $F = \gamma(-\|s' - g\|) - (-\|s - g\|)$ 满足势能条件，是典型安全 shaping；success bonus（+1 on reach goal）属于原始奖励，不是 shaping，不在此约束范围内；避免用依赖子目标顺序的、多路径不一致的中间奖励。

**易错**：势能函数条件只保证策略不变性，不保证收敛速度 / 样本效率不变；好的 shaping 仍可加速训练，关键是不要偏移最优。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q37</b> · 机器人 RL 选算法的基本原则？PPO vs SAC 怎么选？</summary>

**答**：选算法的核心考量：

| 维度 | PPO（on-policy） | SAC（off-policy） |
|---|---|---|
| Sample efficiency | 低（每 batch 用完丢） | 高（replay buffer 复用） |
| 真机适用性 | 差（需大量 rollout） | 好（少量数据可学） |
| 稳定性 | 高（clip 约束） | 中（过估计可能） |
| 超参敏感度 | 中 | 中-高（α / lr 敏感） |
| 与仿真并行 | 好（多 worker） | 好（异步采样） |
| LLM RLHF 对齐 | 标配 | 少用 |

**选 PPO 的场景**：① 仿真并行训练可生成大量样本；② LLM RLHF（标配）；③ 需要稳定收敛、对超参不敏感时。注：VLA 微调主流仍是 BC/flow matching，PPO 用于机器人仍在探索阶段。

**选 SAC 的场景**：① 真机 sample 极贵（每次 rollout 几十秒）；② 连续控制 + 探索要求高；③ 数据可复用（replay buffer 价值大）。

**混合路线**：仿真用 PPO / TD3 预训练 → 真机用 SAC 精调（少量 online 步骤）。

**易错**：不要在真机 from scratch 用 PPO——样本太贵、无法承受 on-policy 的数据丢弃。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q38</b> · 梯度裁剪（Gradient Clipping）在 RL 里为什么比在 SL 里更重要？</summary>

**答**：**SL 里**：损失函数通常较平滑（CE / MSE），梯度爆炸主要出现在深层 RNN 或早期训练；可以靠良好的 BatchNorm / ResNet 结构缓解。

**RL 里更重要的原因**：

1. **优势函数高方差**：$\hat{A}_t$ 可以是很大的正/负值（尤其 GAE $\lambda=1$ 接近 MC），梯度 $= \nabla \log\pi \cdot \hat{A}$，单个 batch 梯度剧烈波动；

2. **Critic 估计不准早期**：Q / V 网络初期估计误差大，TD 误差大 → 反向传播梯度大；

3. **策略 Clip 不完全保护**：PPO Clip 只限制策略更新方向，不限制梯度 L2 范数，大梯度仍可通过；

4. **非平稳目标**：RL 目标随策略变化，等于一直在非平稳问题上学，梯度范数波动比 SL 大。

**实务**：PPO 代码通常 `max_grad_norm = 0.5`；SAC 通常 `max_grad_norm = 1.0`；不裁剪则训练前 1000 步容易 NaN。

**易错**：梯度裁剪是裁剪 L2 范数（整体缩放），不是截断每个参数梯度（那是 clip by value，效果差得多）。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×4</span> <b>Q39</b> · Intrinsic Motivation（内在动机 / 好奇心驱动）是什么？RND 怎么实现？</summary>

**答**：**内在动机（IM）**：给 agent 额外的内在奖励（intrinsic reward）$r^i$，奖励"新颖"或"信息量大"的状态，解决稀疏/无外部奖励场景下的探索。

$$r_t = r_t^e + \beta \cdot r_t^i, \quad r_t^i = \text{novelty}(s_t)$$

**RND（Burda 2019）**：
- 固定随机目标网络 $f$（冻结）+ 可训练预测网络 $\hat{f}$
- 内在奖励 $r^i = \|f(s) - \hat{f}(s)\|^2$：未见状态误差大 → 高奖励；已访问状态误差小 → 低奖励

**优点**：简单有效，无需建动力学模型；用于 Montezuma's Revenge 达超人类水平。

**易错**：RND 对观测归一化极敏感（状态非平稳时 $r^i$ 会漂移）；$\beta$ 需随训练 annealing，后期逐步减小内在奖励权重。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×4</span> <b>Q40</b> · Hierarchical RL（层级强化学习）的基本思路？在机器人长程任务中的优势？</summary>

**答**：**层级 RL（HRL）**：把复杂长程任务分解为 high-level（HL）策略 + low-level（LL）策略的两层（或多层）结构：
- **HL 策略**：在粗粒度时间步 $(t, t+k, t+2k, \ldots)$ 做决策，输出子目标 $g_t$（如"移动到箱子旁边"）；
- **LL 策略**：在每步执行，接收子目标 $g_t$ 作为条件，输出底层控制命令（关节力矩）。

**代表框架**：HIRO（Data-efficient HRL）——HL 每 c 步发一次子目标，LL 在 c 步内追子目标；子目标用相对状态表示。

**长程任务优势**：
1. 时间尺度分离：HL 不需要管每步细节，学"大方向"；LL 专注精细控制；
2. 缓解信用分配（credit assignment）问题：LL 的任务有明确子目标，短期奖励密集；
3. 复用性：LL"抓取"策略可被多个 HL 任务复用（如"开门后放物体 → 两个任务共享抓取 LL"）。

**易错**：HL 和 LL 的训练需要协调——HL 给不可达子目标会导致 LL 不知所措；子目标空间设计（状态空间 vs 潜在空间）对效果影响大。

</details>

---

## §8 低频备选（3 题）

> 频次 1-2 题，来源有效但低于主表阈值。面试中作为扩展题备用。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N1</b> · 【低频备选】Distributional RL（C51 / QR-DQN）是什么？为什么比期望 Q 更强？</summary>

**答**：**分布式 RL**：不仅学期望 $\mathbb{E}[G_t]$，而是学**回报的完整分布** $Z(s,a)$（随机变量）。

**C51（Bellemare 2017）**：把回报分布离散为 51 个 atom，学 $P(Z(s,a) = z_i)$；Bellman 更新也在分布空间做（投影 Bellman）。

**QR-DQN（Dabney 2018）**：用分位数回归估计回报分布的分位点 $\tau_1, \ldots, \tau_N$。

**为什么更强**：① 分布提供更丰富的学习信号（不仅最优期望，还有方差/风险）；② 在优化 Critic 时更稳定（减少过估计）；③ 在 Atari 多游戏上 C51 显著超越 DQN。

**易错**：分布式 RL 的"分布"是**回报的分布**（随机性来自环境），不是策略的分布（策略熵）；两者完全不同概念。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2</span> <b>N2</b> · 【低频备选】RLPD（Reinforcement Learning with Prior Data）和纯 SAC 有何不同？</summary>

**答**：**RLPD（Ball 2023）**：解决"如何在 online RL 中高效利用 offline 先验数据"——把离线数据塞进 replay buffer 一起训练，但做了关键改进：

**与纯 SAC + offline data 的区别**：
1. **Symmetric sampling**：每个 mini-batch 50% 来自 offline 数据、50% 来自 online 数据，保证离线先验持续发挥作用；
2. **Critic LayerNorm + ensemble**：LayerNorm 防止 Q 值过度外推到 OOD 区域；ensemble 提供多头保守估计；
3. **高 UTD（Update-to-Data=20）**：每采 1 步环境数据做 20 次梯度更新；配合 LayerNorm 才不发散。

Actor 仍用标准 SAC 风格最大化 Q，不是 in-sample max（IQL 才是）。

**结论**：RLPD 用极少 online 交互即超越 BC，是 offline-to-online 的简洁强基线。

**易错**：裸 SAC 提高 UTD 会发散；RLPD 的 LayerNorm + ensemble 是高 UTD 稳定的关键，缺一不可。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2</span> <b>N3</b> · 【低频备选】Multi-step PPO / APPO（Asynchronous PPO）在大规模分布式训练中怎么用？</summary>

**答**：**APPO（Sample Factory, PureJaxRL 等）**：将 PPO 的数据收集与策略更新解耦，多个 actor 进程异步采集数据，learner 进程持续更新策略。

**关键设计**：
- Actor 进程运行旧版本策略收集数据（略微 off-policy）；
- Learner 不等所有 actor 完成，收到足够 batch 就更新；
- 用 IS ratio 或 V-trace（IMPALA）纠正 off-policy 误差。

**Scale 优势**：可以在数千 CPU 核 / 数十 GPU 上线性扩展吞吐；具身 sim 训练（IsaacGym 4096 环境并行）配合 APPO 达到极高样本吞吐量。

**与同步 PPO 区别**：同步等所有 worker 完成后再更新（木桶效应）；APPO 不等待慢 worker，吞吐量提升显著但引入略多 off-policy bias。

**易错**：APPO 的 off-policy 程度比 SAC 小得多（旧策略只滞后 1-2 个 batch），通常不需要完整 IS 纠正；V-trace 做的是截断 IS 校正，不是完整重加权。

</details>

---

## §H 手撕代码（10 题）

> RL 核心公式手撕段——与上文"概念+答案"题区分开。本节每题给"考察点 / 实现 / 易错"——附 ≤30 行 Python 实现。

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>✍ H01</b> · 手撕 Q-learning 更新（TD(0) 表格版）</summary>

**考察点**：`Q[s,a] += α·(r + γ·max_a' Q[s',a'] - Q[s,a])`；off-policy 因为 target 用 max 而非实际所采动作。

**实现**：

```python
import numpy as np

def q_learning_step(Q, s, a, r, s2, done, alpha=0.1, gamma=0.99):
    # 终止时 V(s_T)=0：用 (1-done) 截断，避免跨 episode bootstrap
    td_target = r + gamma * Q[s2].max() * (1.0 - done)
    Q[s, a] += alpha * (td_target - Q[s, a])
    return Q

def eps_greedy(Q, s, eps, n_actions, rng):
    # 探索-利用平衡；ε 通常从 1.0 退到 0.05
    if rng.random() < eps:
        return rng.integers(n_actions)
    return int(Q[s].argmax())  # argmax 平 tie 时取索引最小
```

**易错**：α 太大不收敛、太小爬太慢；终止还加 `γ·max Q(s')` 导致跨 episode bootstrap；连续动作不能 max → 换 DDPG/SAC。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>✍ H02</b> · 手撕 DQN（含 target net + replay buffer）</summary>

**考察点**：replay buffer 打破时序相关性；target net 稳定 bootstrap；target 上必须 `detach`。

**实现**：

```python
import random
from collections import deque
import torch
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, cap): self.buf = deque(maxlen=cap)
    def push(self, *t): self.buf.append(t)
    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        # 一次性堆叠为 batch tensor
        return [torch.as_tensor(x).float() if i != 1 else torch.as_tensor(x).long()
                for i, x in enumerate([s, a, r, s2, d])]

def dqn_loss(q_net, q_target_net, batch, gamma=0.99):
    s, a, r, s2, d = batch
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():  # target net 不回传梯度
        target = r + gamma * q_target_net(s2).max(1).values * (1.0 - d)
    return F.smooth_l1_loss(q_sa, target)  # Huber 比 MSE 更稳
```

**易错**：target 未 detach 导致梯度回灌；done=True 还加 `γV(s')`；replay 容量太小样本相关性高。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×14</span> <b>✍ H03</b> · 手撕 PPO clipped objective（含 value loss + entropy）</summary>

**考察点**：`min(r·A, clip(r,1-ε,1+ε)·A)`；为何用 min（取悲观下界）；多 epoch 更新 ratio 偏离。

**实现**：

```python
import torch
import torch.nn.functional as F

def ppo_loss(logp_new, logp_old, adv, value, ret, entropy,
             clip=0.2, c1=0.5, c2=0.01):
    # logp_old 来自采样时旧策略，必须已 detach；adv 已做 GAE+归一化
    ratio = torch.exp(logp_new - logp_old)
    # advantage 标准化降方差；unbiased=False 防 batch=1 时 NaN
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    # min 取悲观下界：A>0 时上裁、A<0 时下裁
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(value, ret)  # 或 clipped value loss
    # entropy 鼓励探索，整体 loss 取负 entropy
    return policy_loss + c1 * value_loss - c2 * entropy.mean()
```

**易错**：`logp_old` 没 detach → 重复梯度；entropy 符号错（loss 里应 -c2·H）；多 epoch 不监控 KL，ratio 漂太远。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>✍ H04</b> · 手撕 GAE 优势估计</summary>

**考察点**：λ 在 0/1 之间权衡 bias/variance；必须从后向前递归；`done` 处截断 bootstrap。

**实现**：

```python
import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    # rewards/values/dones: [T]；last_value: V(s_T)（bootstrap 用）
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):  # 必须反向
        # done 处截断：next_value 与 next_gae 都被 mask 掉
        next_v = last_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    # V_target 用旧 critic 的 values + adv，避免 critic 自指漂移
    returns = adv + values
    return adv, returns
```

**易错**：正向遍历；`done` 处没 mask；`V_target` 用当前 critic 输出而非 `V_old` 导致训练抖。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>✍ H05</b> · 手撕 DDPG / TD3 critic update</summary>

**考察点**：DDPG = 连续动作 Q-learning；TD3 三大改进——双 Q、actor 延迟、target policy smoothing。

**实现**：

```python
import torch
import torch.nn.functional as F

def td3_critic_loss(q1, q2, q1_tgt, q2_tgt, mu_tgt, batch,
                    gamma=0.99, sigma=0.2, c=0.5):
    s, a, r, s2, d = batch
    with torch.no_grad():  # target 必须 detach
        # target policy smoothing：给 target 动作加 clipped noise
        noise = (torch.randn_like(a) * sigma).clamp(-c, c)
        a2 = (mu_tgt(s2) + noise).clamp(-1.0, 1.0)
        # 双 Q 取 min 抑制高估
        y = r + gamma * torch.min(q1_tgt(s2, a2), q2_tgt(s2, a2)) * (1.0 - d)
    return F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)

def td3_actor_loss(q1, mu, s):
    # actor 只用 Q1 上升（用 min 反而抑制学习）；每 2 个 critic step 调一次
    return -q1(s, mu(s)).mean()
```

**易错**：target 未 detach；actor 用 `min(Q1,Q2)` 抑制学习；探索 noise 与 target smoothing noise 混淆（两件事）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>✍ H06</b> · 手撕 SAC（reparameterize + 可学温度 α）</summary>

**考察点**：max-entropy RL 目标；reparam 让梯度穿过采样；tanh 雅可比修正保证 `log π` 归一化。

**实现**：

```python
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def sample_action(mu, log_std):
    # reparameterize：u = μ + σ·ε，ε 与策略参数独立
    std = log_std.exp()
    normal = Normal(mu, std)
    u = normal.rsample()  # rsample 走 reparam path，sample 会断梯度
    a = torch.tanh(u)
    # tanh 雅可比修正：log π(a) = log p(u) - Σ log(1 - tanh(u)²)
    log_p = normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
    return a, log_p.sum(-1, keepdim=True)

def sac_critic_target(r, s2, mu_t, logstd_t, q1_t, q2_t, log_alpha,
                      gamma=0.99, done=0.0):
    with torch.no_grad():
        a2, logp2 = sample_action(mu_t(s2), logstd_t(s2))
        # 双 Q min 抑制高估；减 α·log π 体现 max-entropy 目标
        q_next = torch.min(q1_t(s2, a2), q2_t(s2, a2)) - log_alpha.exp() * logp2
        return r + gamma * q_next * (1.0 - done)

def alpha_loss(log_alpha, log_pi, target_ent):
    # target_ent 一般 -|A|；log π 须 detach
    return -(log_alpha * (log_pi.detach() + target_ent)).mean()
```

**易错**：用 `sample()` 而非 `rsample()` 断梯度；忘 tanh 雅可比 → π 不归一；`α` 不学卡死；`target_ent` 设错。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>✍ H07</b> · 手撕 DPO loss</summary>

**考察点**：从 RLHF 推导；只需 ref + policy 两个模型（无 reward model）；β 控制 KL 强度。

**实现**：

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy, ref, batch, beta=0.1):
    # batch: prompt + chosen/rejected response token ids + mask（标记 response 段）
    # 只在 response token 上累加 log_prob，prompt 段必须屏蔽
    def logp_seq(model, ids, resp_mask):
        logits = model(ids).logits[:, :-1]  # 预测下一 token
        tgt = ids[:, 1:]
        m = resp_mask[:, 1:].float()  # 与 tgt 对齐
        # gather token-level log_prob，再按 response mask 求和
        lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return (lp * m).sum(-1)

    lp_w_pi = logp_seq(policy, batch['ids_w'], batch['mask_w'])
    lp_l_pi = logp_seq(policy, batch['ids_l'], batch['mask_l'])
    with torch.no_grad():  # ref 必须 freeze
        lp_w_ref = logp_seq(ref, batch['ids_w'], batch['mask_w'])
        lp_l_ref = logp_seq(ref, batch['ids_l'], batch['mask_l'])
    # 隐式 reward 差：β·[(logπ-logπ_ref)_w - (logπ-logπ_ref)_l]
    logits = beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))
    return -F.logsigmoid(logits).mean()
```

**易错**：ref 未 freeze 协同漂移；β 跟温度混淆；prompt token 的 log_prob 未屏蔽；padding token 也累加。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×5</span> <b>✍ H08</b> · 手撕 GRPO loss（group relative policy opt）</summary>

**考察点**：DeepSeek-R1 用的 RL；group baseline 替代 critic（省一份 value 模型显存）；在线 RL（vs DPO 离线）。

**实现**：

```python
import torch

def grpo_advantage(rewards):
    # rewards: [B, G]，每 prompt 采 G 个 response
    # baseline=组内均值；除组内 std 做 z-score；unbiased=False 防 G=1 NaN
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False) + 1e-8
    return (rewards - mean) / std

def grpo_loss(logp_new, logp_old, logp_ref, rewards, resp_mask,
              clip=0.2, kl_coef=0.04):
    # logp_*: [B, G, T]；resp_mask: [B, G, T] 标记 response token（屏蔽 prompt/padding）
    adv = grpo_advantage(rewards).unsqueeze(-1)            # [B, G, 1]
    ratio = torch.exp(logp_new - logp_old.detach())
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    pg = -torch.min(unclipped, clipped)                    # token-level
    # k3 KL 估计（DeepSeekMath 用法）：log(p/q)+(q/p)-1，对 ref 做正则
    log_r = logp_ref.detach() - logp_new
    kl = log_r.exp() - log_r - 1
    loss = (pg + kl_coef * kl) * resp_mask                 # 只在 response 段计 loss
    return loss.sum() / resp_mask.sum().clamp(min=1.0)     # 按有效 token 数归一
```

**易错**：把 GRPO 当 DPO（GRPO 在线 RL）；忘 response mask 把 prompt/padding 也算进 loss；漏 KL/ref 正则；`std` 用 `unbiased=True` 在 G=1 时 NaN。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>✍ H09</b> · 手撕 importance sampling 比率</summary>

**考察点**：`E_p[f]=E_q[f·p/q]`；off-policy 必须用；多步累乘 IS 方差指数爆炸 → PPO clip / V-trace 截断。

**实现**：

```python
import torch

def is_ratio(logp_new, logp_old):
    # 用 log_prob 之差再 exp，避免概率小数下溢
    # logp_old 必须来自采样时旧策略（detach），不可用当前网络重算
    return torch.exp(logp_new - logp_old.detach())

def vtrace_truncated(logp_new, logp_old, rho_bar=1.0, c_bar=1.0):
    # V-trace 双截断：ρ 控 target、c 控 trace 传播
    log_r = logp_new - logp_old.detach()
    rho = torch.exp(log_r).clamp(max=rho_bar)
    c = torch.exp(log_r).clamp(max=c_bar)
    return rho, c
```

**易错**：用 `probs` 比例而非 `log_prob` 之差 → 数值不稳；忘 detach old；多步累乘不截断方差爆炸。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>✍ H10</b> · 手撕 Categorical 采样 + log_prob</summary>

**考察点**：多项式采样；为何 `logits` 比 `probs` 数值稳（避免 `log(0)`）；与 `multinomial` 区别。

**实现**：

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def categorical_sample(logits):
    # logits: [B, n_actions]；推荐用 logits 入参，内部做 log_softmax
    dist = Categorical(logits=logits)
    a = dist.sample()                # [B]
    log_prob = dist.log_prob(a)      # [B]，等价于下方 gather 公式
    entropy = dist.entropy()         # [B]，policy 正则常用
    return a, log_prob, entropy

def manual_log_prob(logits, a):
    # 等价于 Categorical.log_prob，但展示底层
    # F.log_softmax 比 log(softmax) 数值稳（用 LogSumExp）
    return F.log_softmax(logits, -1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
```

**易错**：用 `probs` 输入但概率为 0 时 `log(0)`；`torch.multinomial` 不直接返回 log_prob；gather 维度未 unsqueeze。

</details>
