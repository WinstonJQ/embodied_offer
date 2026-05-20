## §0 TL;DR Cheat Sheet

> 💡 **一句话定位** — 本册讲 *现代主力 RL 算法*：你跑 MuJoCo / IsaacGym / 机器人 / RLHF 时实际用的那些。

**先记住这张演化图**：

```
1992  REINFORCE      ── 纯 MC return，方差大，没有 critic
        │ + value baseline
        ▼
2016  A2C / A3C      ── advantage = G_t - V(s)，方差大降
        │ + 步长控制（KL trust region）
        ▼
2015  TRPO           ── 硬约束 KL ≤ δ，单调改进保证，但 CG 太重
        │ + clipped surrogate（去掉硬约束，保留思想）
        ▼
2017  PPO            ── min(r·A, clip(r,1±ε)·A)，SGD 友好，RLHF 默认
```

**6 个关键人物**（每个解决一个特定问题）：

1. **TRPO**（2015）：用 KL 约束控制步长，解 *vanilla PG 一步崩* 问题
2. **PPO**（2017）：用 clip + min 代替硬 KL，工程上压倒 TRPO；**RLHF 的引擎**
3. **GAE**（2016）：给 advantage 估计加一个 $\lambda$ 旋钮，调 bias-variance
4. **DDPG**（2016）：把 Q-learning 推到连续动作（用 deterministic policy）
5. **TD3**（2018）：修 DDPG 的 3 个坑——**Twin** Q + **Delayed** actor + **Smoothed** target
6. **SAC**（2018）：最大熵框架 + reparameterization + 自动温度 $\alpha$，机器人 RL 默认

> ✅ **快速记忆口诀** — 每个算法都在做"trust region + 降方差"两件事

- TRPO：硬 KL 约束，自然梯度
- PPO：软 clip surrogate，SGD
- GAE：长视野方差用 $\lambda$ 压
- DDPG：deterministic PG + replay
- TD3：twin Q 修 max bias
- SAC：entropy bonus + reparameterization

> 📌 **本册不讲什么** — Q-learning / DQN / Bellman / on-off-policy / IS 已在 [V1 rl_foundations](rl_foundations_tutorial.html) 详细讲过；RLHF / DPO / GRPO / Offline RL 在 V3 post_training_rl 单独讲。

## §1 策略梯度家族关系图

### 1.1 一个直觉故事：为什么 PG 家族要演化这么多代

想象你在训练一个 4 关节机械臂抓杯子。你用最朴素的 REINFORCE：

```
for episode in range(N):
    rollout 一整局 → 拿到 reward G
    g = ∇log π(轨迹) · G     ← 高方差！G 跨整局聚合所有噪声
    θ ← θ + lr · g
```

你会遇到三个连环问题：

1. **方差大** → 训练慢、不稳定 → 第一代修法：**加 critic** $V_\phi(s)$ 做 baseline → 这就是 **A2C / A3C**
2. **步长难定** → 一次更新 $\theta$ 走小步在参数空间看似安全，但 $\pi_\theta$ 的输出分布可能突变（"60% 向右伸手" 一步跳到 "99.99% 向上举手"），下一批数据全是垃圾，再也爬不出来。→ 第二代修法：**用 KL 距离约束策略改变量** → 这就是 **TRPO**
3. **TRPO 工程上太重**（自然梯度 + conjugate gradient + 线搜索） → 第三代修法：**用一阶 clip 替代二阶约束** → 这就是 **PPO**

每一代都是"上一代留下的坑"驱动的。下面这张表把每代各自解决了什么、引入了什么列清楚：

| 年 | 算法 | 解决了什么 | 引入了什么 | 留下了什么坑 |
|---|---|---|---|---|
| 1992 | REINFORCE | 给出 PG 的 MC 估计器 | 无偏估计 | 方差爆炸 |
| 2016 | A2C / A3C | 加 baseline 降方差 | actor-critic 范式 | 步长仍然不安全 |
| 2015 | TRPO | KL 约束控制步长 | 单调改进保证 + 自然梯度 | CG / 线搜索工程难 |
| 2017 | PPO | clip surrogate 替代约束 | SGD 友好 + RLHF 友好 | 监督理论保证（仅经验有效）|
| 2016 | GAE | 给 advantage 加 $\lambda$ 旋钮 | bias-variance 可调 | — |

> 💡 **看演化图最重要的事** — 每代不是 "新算法替代旧算法"，而是 **A2C 给 PPO 用、GAE 也给 PPO 用、TRPO 的思想被 PPO 借走**。它们是 *组件*，PPO 是组装。

### 1.2 一张图：PPO 是怎么"组装"出来的

```
PPO = REINFORCE 的 PG 基础                  (∇log π · A)
    + A2C 的 advantage （A = Q - V）        ← critic 头
    + GAE 估 A_t（λ 旋钮）                  ← bias-variance
    + TRPO 的 trust region 思想             ← KL 约束（被 clip 替代）
    + 一打工程 trick（adv-norm / value-clip / KL early-stop 等）
```

记住这一点：**面试官问 "PPO 是什么" 时，别只答 clip 公式——答这张图。**

## §2 为什么需要 trust region

### 2.1 vanilla PG 的 *步长悖论*

PG 的更新公式 $\theta \leftarrow \theta + \eta \nabla J$ 看起来很普通。问题在于：**$\eta$ 该选多少？**

- $\eta$ 太小：训练慢得能耗光预算
- $\eta$ 太大：策略 *分布* 突变 → 下一批数据全在 OOD 区 → critic 信号崩坏 → 爬不出来

**关键洞察**：$\theta$ 在 *参数空间* 走一小步，$\pi_\theta$ 在 *策略空间*（分布空间）可能走巨大一步。两个空间的距离不成比例。

> 💡 **直觉类比** — 在地图上往北走 1 米，可能是平地（无害），也可能是悬崖（直接掉下去）。$\eta$ 是步长米数；KL 是真正的"地形距离"。

### 2.2 自然的距离尺度：KL 散度

衡量"两个策略有多不同"，最自然的尺度是 KL 散度：

$$D_{\text{KL}}\bigl(\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\Vert\, \pi_\theta(\cdot \mid s)\bigr) = \sum_a \pi_{\theta_{\text{old}}}(a \mid s) \log \frac{\pi_{\theta_{\text{old}}}(a \mid s)}{\pi_\theta(a \mid s)}$$

- KL = 0 当且仅当两个分布完全相同
- KL 大 ⇒ 新策略 *分布层面* 跟旧策略差很远

**控制 KL = 控制策略真实变化量** —— 这是 trust region 的灵魂。

### 2.3 Kakade-Langford 单调改进界（CPI bound）

理论支撑：Kakade & Langford 2002 *Conservative Policy Iteration* 证明了

$$\boxed{\;J(\pi_{\text{new}}) \;\ge\; L_{\pi_{\text{old}}}(\pi_{\text{new}}) \;-\; C \cdot \max_s D_{\text{KL}}\!\bigl(\pi_{\text{old}}(\cdot \mid s) \,\Vert\, \pi_{\text{new}}(\cdot \mid s)\bigr)\;}$$

其中：

- $L_{\pi_{\text{old}}}(\pi_{\text{new}}) = J(\pi_{\text{old}}) + \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}},\, a \sim \pi_{\text{new}}}\![A^{\pi_{\text{old}}}(s, a)]$ 是 **surrogate**（用 *旧* 状态分布 + *新* 策略选动作）
- $C = \dfrac{4 \gamma \varepsilon}{(1 - \gamma)^2}$，$\varepsilon = \max_{s, a} \lvert A^{\pi_{\text{old}}}(s, a) \rvert$（TRPO 2015 Thm 1 推出：先用 $D_{\text{TV}}^{\max}$ 得 $C_{\text{TV}} = 4\gamma\varepsilon/(1-\gamma)^2 \cdot \alpha^2$，再用 $D_{\text{TV}}^2 \le D_{\text{KL}}$ 把 TV 换成 KL，常数保持 $4$）

**这个界告诉我们**：只要每步保证 KL 不太大，**每次更新都不会让性能变差**——单调改进。

> ⚠️ **理论 vs 实践** — CPI 界给出的 $C$ 在实际任务里巨大（$\sim 10^4$），用它直接做 step size 调度会非常保守。TRPO / PPO 都是 *启发式简化*——把 max KL 换成 mean KL，把硬界换成软约束。

### 2.4 接下来的两条路

```
                Kakade-Langford CPI 界
                       │
              ┌────────┴─────────┐
              ▼                  ▼
       硬 KL 约束           软 KL 惩罚 / clip
       max L s.t. KL ≤ δ    max [L - β·KL] 或 clip(r)
              │                  │
              ▼                  ▼
            TRPO              PPO
```

## §3 TRPO（高层概念，不深挖数学）

> 📌 **论文** — Schulman, Levine, Moritz, Jordan, Abbeel, *"Trust Region Policy Optimization"*, ICML **2015**, arXiv:1502.05477。

### 3.1 优化问题

$$\max_\theta\; \underbrace{\mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} A^{\pi_{\theta_{\text{old}}}}(s, a)\right]}_{\text{surrogate } L(\theta)}\quad \text{s.t.}\quad \underbrace{\bar D_{\text{KL}}(\pi_{\theta_{\text{old}}} \Vert \pi_\theta)}_{\text{average KL}} \le \delta$$

典型 $\delta = 0.01$。

### 3.2 实操三步

理论很优雅，工程上需要三步特技：

1. **线性化目标 + 二次近似 KL** → 得到自然梯度更新 $\Delta\theta = \alpha\, F^{-1} g$，$F$ 是 Fisher 信息矩阵
2. **Conjugate Gradient (CG)** 算 $F^{-1} g$ —— 不显式建 $F$，用 $\sim 10$ 次 Hessian-vector product 解出来（避免 $\lvert\theta\rvert \times \lvert\theta\rvert$ 矩阵）
3. **Line search** 沿 $\Delta\theta$ 回溯，直到 surrogate 实际改进 **且** $\bar D_{\text{KL}} \le \delta$ —— 这才真正强制 trust region

> 💡 **为什么 TRPO 现在没人直接用** — CG + Hessian-vector + line search 在多 worker 并行 / 共享 value head / shared param 场景下bookkeeping 复杂；PPO 用普通 SGD 拿到同等效果，没人想多打这么多代码。

### 3.3 TRPO 留下的精神遗产

虽然 TRPO 工程上被 PPO 取代，**它的思想活在每个现代 PG 算法里**：

- "用 IS 重写 surrogate" → PPO 的 $r_t(\theta) \hat A_t$ 就是 TRPO 的 surrogate
- "KL 约束控制步长" → PPO 的 clip 是 KL 约束的近似（PPO-Clip）或软化（PPO-KLPen）
- "Pessimistic lower bound" → PPO clip 的 min 就是 pessimistic 选择
- "Trust region" 这词本身成了 deep RL 通用术语

## §4 PPO-Clip 推导与几何

> 📌 **论文** — Schulman, Wolski, Dhariwal, Radford, Klimov, *"Proximal Policy Optimization Algorithms"*, arXiv:1707.06347, **2017-07**。

### 4.1 故事先行：PPO 怎么"绕过"CG + line search

PPO 的核心想法很 *偷懒*：

> "既然 KL 约束在工程上麻烦，那我别真的算 KL——我换一种 *形式* 让大步子自动失去梯度信号不就行了？"

这就是 clip 的初衷。

### 4.2 单步 IS 比

定义当前策略对老策略的 *概率比*：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

两个直觉：

- $r_t = 1$ ⇔ 当前策略和老策略选这个 $(s, a)$ 的概率一样
- $r_t = 2$ ⇔ 当前策略概率是老策略的 2 倍（"更倾向了"）
- $r_t = 0.5$ ⇔ 当前策略概率是老策略的一半（"更避免了"）

> 💡 **重要细节** — 反向传播时 *分母 detach*，只把 $\pi_\theta$（分子）当作 $\theta$ 的函数。否则梯度乱套。

### 4.3 CPI surrogate（不加约束的"前身"）

直接最大化下式：

$$L^{\text{CPI}}(\theta) = \hat{\mathbb{E}}_t\!\bigl[r_t(\theta) \hat A_t\bigr]$$

**问题**：若某个 transition 有大 $\hat A_t$ + $r_t$ 飞到 100，这一项主导整个 batch，策略一步飞出 trust region —— 跟 vanilla PG 一样的步长悖论。

### 4.4 Clipped surrogate（PPO 的招）

$$\boxed{\;L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\Bigl[\min\bigl(r_t(\theta) \hat A_t,\;\operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat A_t\bigr)\Bigr]\;}$$

默认 $\varepsilon = 0.2$（MuJoCo），RLHF 用 0.1-0.3。

### 4.5 几何：4 个象限决定 clip 是否生效

把 $(\hat A_t,\, r_t)$ 分 4 象限，看 clip 何时 *实际* 起作用：

| $\hat A_t$ 符号 | $r_t$ 位置 | $L^{\text{CLIP}}_t$ 取值 | 梯度是否流动 |
|---|---|---|---|
| **$\hat A_t > 0$** | $r_t < 1 - \varepsilon$（动作好但策略反而降低概率了）| $r_t \hat A_t$（unclipped, 小）| **流动**——纠错 |
| **$\hat A_t > 0$** | $r_t \in [1-\varepsilon, 1+\varepsilon]$ | $r_t \hat A_t$ | 流动 |
| **$\hat A_t > 0$** | $r_t > 1 + \varepsilon$（已经把概率提得太高）| $(1+\varepsilon) \hat A_t$（clipped）| **被截**——不再奖励 |
| **$\hat A_t < 0$** | $r_t < 1 - \varepsilon$（已经把概率降得太低）| $(1-\varepsilon) \hat A_t$（clipped, 较小负）| **被截**——不再惩罚 |
| **$\hat A_t < 0$** | $r_t \in [1-\varepsilon, 1+\varepsilon]$ | $r_t \hat A_t$ | 流动 |
| **$\hat A_t < 0$** | $r_t > 1 + \varepsilon$（动作差但策略反而提升概率了）| $r_t \hat A_t$（unclipped, 较大负）| **流动**——纠错 |

```
              r_t = 1 ─────► (1+ε)
                    ε     ε
   A>0：  ←─ 流（纠错）─ 流 ─►│ 截 │
                              ─┼─
   A<0：     │ 截 │── 流 ──► 流（纠错）
                    ε     ε
              (1-ε)◄─── 1
```

**直觉总结**：

- 当 *策略已经按 advantage 方向走过头*（同号且 $\lvert r_t - 1 \rvert > \varepsilon$）→ **clip 起作用，梯度归零** → "够了别再走了"
- 当 *策略走错了方向*（异号）→ **clip 不起作用，梯度仍流** → "改回来"

### 4.6 为什么是 `min` 而不是直接 `clip`

这是 PPO 设计的关键巧思。考虑 "纠错象限"：$\hat A_t > 0$（动作好）但 $r_t < 1-\varepsilon$（策略却把概率降了）。

- 如果只 `clip(r_t, 1-ε, 1+ε)`：得到 $(1-\varepsilon) \hat A_t$，**比真实的 $r_t \hat A_t$ 大**（因为 $r_t < 1-\varepsilon$）——这会让 loss 谎称"我们已经赚了"，**梯度归零**，没法纠错
- `min(r_t \hat A_t, (1-\varepsilon) \hat A_t)` = $r_t \hat A_t$ → 选 *较小* 的（"更悲观"的）→ **梯度仍指向把 $r_t$ 拉回去**

**一句话**：`min` 让 PPO 在"已经犯错"的方向永远诚实，**只悲观估改进，不乐观估损失**。

> 💡 **类比** — PPO clip 像批改作业的严格老师：你已经把答案改对了 +20% 还想多改？老师不给加分（clip 截）；你把答案改错了？老师必须扣分让你改回来（min 选悲观）。

### 4.7 PPO-Clip vs PPO-KLPen（论文原文有两版）

论文原文还给了 KL-penalty 版本：

$$L^{\text{KLPEN}}(\theta) = \hat{\mathbb{E}}_t\!\bigl[r_t(\theta) \hat A_t - \beta \cdot D_{\text{KL}}(\pi_{\theta_{\text{old}}} \Vert \pi_\theta)\bigr]$$

$\beta$ 在线自适应：KL 超目标 1.5 倍则 $\beta \times 2$，低于 0.67 倍则 $\beta /= 2$。

**结论**（原论文 Table 3）：在 7 个 MuJoCo 任务里 clip 版赢 6 个；现代 PPO 一律用 clip。

## §5 GAE 广义优势估计

> 📌 **论文** — Schulman, Moritz, Levine, Jordan, Abbeel, *"High-Dimensional Continuous Control Using Generalized Advantage Estimation"*, ICLR **2016**, arXiv:1506.02438。

### 5.1 起点：TD residual 与 n-step advantage

V1 我们见过 TD residual：

$$\delta_t^V = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

如果 $V = V^\pi$，则 $\mathbb{E}[\delta_t^V] = A^\pi(s_t, a_t)$——**1 步 advantage 估计**，无偏但只看一步信号（高 bias from V 误差）。

把"只看一步"延展到 $n$ 步：

$$\hat A_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l\, r_{t+l+1} + \gamma^n V(s_{t+n}) - V(s_t)$$

- $n = 1$：1 步 TD（低方差、高偏）
- $n \to \infty$：MC return（无偏、高方差）

### 5.2 关键引理：n-step advantage = TD residual 的部分和

$$\hat A_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l\, \delta_{t+l}^V$$

**证明**（telescoping）：展开右边

$$\sum_{l=0}^{n-1} \gamma^l\bigl[r_{t+l+1} + \gamma V(s_{t+l+1}) - V(s_{t+l})\bigr]$$

reward 项是 $\sum_{l=0}^{n-1} \gamma^l r_{t+l+1}$（直接）；V 项形成 telescoping：

$$\gamma V(s_{t+1}) - V(s_t) + \gamma^2 V(s_{t+2}) - \gamma V(s_{t+1}) + \dots = \gamma^n V(s_{t+n}) - V(s_t)$$

合起来正是 $\hat A_t^{(n)}$。$\blacksquare$

### 5.3 GAE 定义

把不同 $n$ 的估计器按指数权重平均：

$$\hat A_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1}\, \hat A_t^{(n)}$$

代入 §5.2 的引理 + 交换求和顺序（中间用几何级数 $(1-\lambda)\sum_{n=l+1}^\infty \lambda^{n-1} = \lambda^l$）：

$$\boxed{\;\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l\, \delta_{t+l}^V\;}$$

**这就是工程实现里直接用的形式。**

### 5.4 递归形式（代码里这么写）

从展开式直接得到反向递归：

$$\hat A_t^{\text{GAE}} = \delta_t^V + \gamma \lambda\, \hat A_{t+1}^{\text{GAE}}$$

边界：$\hat A_T^{\text{GAE}} = 0$（terminal）。**所有 PPO 实现都是反向扫一遍 rollout 算 GAE**。

价值目标顺手得到：

$$\hat V_t^{\text{targ}} = \hat A_t^{\text{GAE}} + V(s_t)$$

### 5.5 两个极端

- **$\lambda = 0$**：$\hat A_t = \delta_t^V$，**纯 1 步 TD**（低方差、高 bias）
- **$\lambda = 1$**：$\hat A_t = \sum_l \gamma^l \delta_{t+l}^V$；telescoping 后等于 $\sum_l \gamma^l r_{t+l+1} - V(s_t)$，**MC return 减 baseline**（高方差、给定准 V 时无偏）

### 5.6 $\gamma$ vs $\lambda$ —— *别搞混*

| 符号 | 角色 | 几何意义 | 由谁定 |
|---|---|---|---|
| $\gamma$ | 任务折扣 | 决定有效视野 $1/(1-\gamma)$；定义"我们的目标是什么" | 问题（reward 设计）|
| $\lambda$ | 估计器旋钮 | 决定多大程度信 $V$ 的 bootstrap；只影响"我们怎么估计 advantage" | 算法（估计器质量）|

虽然 $\gamma\lambda$ 总是一起出现，但**是两个独立的旋钮**——可以 $\gamma = 0.99,\, \lambda = 1$（任务长视野 + 纯 MC 估计）。

**实践默认**：$\gamma = 0.99,\, \lambda = 0.95$；超长视野任务用 $\lambda = 0.97$。

## §6 PPO 工程实操（37 details 精华）

> 📌 **论文** — Engstrom et al, *"Implementation Matters in Deep Policy Gradients"*, ICLR **2020**, arXiv:2005.12729；Huang et al, *"The 37 Implementation Details of PPO"*, ICLR **2022 Blog Track**。

**Engstrom 2020 的核心发现**：PPO 经验上压倒 TRPO 的 **大部分增益来自这些工程 trick**，不是 clip objective 本身。这就是为什么"读 PPO 论文 + 自己实现"和 "PPO baseline 调好"差很多。

| Trick | 细节 | 作用 |
|---|---|---|
| **Advantage normalization** | 每 minibatch 内：$\hat A \leftarrow (\hat A - \mu) / (\sigma + 10^{-8})$ | 让 loss scale 稳定，跨任务可比 |
| **Value function clipping** | $L^V_{\text{CLIP}} = \max((V - V_{\text{targ}})^2,\, (\operatorname{clip}(V, V_{\text{old}} \pm \varepsilon) - V_{\text{targ}})^2)$ | 对 V 头也用 pessimistic clip |
| **Global gradient clip** | $\Vert g \Vert_2 \le 0.5$ | 防爆炸 |
| **Reward scaling** | 除以 *running std of discounted return*（**不中心化**） | 中心化会移动最优点 |
| **Orthogonal init** | hidden $\sqrt{2}$，policy logits 头 $0.01$，value 头 $1.0$ | 初始策略近均匀 |
| **Linear LR decay** | $3 \times 10^{-4} \to 0$ 线性衰减 | 收敛阶段更稳 |
| **Adam $\varepsilon$** | $10^{-5}$（**不是** PyTorch 默认 $10^{-8}$）| 数值稳定 |
| **KL early stop** | KL > 1.5 × target 立刻退出当前 epoch | 防止过度优化 |
| **Minibatch × epoch** | 典型 $4$ epoch × $4$ minibatch（Atari）；$10 \times 32$（连续）| sample reuse |
| **GAE 一次算** | 用 *冻结* $V_{\text{old}}$ 算一次，不要每 epoch 重算 | 算一次足够 |

> ⚠️ **面试常考** — 问"为什么你跑 PPO 跑不出论文效果"时答：检查这 10 条。9 成是 advantage normalization 或 reward scaling 没做对。

## §7 PPO in RLHF：为什么还要 KL to reference model

> 📌 **canonical 论文** — Stiennon et al *"Learning to summarize from human feedback"*, NeurIPS **2020**, arXiv:2009.01325；Ouyang et al *"InstructGPT"*, NeurIPS **2022**, arXiv:2203.02155。

### 7.1 RLHF setup 一句话回顾

```
1. SFT             ── 监督微调，得到 π_ref（"reference model"）
2. RM 训练         ── 拿对比偏好数据训 reward model R_φ
3. PPO 微调        ── 用 R_φ 作 reward 信号，PPO 更新 π_θ；π_θ 初始化 = π_ref
```

注意：**π_ref 在整个 PPO 阶段冻结不动**。

### 7.2 关键公式：per-token reward

$$\boxed{\;\tilde r_t = R_\phi(s_t, a_t) - \beta\, \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}\;}$$

第二项就是 **per-token KL penalty** $\beta \cdot \text{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}})$ 的样本估计。

很多实现里 $R_\phi$ 只在最后一个 token 给（"序列级 reward"），KL 项每个 token 都给。

### 7.3 这个 KL 到底在做什么？三件事

**(1) 防 reward hacking**——这是最重要的

$R_\phi$ 是 *学出来* 的，只在训练分布上可信。没有 KL 锚的话，PPO 会让 $\pi_\theta$ 钻 $R_\phi$ 的漏洞：

- 产出 gibberish（无意义重复 token）但 $R_\phi$ 误报高分
- 总用同样的"安全套话"
- 利用 RM 的 length bias / format bias

KL penalty 让 OOD 行为 *成本变高*，把 $\pi_\theta$ 拉回 $\pi_\text{ref}$ 见过的语言分布。

**(2) 保住自然语言流利度**

$\pi_\text{ref}$（SFT 模型）说人话。KL 锚保证 RL 不会让模型话术变得 *机械* / *不自然*。

**(3) 双重 trust region**

PPO 自带的 clip 控制 $\pi_\theta$ vs $\pi_{\theta_{\text{old}}}$ 的偏移（**单步** trust region）；KL to $\pi_\text{ref}$ 控制 $\pi_\theta$ vs $\pi_\text{ref}$ 的偏移（**全局** trust region）。两个一起防止 *跨多轮缓慢漂移到 reward hacking 区*。

### 7.4 这个 KL **不是** importance sampling

面试官最爱挖的坑。**记住**：

- $\pi_\text{ref}$ 是 **冻结** 的——不是 behavior policy
- rollout 由 *当前* $\pi_\theta$ 采（外层 on-policy）
- PPO 内层 SGD 多 epoch 复用 batch，那是 *$\pi_{\theta_{\text{old}}}$ 对 $\pi_\theta$* 的 IS——和 $\pi_\text{ref}$ 无关

$\pi_\text{ref}$ 只是 **regularizer** 的锚。

> 💡 **类比** — PPO 的 clip = "今天不准走太远"；KL to π_ref = "无论今天走哪里，都别离家太远"。两条绳，方向不同。

### 7.5 KL 在 loss 里的位置

KL 项是 *fold 进 per-step reward $\tilde r_t$* 的：

```
1. rollout：用 π_θ 采轨迹 → 拿 R_φ + KL → 算 \tilde r_t
2. GAE：用 \tilde r_t 算 advantage \hat A_t（KL 自然进了 advantage）
3. value loss：V_ψ 学预测 sum γ^l \tilde r_{t+l}（KL 进了 V 目标）
4. policy loss：标准 PPO clip，不变
```

所以 **value loss 形式不变**，只是 target 里夹了 KL。这一点不少同学搞错。

### 7.6 $\beta$ 怎么选 / 怎么调

两种范式：

- **固定 $\beta$**：常用 $0.01$ ～ $0.2$；简单但需要 tuning
- **自适应 $\beta$**（PPO 原论文 §4 + InstructGPT）：mean KL > 1.5 × target 则 $\beta \times 2$；< 0.67 × target 则 $\beta /= 2$。target 典型 $\text{KL}_\text{targ} \approx 6$

### 7.7 现代替代品（V3 详讲）

- **DPO**（Rafailov+ 2023）：把 KL-regularized RL 重新参数化成 *监督* 学习，去掉显式 RM + RL loop
- **GRPO**（DeepSeek 2024）：group-normalized advantage，省掉 value head；推理 RL 强
- **RLVR**（DeepSeek R1）：reward 用 *程序判分*（数学题判答案、代码跑 test）替代 RM；省掉 reward hacking 根源

**但截至 2026**：production-grade alignment pipeline 主流仍是 **PPO-style RLHF**。

## §8 DDPG（一段带过的前身）

> 📌 **论文** — Lillicrap et al, *"Continuous control with deep reinforcement learning"*, ICLR **2016**, arXiv:1509.02971；前置的 **DPG 定理** = Silver et al ICML **2014**。

DDPG 把 Q-learning 推到连续动作。一句话核心：**actor 输出一个动作（不是分布）$a = \mu_\phi(s)$，loss = $-Q_\theta(s, \mu_\phi(s))$**，靠链式法则把梯度沿 $\frac{\partial Q}{\partial a} \cdot \frac{\partial \mu}{\partial \phi}$ 反传到 actor。

工程结构：

- Actor + Critic 都带 target 网，Polyak $\tau = 0.001$
- 经验回放（DQN style）
- 探索靠 OU 噪声（原版）或 Gaussian 噪声（现代）加在 actor 输出上

**两个臭名昭著的坑**（TD3 修的就是它们）：

1. **Q 过估** —— $Q'(s', \mu'(s'))$ 中 actor 隐式做 $\max$，任何正向 critic 误差被 actor 放大
2. **brittle** —— actor 一小动，下一批数据的 action 分布就变，critic 跟不上

> 💡 **DDPG 一句话总结** — 第一次成功把 deep RL 推到连续动作；但 max bias + brittleness 让它后来基本被 TD3 + SAC 取代。

## §9 TD3：修 DDPG 的 3 个坑

> 📌 **论文** — Fujimoto, van Hoof, Meger, *"Addressing Function Approximation Error in Actor-Critic Methods"*, ICML **2018**, arXiv:1802.09477。

**助记口诀**：**Twin / Delayed / Smoothed**

### 9.1 Fix 1: Twin Q（Clipped Double Q-learning）

训练 *两套* critic $Q_{\theta_1}, Q_{\theta_2}$，target 取 min：

$$y = r + \gamma\, \min_{i = 1, 2} Q_{\theta_i'}\!\bigl(s',\, \mu_{\phi'}(s') + \tilde\varepsilon\bigr)$$

**为什么 min 而不是 mean / max**：

- $\max$：放大正向噪声 → 加剧过估
- $\text{mean}$：减少一点方差，但还是无偏估正噪声
- $\min$：**有意识地引入负向偏差** —— 抵消 max bias

> 💡 **直觉** — DDPG 的过估是 actor 在"挖 Q 噪声的高峰"；用 min Q 把高峰削平，actor 没山头可挖。

### 9.2 Fix 2: Delayed actor update

每 $d$ 个 critic 更新做 *1 次* actor 更新（默认 $d = 2$）+ Polyak target 同步。

**直觉**：critic 还没稳定（Q 估计准之前）就让 actor 跟着动，actor 在跟噪声跳舞。delay 让 critic 先收一会儿，actor 再调整。

### 9.3 Fix 3: Target policy smoothing

往 target action 加 *clipped Gaussian noise*：

$$\tilde\varepsilon = \operatorname{clip}(\mathcal{N}(0, \sigma),\, -c,\, +c),\quad a_{\text{targ}} = \operatorname{clip}\bigl(\mu_{\phi'}(s') + \tilde\varepsilon,\; a_{\min},\; a_{\max}\bigr)$$

注意 **外层 clip**：smoothed target action 还要 clip 回有效动作范围 $[a_{\min}, a_{\max}]$，否则 $Q'(s', a_{\text{targ}})$ 可能查询训练分布外的动作。

**为什么**：deterministic actor 偶尔会找到 $Q$ 函数的 *尖峰*（窄的高值区域），实际部署稍微偏一点 Q 就掉。加 noise 强迫 critic 学 *平滑* 的 Q——actor 找到的高值区也得对 noise 鲁棒。

### 9.4 关键超参（默认）

| 超参 | 值 |
|---|---|
| Polyak $\tau$ | 0.005 |
| 噪声 $\sigma$ | 0.2 |
| 截断 $c$ | 0.5 |
| 延迟 $d$ | 2 |
| Replay 容量 | $10^6$ |
| Batch | 100-256 |

> ⚠️ **常见 bug** — 忘了 *target net Polyak 同步也走 delayed schedule*（每 $d$ 步 1 次），结果 target 跟普通 critic 同步快，全套 fix 失效。

## §10 SAC：最大熵 RL + reparameterization

> 📌 **论文 v1** — Haarnoja, Zhou, Abbeel, Levine, *"Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"*, ICML **2018**, arXiv:1801.01290。
>
> 📌 **论文 v2（auto-α）** — Haarnoja et al, *"Soft Actor-Critic Algorithms and Applications"*, arXiv:1812.05905, **2018-12**。

### 10.1 一句话核心：最大熵框架

把传统 RL 的目标加一个 *熵奖励*：

$$\boxed{\;J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\bigl[r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))\bigr]\;}$$

其中熵 $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$，温度 $\alpha > 0$ 控制 *探索 / 利用* 权衡。

**直觉**：在所有能达到 return $R^*$ 的策略中，选 *最随机* 的那个。

> 💡 **为什么这有用** — 三个理由。

1. **鼓励探索** —— 不会过早 collapse 到 deterministic
2. **对多模态 reward 自然 robust** —— 保留多个高 reward 模式而不是只挑一个
3. **理论扎实** —— 数学上等价于 *energy-based policy*

### 10.2 Soft Bellman 方程

加了熵，Bellman 也得跟着变 *软*：

$$Q^\pi(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s'}[V^\pi(s')]$$
$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\!\bigl[Q^\pi(s, a) - \alpha \log \pi(a \mid s)\bigr]$$

**与标准 Bellman 区别**：$V$ 多减了 $\alpha \log \pi$ 项——这就是 "soft"。

合并成一个递归：

$$Q^\pi(s, a) = r + \gamma\, \mathbb{E}_{s', a' \sim \pi}\!\bigl[Q^\pi(s', a') - \alpha \log \pi(a' \mid s')\bigr]$$

### 10.3 三个网络

```
              State s
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
    Q_θ1(s,a) Q_θ2(s,a) π_φ(a|s)
       │        │           │
    twin       twin     stochastic
    critics    critics   actor (Gaussian)
       │        │
       └───min──┘  ← TD3 trick 借过来
```

- **Twin critics** $Q_{\theta_1}, Q_{\theta_2}$ + target $\bar\theta_1, \bar\theta_2$
- **Stochastic actor** $\pi_\phi$：高斯分布，输出 $(\mu_\phi(s),\, \log \sigma_\phi(s))$；最后过 tanh 把动作压到 $[-1, 1]$

### 10.4 Critic loss（直接 TD）

$$y = r + \gamma\bigl(\min_{j = 1, 2} Q_{\bar\theta_j}(s', a') - \alpha \log \pi_\phi(a' \mid s')\bigr),\quad a' \sim \pi_\phi(\cdot \mid s')$$
$$L_Q(\theta_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\bigl[(Q_{\theta_i}(s, a) - y)^2\bigr]$$

注意 target 里 *只用 target critic*，actor 用的是 *当前* $\pi_\phi$。

### 10.5 Actor loss：reparameterization 入场

直接形式（用 log-derivative trick）：

$$J_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\phi}\!\bigl[\alpha \log \pi_\phi(a \mid s) - \min_j Q_{\theta_j}(s, a)\bigr]$$

问题：$a \sim \pi_\phi$ 是对 $\phi$ 参数化的采样，标准 log-derivative 梯度

$$\nabla_\phi J = \mathbb{E}[\nabla_\phi \log \pi_\phi(a \mid s) \cdot (\alpha \log \pi_\phi - Q)]$$

方差和 $\lvert Q \rvert^2$ 成正比，**特别大**。

**SAC 的解**：用 **reparameterization trick**

$$a = f_\phi(\xi; s) = \tanh\bigl(\mu_\phi(s) + \sigma_\phi(s) \odot \xi\bigr),\quad \xi \sim \mathcal{N}(0, I)$$

**关键**：把 *对参数化分布的采样* 重写成 *对固定分布 $\mathcal{N}(0, I)$ 的采样 + 确定性变换 $f_\phi$*。期望从对 $a$ 改成对 $\xi$，$\phi$ 在 $f_\phi$ 里通过链式法则反传：

$$\nabla_\phi J = \mathbb{E}_{s,\, \xi \sim \mathcal{N}}\bigl[\nabla_\phi\bigl(\alpha \log \pi_\phi(f_\phi(\xi; s) \mid s) - \min_j Q_{\theta_j}(s, f_\phi(\xi; s))\bigr)\bigr]$$

方差变为 $Q$ 的 *曲率*——远小于 $\lvert Q \rvert^2$。

> 💡 **VAE 用过同样的 trick** — 如果你做过 VAE，应该觉得眼熟：VAE 的 latent $z = \mu + \sigma \xi$ 是 *同款* reparameterization。SAC 等于把 VAE 的方差降低魔法搬到了 RL actor 上。

### 10.6 tanh 压缩的 Jacobian 修正（必须！）

$a = \tanh(u)$，$u = \mu + \sigma \xi$。Change of variables：

$$\log \pi_\phi(a \mid s) = \log \mathcal{N}(u \mid \mu_\phi(s),\, \sigma_\phi(s)^2) - \sum_{i = 1}^{\lvert\mathcal{A}\rvert} \log(1 - \tanh^2(u_i))$$

第二项是 **tanh 的 Jacobian determinant**。

**数值稳定写法**（$\lvert u \rvert$ 大时 $\tanh^2(u) \to 1$ 会 log(0)）：

$$\log(1 - \tanh^2(u)) = 2\bigl[\log 2 - u - \operatorname{softplus}(-2u)\bigr]$$

> ⚠️ **SAC 头号 bug** — 忘了减 Jacobian 项。错误密度 → 错误熵估计 → α 朝错误方向自动调 → silent over-exploration。出 bug 不报错，性能默默下降。

### 10.7 自动温度 α（v2 版本）

v1 SAC 把 $\alpha$ 当超参（很麻烦——不同任务最优 $\alpha$ 差大）。v2 把它写成 *约束优化* 自动调：

$$\max_\pi \mathbb{E}\bigl[\sum_t r_t\bigr]\quad \text{s.t.}\quad \mathbb{E}[\mathcal{H}(\pi(\cdot \mid s))] \ge \bar{\mathcal{H}}$$

$\bar{\mathcal{H}}$ 是 *目标熵*，典型取 $\bar{\mathcal{H}} = -\lvert\mathcal{A}\rvert$（动作维度的负值，约定俗成）。

**对偶 Lagrangian**（primal max over $\pi$，dual **min** over $\alpha \ge 0$）：

$$\mathcal{L}(\pi, \alpha) = \mathbb{E}\bigl[\sum_t r_t + \alpha\, \mathcal{H}(\pi_t)\bigr] - \alpha\, \bar{\mathcal{H}}$$

对 $\alpha$ 做 **对偶下降**（gradient descent on $\mathcal{L}$ w.r.t. dual var $\alpha$）：

$$\nabla_\alpha \mathcal{L} = \mathbb{E}[\mathcal{H}(\pi)] - \bar{\mathcal{H}} = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)] - \bar{\mathcal{H}}$$

α 更新 $\alpha \leftarrow \alpha - \eta\, \nabla_\alpha \mathcal{L}$。

**动力学**（注意符号！）：

- 当前熵 $\mathbb{E}[\mathcal{H}] < \bar{\mathcal{H}}$（策略太 deterministic，约束违反）→ $\nabla_\alpha \mathcal{L} < 0$ → 对偶下降让 $\alpha$ **变大** → 更鼓励探索
- 当前熵 $\mathbb{E}[\mathcal{H}] > \bar{\mathcal{H}}$（策略太随机，约束有 slack）→ $\nabla_\alpha \mathcal{L} > 0$ → 对偶下降让 $\alpha$ **变小** → 减少熵奖励

**工程**：实现里直接 minimize $J(\alpha) = \mathbb{E}[-\alpha \log \pi - \alpha \bar{\mathcal{H}}]$（SAC v2 Eq 18），对 $\rho = \log \alpha$ 用 Adam（保 $\alpha > 0$）。

### 10.8 SAC 完整训练循环（伪代码）

```
init: π_φ, Q_θ1, Q_θ2, target Q_bar1, Q_bar2, log α, replay D

for step in range(N):
    a = π_φ(s).sample()  # reparameterized + tanh
    s' = env.step(a)
    D.add(s, a, r, s')
    
    sample batch from D
    
    # critic update
    a' = π_φ(s').sample(); log π' = π_φ.log_prob(a')
    y = r + γ (min Q_bar(s', a') - α log π')
    L_Q1 = MSE(Q_θ1(s,a), y); L_Q2 = MSE(Q_θ2(s,a), y)
    
    # actor update (reparameterization)
    ξ ~ N(0, I); a~ = tanh(μ_φ(s) + σ_φ(s) ⊙ ξ); log π~ = π_φ.log_prob(a~)
    L_π = α log π~ - min Q(s, a~)
    
    # α update
    L_α = -log α · (log π~ + H_bar).detach()
    
    backward all; soft update Q_bar ← τ Q + (1-τ) Q_bar
```

## §11 SAC vs PPO 选型

### 11.1 决策表

| 维度 | PPO | SAC |
|---|---|---|
| 策略类型 | 高斯 / 离散 | 高斯压缩到 $[-1,1]$ |
| on / off-policy | 近似 on（单 batch 多 epoch）| off（replay）|
| 样本效率 | 低（百万步起）| **高**（比 PPO 好 10-100×）|
| Wall-clock 效率 | **高**（embarrassingly parallel）| 中等 |
| 稳定性 | **高** | 中等（α 调好后高）|
| 连续动作 | ✓ | ✓（原生设计）|
| 离散动作 | ✓ 原生 | Christodoulou 2019 有版本，少见 |
| RLHF | **主流** | 几乎没用 |
| 机器人 RL | 模拟 + 数据丰富时用 | **真机 / 昂贵 sim 主流** |
| Compute 配置 | 多并行 env | 单 env + replay |

### 11.2 决策树

```
要做的事是？
├─ LLM RLHF / 对齐
│   └─ PPO（clip 版本）—— 几乎没有替代
├─ Atari / MuJoCo / IsaacGym 等廉价并行 sim
│   └─ PPO（多并行 env wall-clock 占优）
├─ 真机机器人 / 昂贵 sim / 数据少
│   └─ SAC（off-policy 样本效率压倒 PPO）
├─ 探索极难（稀疏 reward）
│   ├─ SAC（entropy bonus 帮探索）
│   └─ + curiosity bonus / RND
└─ 离散 + 大动作空间（如棋类）
    └─ PPO 或 MuZero
```

### 11.3 共同失败模式

- **PPO**：reward scale + advantage normalization 没调好；KL early stop 阈值错；GAE λ 选错
- **SAC**：tanh Jacobian 忘加；target entropy $\bar{\mathcal{H}}$ 选错；early-stage α 飘
- **两者**：reward 太稀疏，得加 reward shaping / curriculum / RND

## §12 25 道高频面试题

### L1 — 必会（10 题）

<details>
<summary><strong>1. 写 PPO-Clip 的目标函数。</strong></summary>

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\bigl[\min(r_t(\theta) \hat A_t,\, \operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat A_t)\bigr]$$

其中 $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$，默认 $\varepsilon = 0.2$。
</details>

<details>
<summary><strong>2. 写 TD residual 和 GAE 公式。</strong></summary>

TD residual：$\delta_t^V = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$

GAE：$\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$，递归形式 $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$。
</details>

<details>
<summary><strong>3. 列 TD3 的 3 个 fix。</strong></summary>

**Twin / Delayed / Smoothed**：

1. **Twin Q**（clipped double Q）：两套 critic，target 取 $\min$
2. **Delayed actor update**：actor + target Polyak 每 $d = 2$ 步做 1 次
3. **Target policy smoothing**：target action 加 clipped Gaussian noise
</details>

<details>
<summary><strong>4. SAC 的 entropy 项有什么用？</strong></summary>

鼓励探索（不 collapse 到 deterministic），保留 *多模态* 高 reward 分布。形式：目标改为 $J(\pi) = \mathbb{E}[\sum_t (r + \alpha \mathcal{H}(\pi))]$。
</details>

<details>
<summary><strong>5. RLHF-PPO 的 per-token reward 公式。</strong></summary>

$$\tilde r_t = R_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}$$

第二项是 KL penalty（不是 importance sampling！）。
</details>

<details>
<summary><strong>6. 为什么 PG 要用 value baseline？</strong></summary>

降梯度方差（不影响均值——baseline 不依赖 $a$）。证明：$\mathbb{E}_a[\nabla \log \pi \cdot b(s)] = b(s) \nabla \sum_a \pi = 0$。
</details>

<details>
<summary><strong>7. 写 TRPO 的约束。</strong></summary>

$$\max_\theta\; \mathbb{E}\!\left[\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} A\right]\quad \text{s.t.}\quad \bar D_{\text{KL}}(\pi_{\theta_{\text{old}}} \Vert \pi_\theta) \le \delta$$

典型 $\delta = 0.01$。
</details>

<details>
<summary><strong>8. DDPG vs TD3 一句话。</strong></summary>

TD3 = DDPG + Twin Q（修 max bias）+ Delayed actor（防 critic 不稳）+ Target policy smoothing（防尖峰 Q）。
</details>

<details>
<summary><strong>9. PPO 默认超参速查。</strong></summary>

$\varepsilon = 0.2$，$\gamma = 0.99$，$\lambda = 0.95$，lr $= 3 \times 10^{-4}$（Adam，eps = $10^{-5}$），$K = 4 \sim 10$ epochs × $4 \sim 32$ minibatches，value-loss coef 0.5，entropy coef 0.01。
</details>

<details>
<summary><strong>10. advantage 是什么？</strong></summary>

$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ —— 动作 $a$ 相对该策略 *平均* 表现的差额。正 advantage = "比平均好"，负 = "比平均差"。
</details>

### L2 — 进阶（10 题）

<details>
<summary><strong>11. PPO 为什么是 min(...) 而不是 直接 clip？</strong></summary>

考虑"纠错象限"：$\hat A > 0$ 但 $r_t < 1 - \varepsilon$（动作好但策略却把概率降了，需要纠回去）。

- 单纯 `clip`：$(1-\varepsilon) \hat A_t$，**比真实 $r_t \hat A_t$ 大**，让 loss 误报"已经赚了" → 梯度归零 → 无法纠错
- `min`：选较小的 $r_t \hat A_t$（pessimistic）→ **梯度仍指向把 $r_t$ 拉回去**

一句话：`min` 让 PPO 在"已经犯错"方向永远诚实——**只悲观估改进，不乐观估损失**。
</details>

<details>
<summary><strong>12. GAE 的 $\lambda$ 为什么和 $\gamma$ 是 *独立* 的旋钮？</strong></summary>

- $\gamma$ 定义 *任务目标* —— 多远的 reward 还算数（有效视野 $1/(1-\gamma)$）
- $\lambda$ 定义 *估计器质量* —— 多大程度信 $V$ 的 bootstrap

两者只在 GAE 公式里以 $\gamma\lambda$ 配对出现，是 *巧合*。可以 $\gamma = 0.99,\, \lambda = 1$（长视野 + 纯 MC 估计）。
</details>

<details>
<summary><strong>13. baseline 为什么不引偏？</strong></summary>

$\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a \mid s) \cdot b(s)] = b(s) \sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = b(s) \nabla \sum_a \pi(a \mid s) = b(s) \nabla 1 = 0$。

**关键条件**：$b$ 只依赖 $s$，不依赖 $a$。
</details>

<details>
<summary><strong>14. 真机机器人为什么选 SAC 不选 PPO？</strong></summary>

3 个原因：

1. **样本效率**：真机一小时只能采几千 transitions，PPO 单 batch 用完就扔；SAC 用 replay 反复学
2. **探索**：entropy bonus 帮助 sparse reward 任务
3. **不能并行**：你不可能同时跑 64 个真实机械臂；PPO 的并行 rollout 优势失效
</details>

<details>
<summary><strong>15. TD3 的 min Q 为什么能修过估？</strong></summary>

DDPG 的过估根源：actor 在 max 噪声 Q。Jensen 不等式 $\mathbb{E}[\max_a Q(s,a)] \ge \max_a \mathbb{E}[Q(s,a)]$ → bias 正向。

Twin Q + min 把 bias 翻成 *负向*：对任意两个 iid 噪声估计器 $\hat Q_1, \hat Q_2$ of $Q^*$，$\mathbb{E}[\min(\hat Q_1, \hat Q_2)] \le Q^*$。actor 没有正向 noise 可挖。
</details>

<details>
<summary><strong>16. RLHF 里的 KL to π_ref 跟 PPO 的 clip 各管什么？</strong></summary>

两个 trust region 方向不同：

- **PPO clip**：控制 *单步* $\pi_\theta$ vs $\pi_{\theta_{\text{old}}}$ 的偏移（防一步崩）
- **KL to π_ref**：控制 *全局* $\pi_\theta$ vs $\pi_\text{ref}$（冻结 SFT 模型）的偏移（防多轮缓慢漂移到 reward hacking 区）

形象：clip = "今天不走太远"，KL to ref = "无论走哪天都别离家太远"。
</details>

<details>
<summary><strong>17. SAC actor 为什么用 reparameterization？</strong></summary>

直接 log-derivative gradient 方差 $\propto \lvert Q \rvert^2$（很大）。

Reparameterize $a = \tanh(\mu + \sigma \xi)$，$\xi \sim \mathcal{N}(0, I)$：期望对 $\xi$ 而非 $a$，梯度沿 $f_\phi$ 链式反传，方差变为 $Q$ 的 *曲率*——远小于 $\lvert Q \rvert^2$。**VAE 同款 trick**。
</details>

<details>
<summary><strong>18. PPO 为什么不需要 value 网的 target net？</strong></summary>

PPO 是 *近似 on-policy*——每轮重采新 rollout，KL 受 clip 控制，value drift 自然有界。

DDPG / TD3 / SAC 是 off-policy 复用 replay 数据，bootstrap target 用 $\theta$ 直接算会发散（"追自己尾巴"），所以需要 *慢更新* 的 target net。
</details>

<details>
<summary><strong>19. PPO 的 value function clipping 是什么？为什么要？</strong></summary>

$$L^V_{\text{CLIP}} = \max\bigl((V_\theta - V_{\text{targ}})^2,\, (\operatorname{clip}(V_\theta, V_{\text{old}} \pm \varepsilon) - V_{\text{targ}})^2\bigr)$$

对 value 头也做 pessimistic clip。**作用**：防止 value 一步更新跨太远；和 policy clip 思想一致。

37 details 论文里有，没这条不能复现原 PPO 效果。
</details>

<details>
<summary><strong>20. TD3 的 delayed actor update 直觉？</strong></summary>

critic 还没稳定（$Q$ 估计准之前）就让 actor 跟着动 → actor 在追噪声跳舞。

Delay 让 critic 先收一会儿（$d = 2$ 步），actor 再走 1 步——避免 actor 把 *瞬时 critic 误差* 当 ground truth。
</details>

### L3 — 顶级 lab（5 题）

<details>
<summary><strong>21. 证：$L^{\text{CLIP}} \le L^{\text{CPI}}$ 逐点成立。</strong></summary>

对每个样本：$\min(a, b) \le a$ 显然，所以 $\min(r_t \hat A_t,\, \operatorname{clip}(r_t) \hat A_t) \le r_t \hat A_t$。

取期望保持不等号：$L^{\text{CLIP}}(\theta) \le L^{\text{CPI}}(\theta)$。

**等号条件**：$r_t \in [1-\varepsilon, 1+\varepsilon]$（clip 不生效），或者 clip 后的值反而 *大于* unclipped（"纠错象限"，min 选 unclipped）。

**意义**：$L^{\text{CLIP}}$ 是 $L^{\text{CPI}}$ 的 **pessimistic lower bound** —— 最大化它隐式 maximize 一个保守界，效果类似 trust region。
</details>

<details>
<summary><strong>22. 从对偶问题推 SAC auto-α 更新公式。</strong></summary>

约束问题：$\max_\pi \mathbb{E}[\sum r]$ s.t. $\mathbb{E}[\mathcal{H}(\pi)] \ge \bar{\mathcal{H}}$。

Lagrangian：

$$\mathcal{L}(\pi, \alpha) = \mathbb{E}\bigl[\sum r_t + \alpha \mathcal{H}(\pi_t)\bigr] - \alpha \bar{\mathcal{H}}$$

KKT stationarity in $\pi$（固定 $\alpha$）恢复 soft Bellman backup（§10.2）。

Primal max over $\pi$，dual **min** over $\alpha \ge 0$ → **对偶下降**：

$$\nabla_\alpha \mathcal{L} = \mathbb{E}[\mathcal{H}(\pi)] - \bar{\mathcal{H}} = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)] - \bar{\mathcal{H}}$$

α 更新 $\alpha \leftarrow \alpha - \eta \nabla_\alpha \mathcal{L}$。

**动力学**：
- 当前熵 < 目标 → $\nabla_\alpha \mathcal{L} < 0$ → 对偶下降使 $\alpha \uparrow$ → 更鼓励探索
- 当前熵 > 目标 → $\nabla_\alpha \mathcal{L} > 0$ → 对偶下降使 $\alpha \downarrow$ → 减少熵奖励

工程实现：minimize $J(\alpha) = \mathbb{E}[-\alpha \log \pi - \alpha \bar{\mathcal{H}}]$（SAC v2 Eq 18），参数化 $\rho = \log \alpha$ 保 $\alpha > 0$，Adam 独立 lr。
</details>

<details>
<summary><strong>23. 为什么 TD3 的 twin Q 能修过估，DDPG 不能？</strong></summary>

DDPG: 单 critic，target $y = r + \gamma Q'(s', \mu'(s'))$。actor 在 $Q'$ 上做"伪 max"（找 $\mu'$ 输出对应的 $Q'$ 最高的状态）。Jensen + noise：

$$\mathbb{E}[\max_a Q'(s, a)] \ge \max_a \mathbb{E}[Q'(s, a)]$$

→ bias 正向，每次 bootstrap 累积。

TD3: 两个 *iid* 噪声估计器 $\hat Q_1, \hat Q_2$ of true $Q^*$。取 $\min$：对任意 iid $X, Y$ unbiased of $Q^*$，$\mathbb{E}[\min(X, Y)] \le \mathbb{E}[X] = Q^*$ → **bias 翻负向**。actor 没有正向噪声可挖。

> ⚠️ 但请注意：min Q 仍然是 *biased* 估计器（只是 bias 方向变了），不是 *unbiased*。**针对 max bias 本身** 的进一步修法：clipped double Q（TD3 自带）、ensemble + uncertainty penalty（REDQ）、conservative objectives（CQL）、calibration。**Distributional RL（C51 / QR-DQN）正交方向**——它建模回报分布而不是消 max bias，与上述方法 *叠加* 而非 *替代*。
</details>

<details>
<summary><strong>24. 证 GAE 的两个极限。</strong></summary>

GAE：$\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$。

**$\lambda = 0$**：$\hat A_t = \delta_t^V = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$。**1 步 TD residual**。

**$\lambda = 1$**：$\hat A_t = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V$。展开：

$$= \sum_l \gamma^l (r_{t+l+1} + \gamma V(s_{t+l+1}) - V(s_{t+l}))$$

$V$ 项 telescoping（参考 §5.2 证明）→ 只剩 $-V(s_t)$（terminal 项 $\gamma^\infty V \to 0$）：

$$\hat A_t = \sum_l \gamma^l r_{t+l+1} - V(s_t) = G_t - V(s_t)$$

→ **MC return 减 baseline**。给定准 $V$ 时无偏。
</details>

<details>
<summary><strong>25. PPO-Clip / PPO-KLPen / TRPO，哪个有单调改进保证？</strong></summary>

只有 **TRPO** 严格携带——通过 Kakade-Langford CPI 界 + 硬 KL 约束 + 线搜索 *实际强制*。

- **PPO-KLPen**：把硬约束换软惩罚 $-\beta \cdot \text{KL}$，但 $\beta$ 不保证满足 CPI 界要求。理论上是 *启发式*。
- **PPO-Clip**：不直接约束 mean KL，只约束 *样本级* IS ratio。理论上离 CPI 更远。

**实践中**：三个都稳定有效；理论保证 → 经验有效性的链条在 deep RL 里普遍很弱。**TRPO 仅有的理论优势在工程麻烦面前不值得**——这就是 PPO 取胜的根本原因。
</details>

## §A 附录：from-scratch 代码

### A.1 PPO on Pendulum (~150 行)

```python
import gymnasium as gym, torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))   # state-independent
        self.v = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.shared(x)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        v = self.v(h).squeeze(-1)
        return Normal(mu, std), v

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_low, act_high = env.action_space.low[0], env.action_space.high[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

ac = ActorCritic(obs_dim, act_dim).to(device)
opt = torch.optim.Adam(ac.parameters(), lr=3e-4, eps=1e-5)

# hparams
gamma, lam = 0.99, 0.95
clip_eps, vf_coef, ent_coef = 0.2, 0.5, 0.0
T_rollout, n_epochs, mb_size = 2048, 10, 64
target_kl = 0.02

def compute_gae(rewards, values, dones, last_v):
    """ Backward sweep: A_t = δ_t + γλ A_{t+1} """
    adv = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        next_v = last_v if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    return adv, adv + values

def rollout():
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    state, _ = env.reset()
    for _ in range(T_rollout):
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, v = ac(x)
            a = dist.sample()                            # unbounded Normal sample
            logp = dist.log_prob(a).sum()                # log-prob of the unclipped a
        # NOTE: simplified action bounding — store the unclipped a (matches logp),
        # but send a clamped copy to env. For action range > 1 use tanh-squashed
        # Gaussian + Jacobian (see §A.2 SAC), or scale via action_high.
        a_env = a.clamp(act_low, act_high).cpu().numpy()
        nxt, r, terminated, truncated, _ = env.step(a_env)
        obs_buf.append(state); act_buf.append(a.cpu().numpy())
        logp_buf.append(logp.item()); rew_buf.append(r); val_buf.append(v.item())
        # Gymnasium: only `terminated` zeros bootstrap; for `truncated` we should
        # ideally bootstrap from V(nxt) before reset. Simplified educational code
        # below treats both as terminal for GAE — for production, see CleanRL's
        # `ppo_continuous_action.py` which handles truncation cleanly.
        done_buf.append(float(terminated or truncated))
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = nxt
    with torch.no_grad():
        _, last_v = ac(torch.as_tensor(state, dtype=torch.float32, device=device))
    adv, ret = compute_gae(np.array(rew_buf), np.array(val_buf),
                            np.array(done_buf), last_v.item())
    return (np.array(obs_buf), np.array(act_buf), np.array(logp_buf),
            adv, ret, np.array(val_buf))

for iteration in range(200):
    obs, acts, old_logp, adv, ret, old_v = rollout()
    # advantage normalization (37 details!)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    acts_t = torch.as_tensor(acts, dtype=torch.float32, device=device)
    old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
    old_v_t = torch.as_tensor(old_v, dtype=torch.float32, device=device)
    
    early_stop = False
    for epoch in range(n_epochs):
        idx = np.random.permutation(T_rollout)
        for start in range(0, T_rollout, mb_size):
            mb = idx[start:start + mb_size]
            dist, v = ac(obs_t[mb])
            logp = dist.log_prob(acts_t[mb]).sum(-1)
            ratio = (logp - old_logp_t[mb]).exp()
            
            # PPO clip
            surr1 = ratio * adv_t[mb]
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[mb]
            pg_loss = -torch.min(surr1, surr2).mean()
            
            # value clip (37 details!)
            v_clipped = old_v_t[mb] + (v - old_v_t[mb]).clamp(-clip_eps, clip_eps)
            v_loss = torch.max((v - ret_t[mb])**2, (v_clipped - ret_t[mb])**2).mean()
            
            ent = dist.entropy().sum(-1).mean()
            loss = pg_loss + vf_coef * v_loss - ent_coef * ent
            
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)  # grad clip
            opt.step()
        
        # KL early stop (37 details!)
        with torch.no_grad():
            dist, _ = ac(obs_t)
            kl = (old_logp_t - dist.log_prob(acts_t).sum(-1)).mean().item()
        if kl > 1.5 * target_kl:
            early_stop = True
            break
    
    if iteration % 5 == 0:
        ret_mean = ret.mean()
        print(f"iter {iteration:3d}  return {ret_mean:7.1f}  kl {kl:.4f}  stopped@{epoch}")
```

**预期**：~150 iterations 把 Pendulum return 拉到 $-200$ 以上（满分约 $-150$）。

### A.2 SAC on Pendulum (~150 行)

```python
import gymnasium as gym, torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np, collections, random

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
    def sample(self, s):
        h = self.net(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        normal = Normal(mu, std)
        u = normal.rsample()                    # reparameterized
        a = torch.tanh(u)
        # tanh Jacobian correction (numerically stable form)
        log_pi = normal.log_prob(u).sum(-1) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, log_pi

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s, a): return self.q(torch.cat([s, a], -1)).squeeze(-1)

env = gym.make("Pendulum-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_high = env.action_space.high[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

actor = Actor(obs_dim, act_dim).to(device)
q1, q2 = Critic(obs_dim, act_dim).to(device), Critic(obs_dim, act_dim).to(device)
q1_bar, q2_bar = Critic(obs_dim, act_dim).to(device), Critic(obs_dim, act_dim).to(device)
q1_bar.load_state_dict(q1.state_dict()); q2_bar.load_state_dict(q2.state_dict())

log_alpha = torch.zeros(1, device=device, requires_grad=True)
target_entropy = -float(act_dim)             # H_bar = -|A|

opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
opt_alpha = torch.optim.Adam([log_alpha], lr=3e-4)

buf = collections.deque(maxlen=10_000)
gamma, tau, batch = 0.99, 0.005, 256

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1 - tau); tp.data.add_(tau * sp.data)

total_steps = 0
for ep in range(300):
    s, _ = env.reset()
    ep_ret, done, truncated = 0, False, False
    while not (done or truncated):
        with torch.no_grad():
            a, _ = actor.sample(torch.as_tensor(s, dtype=torch.float32, device=device))
        a_np = (a * act_high).cpu().numpy()
        s2, r, done, truncated, _ = env.step(a_np)
        buf.append((s, a_np, r, s2, float(done)))
        s, ep_ret, total_steps = s2, ep_ret + r, total_steps + 1

        if len(buf) >= 1000:
            mb = random.sample(buf, batch)
            s_, a_, r_, s2_, d_ = map(np.array, zip(*mb))
            s_  = torch.as_tensor(s_,  dtype=torch.float32, device=device)
            a_  = torch.as_tensor(a_,  dtype=torch.float32, device=device) / act_high
            r_  = torch.as_tensor(r_,  dtype=torch.float32, device=device)
            s2_ = torch.as_tensor(s2_, dtype=torch.float32, device=device)
            d_  = torch.as_tensor(d_,  dtype=torch.float32, device=device)
            
            alpha = log_alpha.exp().detach()
            # critic update
            with torch.no_grad():
                a2, logp2 = actor.sample(s2_)
                y = r_ + gamma * (1 - d_) * (torch.min(q1_bar(s2_, a2), q2_bar(s2_, a2)) - alpha * logp2)
            l_q = ((q1(s_, a_) - y) ** 2).mean() + ((q2(s_, a_) - y) ** 2).mean()
            opt_q.zero_grad(); l_q.backward(); opt_q.step()
            
            # actor update (reparameterization)
            a_new, logp_new = actor.sample(s_)
            q_new = torch.min(q1(s_, a_new), q2(s_, a_new))
            l_actor = (alpha * logp_new - q_new).mean()
            opt_actor.zero_grad(); l_actor.backward(); opt_actor.step()
            
            # α update (dual descent on Lagrangian; SAC v2 Eq 18)
            # minimize l_alpha = -log_alpha · (logp + H_bar) → α grows when entropy below H_bar
            l_alpha = -(log_alpha * (logp_new.detach() + target_entropy)).mean()
            opt_alpha.zero_grad(); l_alpha.backward(); opt_alpha.step()
            
            # Polyak soft update
            soft_update(q1_bar, q1, tau); soft_update(q2_bar, q2, tau)
    
    if ep % 10 == 0:
        print(f"ep {ep:3d}  return {ep_ret:7.1f}  α {log_alpha.exp().item():.3f}")
```

**预期**：~150 episodes 把 Pendulum return 拉到 $-200$ 以上；$\alpha$ 自动收敛到合理范围（一般 0.1-0.3）。

**关键实现点对应 §10**：

- `actor.sample()` 里 `rsample()` 触发 reparameterization（不要用 `.sample()`！）
- tanh Jacobian 用数值稳定形式 $2 (\log 2 - u - \operatorname{softplus}(-2u))$
- α 用 $\log\alpha$ 参数化保正
- 两套 Q + min（TD3 trick 借过来）
- Polyak $\tau = 0.005$

## §B 一手资料

| 算法 / 概念 | 论文 | 一作 | 年 | 出处 | arXiv / DOI |
|---|---|---|---|---|---|
| REINFORCE | *Simple Statistical Gradient-Following Algorithms* | R. J. Williams | 1992 | *Machine Learning* 8 | DOI 10.1007/BF00992696 |
| CPI / 单调改进界 | *Approximately Optimal Approximate Reinforcement Learning* | S. Kakade, J. Langford | 2002 | ICML | — |
| DPG 定理 | *Deterministic Policy Gradient Algorithms* | D. Silver | 2014 | ICML | PMLR v32 |
| **DDPG** | *Continuous Control with Deep Reinforcement Learning* | T. Lillicrap | 2016 | ICLR | arXiv:1509.02971 |
| **TRPO** | *Trust Region Policy Optimization* | J. Schulman | 2015 | ICML | arXiv:1502.05477 |
| **GAE** | *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | J. Schulman | 2016 | ICLR | arXiv:1506.02438 |
| A3C / A2C | *Asynchronous Methods for Deep RL* | V. Mnih | 2016 | ICML | arXiv:1602.01783 |
| **PPO** | *Proximal Policy Optimization Algorithms* | J. Schulman | 2017 | arXiv | arXiv:1707.06347 |
| **TD3** | *Addressing Function Approximation Error in Actor-Critic Methods* | S. Fujimoto | 2018 | ICML | arXiv:1802.09477 |
| **SAC v1** | *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* | T. Haarnoja | 2018 | ICML | arXiv:1801.01290 |
| **SAC v2** (auto-α) | *Soft Actor-Critic Algorithms and Applications* | T. Haarnoja | 2018-12 | arXiv | arXiv:1812.05905 |
| Discrete SAC | *Soft Actor-Critic for Discrete Action Settings* | P. Christodoulou | 2019 | arXiv | arXiv:1910.07207 |
| Implementation Matters | *Implementation Matters in Deep Policy Gradients* | L. Engstrom | 2020 | ICLR | arXiv:2005.12729 |
| 37 PPO Details | *The 37 Implementation Details of PPO* | S. Huang | 2022 | ICLR Blog Track | iclr-blog-track.github.io |
| RLHF（canonical） | *Learning to summarize from human feedback* | N. Stiennon | 2020 | NeurIPS | arXiv:2009.01325 |
| **InstructGPT** | *Training language models to follow instructions with human feedback* | L. Ouyang | 2022 | NeurIPS | arXiv:2203.02155 |
| Constitutional AI | *Constitutional AI: Harmlessness from AI Feedback* | Y. Bai | 2022 | arXiv | arXiv:2212.08073 |

**延伸阅读**：

- OpenAI Spinning Up（[spinningup.openai.com](https://spinningup.openai.com)）：每个算法 from-scratch + 数学推导
- CleanRL（[github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)）：单文件实现，PPO / SAC / TD3 全有；和 37 details 论文配套
- Lil'Log RL post-series（[lilianweng.github.io](https://lilianweng.github.io)）：理论扎实的中英文笔记
