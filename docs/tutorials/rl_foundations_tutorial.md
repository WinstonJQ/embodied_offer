## §0 TL;DR Cheat Sheet

> 💡 **6 句话搞定强化学习基础** — 从 MDP 到 DQN 全家桶 + 策略梯度，**面试官最爱追问的一条主线**。

1. **MDP / Bellman**：$V^\pi$ 满足 *Bellman expectation*（带 $\sum_a \pi$），$V^*$ 满足 *Bellman optimality*（带 $\max_a$）。两个算子都是 sup-norm 下的 **γ-收缩**，所以 VI / PI 都收敛。

2. **DP 解**：VI 一步合并 evaluation + improvement，**几何收敛**（永远不到点）；PI 评估 → 改进交替，**有限步收敛**（策略空间有限 $\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$）。

3. **MC vs TD**：MC 用 episode return $G_t$（无偏，高方差）；TD(0) bootstrapping $r + \gamma V(s')$（**有偏**，低方差）。n-step 在两者之间调旋钮。

4. **on/off-policy + IS**：on = 行为策略 $\pi_b$ = 目标策略 $\pi_t$；off = 不等，需要 importance sampling $\rho_t = \pi_t/\pi_b$。轨迹 IS 权重是乘积 $\prod_t \rho_t$，方差随 horizon **指数级炸开**。Q-learning 用 max 算子 *吸收* IS，DQN/SAC 直接用回放——这是面试官最爱挖的坑。

5. **DQN 三件套**：experience replay（去相关）+ target network（稳定 bootstrap target）+ reward clipping（统一尺度）。改进家族：DDQN（修 max bias）+ Dueling（V+A 分解）+ PER（按 TD error 采样）。

6. **PG 定理 → A2C**：$\nabla_\theta J = \mathbb{E}[\nabla \log \pi_\theta(a\mid s) \cdot Q^\pi(s,a)]$，加 baseline $V^\pi$ 不影响均值但降方差 → advantage $A = Q - V \approx \delta^V_t$（TD residual）→ A2C。

> ✅ **快速记忆口诀** — 每个 trick 解决一个具体方差/偏差问题。

- 折扣 γ = **horizon 旋钮**（$1/(1-\gamma)$ 是有效视野）
- replay buffer = **去相关 + 样本复用**
- target net = **不让 bootstrap target 追自己**
- baseline = **降方差不引入偏**
- IS clip / truncation = **bias-variance 折中**

## §1 MDP 与 Bellman 方程

### 1.1 MDP 六元组

马尔可夫决策过程 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0)$（教科书有时写五元组省略 $\rho_0$）：

- $\mathcal{S}$：状态空间；$\mathcal{A}$：动作空间
- $P(s' \mid s, a)$：转移概率
- $r(s, a)$ 或 $r(s, a, s')$：奖励函数（约定 $r_{t+1}$ 是在 $s_t$ 取 $a_t$ 后收到的奖励）
- $\gamma \in [0, 1)$：折扣因子
- $\rho_0$：初始状态分布

**马尔可夫性**：$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)$ —— 未来只依赖当前。

### 1.2 Bellman expectation equation

固定策略 $\pi(a \mid s)$，价值函数定义为期望折扣回报：

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^\infty \gamma^k r_{t+k+1} \,\middle|\, s_t = s\right]$$

展开得到 *Bellman expectation*：

$$\boxed{\;V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^\pi(s')\bigr]\;}$$

$$Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\Bigl[r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\Bigr]$$

紧凑算子记号：$V^\pi = T^\pi V^\pi$，称 $T^\pi$ 为 **策略评估算子**（**仿射**：$T^\pi V = R^\pi + \gamma P^\pi V$，去掉常数 $R^\pi$ 后才是线性的）。

### 1.3 Bellman optimality equation

把 $\sum_a \pi$ 换成 $\max_a$，得到最优价值：

$$\boxed{\;V^*(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^*(s')\bigr]\;}$$

$$Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a)\Bigl[r + \gamma \max_{a'} Q^*(s', a')\Bigr]$$

算子记号 $V^* = T^* V^*$。最优策略由 $\pi^*(s) = \arg\max_a Q^*(s, a)$ 抽取。

> ⚠️ **常见混淆** — $T^\pi$ 是 *仿射* 算子（$T^\pi V = R^\pi + \gamma P^\pi V$，含常数 $R^\pi$），$T^*$ 是 *非线性* 算子（$\max$ 引入非线性）。**评估方程** $(I - \gamma P^\pi) V = R^\pi$ 是线性方程组（把仿射不动点改写得到），所以 PI 评估能解析解；VI 在 $T^*$ 下只能迭代。

### 1.4 收缩性质（contraction）

定义 sup-norm $\lVert V \rVert_\infty = \max_s \lvert V(s) \rvert$。对任意两个价值函数 $V_1, V_2$：

$$\lVert T^\pi V_1 - T^\pi V_2 \rVert_\infty \le \gamma \lVert V_1 - V_2 \rVert_\infty$$
$$\lVert T^* V_1 - T^* V_2 \rVert_\infty \le \gamma \lVert V_1 - V_2 \rVert_\infty$$

由于 $\gamma < 1$，两个算子都是 **γ-收缩**。$\mathbb{R}^{\lvert S\rvert}$ 在 sup-norm 下完备，由 **Banach 不动点定理**：存在唯一不动点，且 $V_{k+1} = T V_k$ 几何速率收敛。

**$T^*$ 收缩性的关键引理**：$\lvert \max_a f(a) - \max_a g(a)\rvert \le \max_a \lvert f(a) - g(a)\rvert$。代入展开即得（详见 §12 L3-1）。

> 💡 **为什么 γ 选 0.99** — 有效视野 $\approx 1/(1-\gamma) = 100$ 步，匹配典型 episode 长度；γ=1 让算子不再是 strict contraction，无收敛保证（除非 episode 必然终止）。

## §2 Value Iteration vs Policy Iteration

### 2.1 Value Iteration（VI）

单步更新合并 evaluation + improvement：

$$V_{k+1}(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V_k(s')\bigr]$$

终止条件 $\lVert V_{k+1} - V_k \rVert_\infty < \epsilon$。最终从 $V_K$ 抽取贪心策略 $\pi(s) = \arg\max_a Q_{V_K}(s, a)$。

**收敛**：$T^*$ 是 γ-收缩，所以 $\lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^*\rVert_\infty$，**几何收敛但永远不到点**。

策略误差界：$\lVert V^{\pi_k} - V^* \rVert_\infty \le \dfrac{2\gamma}{1-\gamma}\lVert V_k - V^* \rVert_\infty$。

### 2.2 Policy Iteration（PI）

两步交替：

1. **Policy evaluation**：解 $V^{\pi_k} = T^{\pi_k} V^{\pi_k}$（迭代到收敛 *或* 解线性系统 $(I - \gamma P^{\pi_k})V = R^{\pi_k}$）。
2. **Policy improvement**：$\pi_{k+1}(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^{\pi_k}(s')\bigr]$。

**收敛**：策略空间是有限集合 $\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$；由 **policy improvement theorem**，每次改进要么严格提升 $V$（某处 strict inequality），要么收敛 → 最多 $\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$ 步 **有限步收敛**。

### 2.3 横向对比

| 维度 | VI | PI |
|---|---|---|
| 每步计算 | 一次 max + 期望 | 评估 + 改进两步 |
| 收敛步数 | 几何（永远不到） | 有限步（finite MDP） |
| 评估代价 | 0 | $O(\lvert S\rvert^3)$ 解线性系统 *或* 多轮迭代 |
| 总成本 | 多次廉价迭代 | 少次昂贵迭代 |
| 实践 | tabular RL 默认 | 教科书常用，工程少用 |

> ⚠️ **Modified PI** — evaluation 不必算到完美，只跑 $m$ 步 $T^\pi$ 也能收敛（$m \to \infty$ 即 PI，$m=1$ 即 VI）。这是 deep RL 里 actor-critic 的雏形。

## §3 MC vs TD

### 3.1 Monte Carlo（first-visit）

对每个 episode 算回报 $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$，对状态 $s$ 的所有首次访问取平均：

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl[G_t - V(s)\bigr]$$

- **无偏**：$\mathbb{E}[G_t] = V^\pi(s)$
- **高方差**：$G_t$ 聚合了从 $t$ 到 episode 末所有 reward / transition / policy 的随机性

### 3.2 TD(0)

一步 bootstrapping：

$$\boxed{\;V(s_t) \leftarrow V(s_t) + \alpha\underbrace{\bigl[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\bigr]}_{\delta_t\;\text{(TD error)}}\;}$$

- **有偏**（while $V \ne V^\pi$）：bootstrap 的 $V(s_{t+1})$ 自己也是估计
- **低方差**：每次只用一步的随机性

### 3.3 n-step TD

$$G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n} + \gamma^n V(s_{t+n})$$

更新 $V(s_t) \leftarrow V(s_t) + \alpha[G_t^{(n)} - V(s_t)]$。极限：$n=1$ 是 TD(0)，$n \to \infty$ 是 MC。

**λ-return**（有限期 episodic 版）把 $n$-step 加权平均到 episode 终止：

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_t^{(n)} + \lambda^{T-t-1} G_t$$

其中 $G_t = \sum_{k=t}^{T-1}\gamma^{k-t} r_{k+1}$ 是终止 MC 回报。$\lambda$ 是 bias-variance 调节钮：$\lambda=0$ 全 bootstrap（TD(0)）；$\lambda=1$ 退化为 $G_t$（MC）。连续任务 ($T\to\infty$) 退化为 $G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$（仅对 $\lambda < 1$ 成立）。

### 3.4 Bias-variance 总结表

| 方法 | 偏 | 方差 | 在线学习 | bootstrapping |
|---|---|---|---|---|
| MC | 无偏 | 高 | 否（需 episode 终止） | 否 |
| TD(0) | 有偏 | 低 | 是 | 是（1 步） |
| n-step TD | 中间 | 中间 | 是 | 是（n 步） |
| TD(λ) | 中间（受 λ 控） | 中间 | 是（eligibility traces） | 是 |

> 💡 **直觉** — TD 像"每走一步就用现有信念修一次"，MC 像"走完一整局再总结"。TD 学得快但可能学偏，MC 学得慢但忠于事实。

## §4 on-policy vs off-policy + Importance Sampling

> ⚠️ **本节为新增独立章节** — 面试官最爱追问的概念线。不要把它和 §5 Q-learning vs SARSA 混在一起讲。

### 4.1 定义

- **Behavior policy** $\pi_b(a \mid s)$：*生成* 数据的策略
- **Target policy** $\pi_t(a \mid s)$（或 $\pi_\theta$）：想要 *评估* 或 *优化* 的策略
- **On-policy**：$\pi_b \equiv \pi_t$（数据每轮重新从当前学习策略采样）
- **Off-policy**：$\pi_b \not\equiv \pi_t$（数据来自旧策略 / 经验回放 / 专家 / 随机探索）

> ⚠️ **常见混淆** — "off-policy" $\ne$ "用 replay buffer"。DQN 在 $\varepsilon$-greedy 下采样、$\max$ 算子做 bootstrap，**即使没有 replay 也是 off-policy**——因为 bootstrap target 与采样行为不一致。

### 4.2 通用 importance sampling 估计

要估 $\mu = \mathbb{E}_{x \sim p}[f(x)]$ 但只能从 $q$ 采样：

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\!\left[\frac{p(x)}{q(x)} f(x)\right]$$

权重 $w(x) = p(x)/q(x)$ 称为 **IS weight**。**前提**：$\operatorname{supp}(p) \subseteq \operatorname{supp}(q)$（绝对连续）；否则估计 undefined。

**Ordinary IS**（无偏）：

$$\hat\mu_{\text{OIS}} = \frac{1}{N}\sum_{i=1}^N \frac{p(x_i)}{q(x_i)} f(x_i)$$

方差可以 **无界**——只要 $\mathbb{E}_q[(p/q)^2 f^2] = \infty$。

**Weighted IS (WIS)**（有偏、低方差、一致）：

$$\hat\mu_{\text{WIS}} = \frac{\sum_i w(x_i) f(x_i)}{\sum_i w(x_i)}$$

WIS 偏 $O(1/N)$；当 $f$ 有界时方差也有界（rewards bounded ⇒ returns bounded ⇒ WIS variance bounded by $\sup_x \lvert f(x)\rvert^2$）。实际工程几乎一律用 WIS。

> ⚠️ **必背区分** — *unbiased* ≠ *consistent*。OIS 无偏；WIS 有偏但 consistent（$\hat\mu \xrightarrow{a.s.} \mu$）。面试官爱挖这个。

### 4.3 轨迹 IS 比

轨迹 $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$ 由 $\pi_b$ 采样（共 $T$ 个动作 $a_0, \dots, a_{T-1}$ 与 $T$ 个奖励 $r_1, \dots, r_T$）。转移 $P$ 与 $\rho_0$ 在 $\pi_b / \pi_t$ 下相同，**在比值中消掉**：

$$\rho_{0:T-1} = \prod_{t=0}^{T-1} \frac{\pi_t(a_t \mid s_t)}{\pi_b(a_t \mid s_t)}$$

$\text{Var}[\rho_{0:T-1}]$ 随 horizon **乘性增长** —— 一步 $\pi_b(a_t)$ 极小就让权重爆炸。

**Per-decision IS**（Precup, Sutton, Singh 2000）只对 reward $r_{t+1}$ 用 prefix ratio $\rho_{0:t} = \prod_{k=0}^t \rho_k$：

$$\hat V_{\text{PDIS}}^{\pi_t}(s_0) = \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T_i-1} \gamma^t\, \rho_{0:t}^{(i)}\, r_{t+1}^{(i)}$$

省一些方差——但仍可能爆炸。

### 4.4 截断 IS

把每步 ratio 截断在 $c$ 以下：

$$\rho_{0:t}^{(c)} = \prod_{k=0}^t \min\!\left(c,\;\frac{\pi_t(a_k \mid s_k)}{\pi_b(a_k \mid s_k)}\right)$$

**引入偏换方差**——bias-variance 调节钮。$c$ 通过 held-out OPE MSE 选。

### 4.5 算法实例（必背）

#### (a) Q-learning：off-policy，**无显式 IS**

$$y_t^{\text{QL}} = r_{t+1} + \gamma \max_{a'} Q_\theta(s_{t+1}, a')$$

$\max$ 算子选的是 **target 策略** 的动作，与 behavior 选什么无关 → IS ratio 退化为 indicator，被 $\max$ 吸收。

> ⚠️ **trick 仅在 tabular / 离散动作下成立**。函数近似 + bootstrap + off-policy 同时出现 → 著名的 **deadly triad**（Sutton & Barto Ch 11），收敛性不再保证；DQN 的 target net + replay 都是工程补丁。

#### (b) SARSA：on-policy，IS = 1

$$y_t^{\text{SARSA}} = r_{t+1} + \gamma Q_\theta(s_{t+1}, a_{t+1}),\quad a_{t+1} \sim \pi_b(\cdot \mid s_{t+1})$$

bootstrap 用的 $a_{t+1}$ 是行为策略真正采的下一步动作 → $\pi_b = \pi_t$，无需 IS。

> 💡 **后果** — SARSA 学的是 *$\varepsilon$-greedy 策略* 的 $Q^\pi$（**包含探索代价**）；Q-learning 学的是 *greedy 策略* 的 $Q^*$（不管 $\varepsilon$）。Cliff walking 中 SARSA 选安全路、Q-learning 选悬崖边——根因在此。

#### (c) PPO：*近似* on-policy

单步 IS 比：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

clip 起来作 surrogate：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\bigl(r_t \hat A_t,\;\operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat A_t\bigr)\right]$$

外层每轮重采（on-policy），内层多 epoch 在同一 batch 上更新（off-policy）→ **这就是为什么需要 IS 修正**。

#### (d) DDPG / TD3 / SAC：off-policy，**无显式 IS**

- DDPG/TD3 走 deterministic PG：$\nabla_\phi J \approx \mathbb{E}[\nabla_a Q(s, a)\rvert_{a = \mu_\phi(s)} \nabla_\phi \mu_\phi(s)]$ → 链式法则过 $Q$，不需要 $\nabla \log \pi$
- SAC 走 reparameterization：$a = f_\phi(\varepsilon; s)$，每次采 *新* $\varepsilon$，期望对 $\varepsilon$ 而非 replayed $a$ → 不需 IS

> ⚠️ **代价** — "no IS" $\ne$ "no bias"。DPG 静默假设 $Q_\theta$ 在 $\mu_\phi$ 可能查询的所有点都准——replay buffer 经常违反这点。

#### (e) RLHF-PPO：on-policy w.r.t. $\pi_\theta$，**KL to $\pi_{\text{ref}}$**

每 token reward：

$$\tilde r_t = r_\phi(s_t, a_t) - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}$$

log-ratio 是 **KL 正则项** $\beta \cdot \text{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}})$，**不是 importance sampling**。$\pi_{\text{ref}}$ 是冻结的 SFT 模型作正则锚，不是 behavior policy。rollout 仍从 $\pi_\theta$ 采，所以外层 on-policy + 内层走 §4.5(c) 的 PPO 公式。

> ⚠️ **面试坑** — 不要把 RLHF 里的 $\log \pi_\theta / \pi_{\text{ref}}$ 叫 IS——它是 KL penalty。

#### (f) IMPALA V-trace

双截断 ratio：

$$\rho_t = \min\!\left(\bar\rho,\;\frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}\right),\quad c_t = \min\!\left(\bar c,\;\frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}\right),\quad \bar c \le \bar\rho$$

V-trace target：

$$v_s = V(s_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s}\Bigl(\prod_{i=s}^{t-1} c_i\Bigr) \rho_t \delta_t V$$

其中 $\delta_t V = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$。**两个阈值各管一件事**：$\bar\rho$ 控制 fixed-point（target 策略），$\bar c$ 控制 multi-step 方差。**当 clipping 实际生效**（即某些 ratio 被 $\bar\rho$ 截断），收敛的是 *interpolated* 策略 $\pi_{\bar\rho}$，不是原 $\pi$；若 $\bar\rho$ 足够大使 clipping 从不 active，仍收敛到 $\pi$。IMPALA 默认 $\bar\rho = 1$ 通常 active。

#### (g) Retrace(λ)

$$c_t = \lambda \min(1, \rho_t)$$

Cap 在 1 保证算子对任意 behavior $\mu$ 都是 γ-contraction（Munos et al 2016 Thm 1）→ **safe**。代价：$\pi \gg \mu$ 时低权重浪费样本。

### 4.6 速查表

| 算法 | on/off | 显式 IS | 方差控制机制 |
|---|---|---|---|
| MC vanilla | on | 否 | 无 |
| Off-policy MC (OIS) | off | 是（full $\rho_{0:T-1}$） | 无（方差可无界） |
| Off-policy MC (WIS) | off | 是（self-norm） | 自归一化（牺牲偏） |
| Per-decision IS | off | 是（prefix） | 更短乘积 |
| **Q-learning / DQN** | off | **否**（max 吸收） | target net + replay + Huber |
| **SARSA** | on | 否（ratio = 1） | n/a |
| **PPO** | "近似 on" | 是（单步 $r_t(\theta)$） | clip + KL early-stop |
| TRPO | on（每步） | 是（单步） | hard KL trust region |
| **DDPG / TD3** | off | **否**（DPG 链式法则） | twin Q + delayed actor |
| **SAC** | off | **否**（reparameterization） | 熵 + twin Q |
| **RLHF-PPO** | on（rollout）+ inner PPO IS | 内层是；外层 KL（不是 IS） | clip + $\beta \cdot \text{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ |
| IMPALA (V-trace) | off | 是（截断 $\rho_t, c_t$） | 双阈值 |
| Retrace(λ) | off | 是（cap at 1） | γ-contraction 保证 |
| Doubly-Robust OPE | off（评估） | 是 + model | 控制变量降方差 |

## §5 Q-learning vs SARSA

### 5.1 更新式对照

**Q-learning**（Watkins 1989，off-policy TD control）：

$$\boxed{\;Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\Bigl[r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\Bigr]\;}$$

**SARSA**（Rummery & Niranjan 1994，on-policy TD control）：

$$\boxed{\;Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\Bigl[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\Bigr]\;}$$

名字来自五元组 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$。

### 5.2 Cliff Walking（Sutton & Barto Ex 6.6）

```
┌─────────────────────────────┐
│ S  ·  ·  ·  ·  ·  ·  ·  ·  G │   每步 -1
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  · │   跌入 cliff → -100 并回 S
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  · │   ε-greedy, ε = 0.1
│ S [cliff ───────────────] G │
└─────────────────────────────┘
```

- **Q-learning** 学到 *最优* 策略——贴着 cliff 边走（最短路）。但训练中 $\varepsilon$-greedy 偶尔失足跌下去，per-episode reward 很差。
- **SARSA** 学到 *更安全* 的策略——绕远路走顶端。因为 SARSA 的 $Q(s', a')$ 用的是 $\varepsilon$-greedy 实际下一步，cliff 边 $Q$ 被惩罚（探索时可能掉下去）。

> 💡 **关键直觉** — Q-learning 学 "假如我从这里始终 greedy"，SARSA 学 "考虑到我会偶尔探索"。$\varepsilon \to 0$ 时两者收敛到同一最优策略。

### 5.3 收敛条件

**Q-learning**（Watkins & Dayan 1992）在以下条件下 $Q_t \xrightarrow{a.s.} Q^*$：

- 任意 behavior $\pi_b$ 使每个 $(s, a)$ **无限次访问**
- Robbins-Monro 步长：$\sum_t \alpha_t(s,a) = \infty,\;\sum_t \alpha_t^2(s,a) < \infty$
- 奖励有界

**SARSA** 同样需要上述条件 *加* 策略 **GLIE**（Greedy in the Limit with Infinite Exploration，例如 $\varepsilon_t \to 0$ at appropriate rate）才能收敛到 $Q^*$。

## §6 DQN — 深度 Q 网络的三件套

> 📌 **论文** — Mnih et al, *"Playing Atari with Deep Reinforcement Learning"*, NeurIPS DLW **2013**, arXiv:1312.5602；后续 Nature 版 *"Human-level control through deep reinforcement learning"*, *Nature* **518**:529–533, 2015, DOI 10.1038/nature14236。

### 6.1 为什么不能直接把 Q-learning 套上神经网络

直接套，三件事一起出现就崩：

1. **样本相关性**：Atari 连续帧高度相关，违反 SGD 的 i.i.d. 假设
2. **target 移动**：每步更新都改变 $Q_\theta$，bootstrap target $r + \gamma \max Q_\theta(s')$ **追自己尾巴** → 发散
3. **奖励尺度异构**：49 个 Atari 游戏奖励量级从 ±1 到 $10^5$，单一学习率 / 网络架构搞不定

### 6.2 三件套

1. **Experience replay**：环 buffer $\mathcal{D}$（容量 $\sim 10^6$ transitions），SGD 时从 $\mathcal{D}$ 均匀采小批量
   - **样本效率**：每条 transition 可重复用
   - **去相关**：随机采样打破时间相关性

2. **Target network**：复制一份参数 $\theta^-$，target 用 $\theta^-$ 算；每 $C$ 步硬拷贝 $\theta^- \leftarrow \theta$（Nature: $C = 10{,}000$）
   - 每个 batch 变成"对固定 target 的监督回归"，避免追尾
   - 软更新 alternative：$\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$，$\tau \approx 0.005$（DDPG/SAC 用）

3. **Reward clipping**：$r \to \operatorname{sign}(r) \in \{-1, 0, +1\}$
   - 让一组超参跨所有游戏可用
   - 代价：丢失奖励大小信息（后续 Pop-Art / 分布式 RL 修这个）

### 6.3 DQN loss

$$\boxed{\;L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\!\left[\Bigl(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\Bigr)^2\right]\;}$$

Nature 版还对 TD 误差用 Huber-like 梯度 clipping（$\pm 1$）。

### 6.4 Nature 版超参速查

| 超参 | 值 |
|---|---|
| Replay buffer 容量 | 1M transitions |
| Minibatch | 32 |
| $\gamma$ | 0.99 |
| 学习率 | $2.5 \times 10^{-4}$（RMSProp）|
| Target net 同步周期 $C$ | 10,000 env steps |
| $\varepsilon$ 退火 | $1.0 \to 0.1$，前 1M 帧线性 |
| 训练帧数 | 50M / 游戏 |
| Frame skip | 4 |

## §7 DQN 改进家族 —— DDQN / Dueling / PER（Rainbow 一句话）

### 7.1 Double DQN（DDQN）

> 📌 **论文** — van Hasselt, Guez, Silver, *"Deep Reinforcement Learning with Double Q-Learning"*, AAAI **2016**, arXiv:1509.06461。

**问题**：$\max_{a'} Q(s', a')$ 同时承担 *选择* 和 *评估*。当 $Q$ 含噪声，Jensen 不等式给出

$$\mathbb{E}[\max_a Q(s, a)] \ge \max_a \mathbb{E}[Q(s, a)]$$

→ **正向偏差**（overestimation bias）随 bootstrap 累积。

**修法**：decouple 选择 vs 评估。**online 网选**，**target 网评**：

$$a^* = \arg\max_{a'} Q_\theta(s', a'),\qquad y^{\text{DDQN}} = r + \gamma Q_{\theta^-}(s', a^*)$$

两个估计器去相关 → bias 显著降低。

> ⚠️ **DDQN 没完全消 bias** —— online / target 共享数据 + 架构，仍有残余相关噪声；部分 Atari 游戏甚至 *underestimate*。**针对 max bias 本身**的进一步修法：**Clipped Double Q-learning**（TD3，用 $\min$ over twin Q）/ **Maxmin Q-learning**（min over ensemble）/ **REM**。distributional RL（C51 / QR-DQN）正交方向——它建模回报分布而非修 max bias，但实证能稳定训练，与 DDQN 配合时效果叠加。

### 7.2 Dueling DQN

> 📌 **论文** — Wang et al, *"Dueling Network Architectures for Deep RL"*, ICML **2016** best paper, arXiv:1511.06581。

共享 conv trunk，分两个 head：value $V(s)$ 与 advantage $A(s, a)$，合成：

$$Q(s, a) = V(s) + \Bigl(A(s, a) - \frac{1}{\lvert\mathcal{A}\rvert}\sum_{a'} A(s, a')\Bigr)$$

**为什么减均值**：$V$ 与 $A$ 从 $Q$ 中 *不可识别*——$(V + c, A - c)$ 给出同样 $Q$。减均值（或减 max）pin down 唯一分解。

```
┌──── Conv trunk ────┐
│  (shared encoder)   │
└────┬────────────┬───┘
     │            │
     ▼            ▼
  V(s)         A(s, ·)
   │              │
   └───┬──────────┘
       ▼
  Q(s, a) = V(s) + A(s, a) - mean(A)
```

**好处**：很多动作 Q 接近时（如奖励几乎只依赖 $s$，不依赖 $a$），网络可从大量 transitions 学 $V$，不必每个 $(s, a)$ 单独学。

### 7.3 Prioritized Experience Replay（PER）

> 📌 **论文** — Schaul, Quan, Antonoglou, Silver, *"Prioritized Experience Replay"*, ICLR **2016**, arXiv:1511.05952。

按 TD error 优先采样 transition $i$：

$$P(i) \propto p_i^\alpha,\quad p_i = \lvert\delta_i\rvert + \epsilon$$

$\alpha \in [0, 1]$ 调"优先度多狠"（$\alpha = 0$ 退化为均匀）。

**问题**：采样有偏 → 梯度估计有偏。

**修法**：importance-sampling 权重

$$w_i = \Bigl(\frac{1}{N \cdot P(i)}\Bigr)^\beta \Bigm/ \max_j w_j$$

$\beta$ 从 0.4 **线性退火到 1.0**（训练后期才需要 unbiased）。

> 💡 **β 退火的直觉** — 早期 $Q$ 到处都错，bias 不重要；后期接近收敛，unbiased 梯度才让算法收到正确不动点。

### 7.4 Rainbow（一句话带过）

> 📌 **论文** — Hessel et al, *"Rainbow: Combining Improvements in Deep RL"*, AAAI **2018**, arXiv:1710.02298。

把 6 个正交改进堆一起：**DDQN + PER + Dueling + Multi-step return + Distributional RL (C51) + NoisyNet**。SOTA on Atari median human-normalized score；ablation 显示 **PER + multi-step + distributional** 贡献最大，Dueling / NoisyNet 弱。

> ⚠️ **面试常考** — 不要被 Rainbow 全堆吓住，记 3 件套（DDQN/Dueling/PER）+ Rainbow 是它们的 "all in"，够 95% 面试问。

## §8 策略梯度定理与 REINFORCE

### 8.1 目标函数

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \gamma^t r_{t+1}\right]$$

### 8.2 log-derivative trick

对任意 $p_\theta$：

$$\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}\!\bigl[f(x) \nabla_\theta \log p_\theta(x)\bigr]$$

推导：$\nabla \int p_\theta f = \int (\nabla p_\theta) f = \int p_\theta (\nabla \log p_\theta) f$。

### 8.3 策略梯度定理

> 📌 **论文** — Sutton, McAllester, Singh, Mansour, *"Policy Gradient Methods for RL with Function Approximation"*, NeurIPS **1999/2000**。

$$\boxed{\;\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(a \mid s) \cdot Q^{\pi_\theta}(s, a)\bigr]\;}$$

这里 $d^\pi$ 是 **unnormalized 折扣状态访问** $d^\pi(s) = \sum_{t=0}^\infty \gamma^t \Pr(s_t = s \mid \pi_\theta)$（Sutton & Barto 2018 Eq 13.6 约定；不除以 $1-\gamma$ 不归一化）。**关键**：梯度不需要对环境动力学 $P$ 或 $Q^\pi$ 求导——只对 $\log \pi$。

> ⚠️ **归一化约定坑** — 文献里 $d^\pi$ 有时定义为 normalized 概率分布 $(1-\gamma)\sum_t \gamma^t \Pr$，那样上式需要补一个 $\dfrac{1}{1-\gamma}$ 因子。本文一律用 unnormalized 形式，方便公式干净。

### 8.4 REINFORCE（Williams 1992）

用蒙特卡洛 return $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}$ 估 $Q$：

$$g_t = \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t$$

轨迹级形式：$g = \Bigl(\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Bigr) G(\tau)$。

**问题**：$G_t$ 聚合从 $t$ 到终止的所有随机性 → 方差爆炸。

### 8.5 Baseline 降方差

加任意 $b(s)$（不依赖 $a$）不改变梯度均值：

$$\mathbb{E}_a\!\bigl[\nabla \log \pi(a \mid s) \cdot b(s)\bigr] = b(s) \sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = b(s) \nabla \!\sum_a \pi = b(s) \nabla 1 = 0$$

→ $g_t = \nabla \log \pi(a_t \mid s_t) \cdot [G_t - b(s_t)]$ 仍无偏，但方差大降。

**最佳 baseline**（Greensmith et al 2004）：$b^* = \mathbb{E}[\lVert\nabla\log\pi\rVert^2 G] / \mathbb{E}[\lVert\nabla\log\pi\rVert^2]$；实践中用 state-value $V^\pi(s)$。这一替换直接通往 actor-critic。

## §9 Actor-Critic 与 A2C

### 9.1 Advantage 视角

把 baseline 选成 $V^\pi(s)$，得到 **advantage** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s, a}\!\bigl[\nabla_\theta \log \pi_\theta(a \mid s) \cdot A^\pi(s, a)\bigr]$$

**Actor** = 策略 $\pi_\theta$（policy gradient 更新）；**Critic** = 价值 $V_\phi$（用 MC 或 TD 监督学）。

Critic loss：$L_V(\phi) = (V_\phi(s_t) - \hat V_t)^2$，$\hat V_t$ 是 $G_t$ 或 n-step 目标。

### 9.2 A3C vs A2C

**A3C**（Mnih et al, ICML **2016**, arXiv:1602.01783）：多线程异步 worker，每个 worker 持本地 net + 一段环境，异步推梯度到中央 param server。异步本身去相关（不需 replay）。

**A2C**：同步版本（OpenAI baselines 推广，无独立论文）。master 等所有 worker 跑完一段，batch 起来一次更新，再 broadcast 新参。**实践中匹配或超过 A3C**：更简单、更 GPU 友好、可复现（无 race condition）。

> 💡 **后人才知道的事** — A3C 的"异步"其实是性能 hack 而非算法贡献；A2C 同步版证明只要并行 worker 够多，去相关效果同样好。

### 9.3 TD-residual advantage 与 GAE

单步 TD 估 $A$：

$$\hat A_t \approx \delta_t^V = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**注意**：这是 *biased* 估计（除非 $V_\phi = V^\pi$），偏来自 critic error。换 variance 拿 bias 的典型工程权衡。

**GAE**（Schulman et al ICLR **2016**, arXiv:1506.02438）把 n-step advantage 指数加权：

$$A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$$

$\lambda = 0$ 全 bootstrap（低 var、高 bias），$\lambda = 1$ 全 MC（高 var、给定准 critic 时无偏）。**$\lambda$ 是 bias-variance 调旋钮**——PPO 默认 $\lambda = 0.95$。

## §10 探索 vs 利用

| 方法 | 公式 | 优点 | 缺点 |
|---|---|---|---|
| **$\varepsilon$-greedy** | $\Pr(\text{random}) = \varepsilon$ | 简单，全局保证 | 探索无方向 |
| **Softmax / Boltzmann** | $\pi(a) = \exp(Q/T) / \sum$ | 按 $Q$ 加权 | 对 $Q$ 绝对值敏感 |
| **Entropy bonus** | $L \to L - \beta H[\pi]$ | 防止策略过早 collapse | 需调 $\beta$（A3C/PPO: 0.01） |
| **NoisyNet** | $y = (\mu + \sigma \odot \epsilon) x + \dots$ | 状态相关探索，自动退火 | 多参，比 $\varepsilon$ 复杂 |
| **UCB** | $\arg\max [\hat\mu + c\sqrt{\ln t / N}]$ | 理论 $O(\sqrt{T \log T})$ regret | 需要 visit count |
| **Thompson Sampling** | 从 posterior 采 $\tilde Q$ | Bayes optimal | 需要 posterior |

> 💡 **NoisyNet 一句话** — 把线性层 $Wx + b$ 换成参数自身带噪声 $(\mu^W + \sigma^W \odot \epsilon^W) x$，$\sigma$ 在自信处自动收缩——比 $\varepsilon$-greedy 更"聪明"，但工程价值有限，Rainbow 消融里贡献最小。

> ⚠️ **退火 schedule** — Atari DQN: $\varepsilon$ 线性 $1.0 \to 0.1$ 在前 1M 帧；评估时进一步降到 $\varepsilon = 0.05$。entropy bonus 一般不退火（除非任务很短）。

## §11 关键直觉与陷阱

| # | 陷阱 | 正解 |
|---|---|---|
| 1 | "expectation Bellman 和 optimality Bellman 都是线性的吧？" | **半对**。$T^\pi$ 是 *仿射*（$T^\pi V = R^\pi + \gamma P^\pi V$，含常数 $R^\pi$），评估方程 $(I-\gamma P^\pi)V = R^\pi$ 改写后是线性方程组；$T^*$ 因 $\max_a$ **非线性**。 |
| 2 | "γ 越接近 1 越好吧？" | **错**。$\gamma \to 1$ 让算子失去 strict contraction；折扣对应有效视野 $1/(1-\gamma)$。Atari 选 0.99 是匹配 episode 长度。 |
| 3 | "DQN 用 $\varepsilon$-greedy 采样，所以是 on-policy 吧？" | **错**。off-policy 与否看 *bootstrap target* 是否依赖行为策略——$\max$ 算子让 target 独立于 $\varepsilon$，所以 DQN off-policy。 |
| 4 | "Double DQN 完全消了 max bias 吧？" | **错**。仅显著降低；online/target 网共享数据 + 架构，残余相关噪声仍在。某些游戏 DDQN 反而 *underestimate*。 |
| 5 | "DQN 直接拿来做连续控制就行吧？" | **错**。$\max_a Q(s, a)$ 在连续 $a$ 上是 nonconvex 优化，每步算不动。解：DDPG / SAC 用 policy net；NAF 用 quadratic Q。 |
| 6 | "PER β 设 1.0 一开始就 unbiased 最稳吧？" | **错**。早期 $Q$ 到处错，bias 不是主矛盾；β 退火 0.4 → 1.0 是为了"早期偏一点学得快，后期 unbiased 收得对"。 |
| 7 | "REINFORCE 高方差只能靠 baseline 修？" | **半对**。baseline + bootstrapping（actor-critic）+ 奖励归一化 + GAE 全用上才稳。 |
| 8 | "A2C 的 advantage = $\delta^V_t$ 是无偏的吧？" | **错**。仅当 critic $V_\phi = V^\pi$ 才无偏；实际 critic 误差引入偏差，这是 actor-critic 用偏差换方差的代价。 |
| 9 | "RLHF 里 $\log \pi_\theta / \pi_{\text{ref}}$ 是 importance sampling 吧？" | **错**。这是 KL 正则项 $\beta \cdot \text{KL}(\pi_\theta \Vert \pi_{\text{ref}})$；rollout 仍从 $\pi_\theta$ 采，不是 off-policy 校正。 |
| 10 | "Cliff Walking 中 SARSA 始终比 Q-learning 好？" | **错**。SARSA 在训练中 reward 高（safer），但 Q-learning 学的是真正最优策略；$\varepsilon \to 0$ 两者重合。 |

## §12 25 道高频面试题

### L1 — 必会（10 题）

<details>
<summary><strong>1. 写出 $V^\pi$ 的 Bellman expectation 方程。</strong></summary>

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^\pi(s')\bigr]$$

紧凑形式 $V^\pi = T^\pi V^\pi$。
</details>

<details>
<summary><strong>2. $V^\pi$ 与 $V^*$ 区别？</strong></summary>

$V^\pi$ 是给定（可能次优）策略 $\pi$ 下的价值；$V^*(s) = \max_\pi V^\pi(s)$ 是所有策略下的最优价值。后者对应非线性算子 $T^*$（带 max），前者对应 *仿射* 算子 $T^\pi V = R^\pi + \gamma P^\pi V$（评估方程改写后是线性方程组）。
</details>

<details>
<summary><strong>3. 写 TD(0) 的更新公式。</strong></summary>

$V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$。括号内是 TD 误差 $\delta_t$。
</details>

<details>
<summary><strong>4. γ 控制什么？为什么常选 0.99？</strong></summary>

γ 是折扣，等效有效视野 $\approx 1/(1-\gamma)$。γ = 0.99 ↔ 100 步视野，匹配典型 episode 长度；γ < 1 还保证 Bellman 算子是收缩。γ = 1 让算子失去 strict contraction，无收敛保证（除非 episode 必然终止）。
</details>

<details>
<summary><strong>5. Q-learning vs SARSA，哪个 on-policy？</strong></summary>

**SARSA on-policy**——bootstrap 用实际下一步动作 $a_{t+1} \sim \pi$；Q-learning 用 $\max_{a'}$，与 behavior 无关，**off-policy**。
</details>

<details>
<summary><strong>6. 列 DQN 的 3 个稳定技巧。</strong></summary>

① **experience replay**（去相关 + 样本复用）；② **target network**（稳定 bootstrap target）；③ **reward clipping** to $[-1, +1]$（统一尺度）。
</details>

<details>
<summary><strong>7. experience replay 为什么有用？</strong></summary>

两点：① **样本效率** —— 每条 transition 反复用；② **去相关** —— Atari 连续帧高度相关，违反 SGD 的 i.i.d. 假设，随机采样打破时间相关。
</details>

<details>
<summary><strong>8. MC 和 TD 的核心区别？</strong></summary>

- **MC** 用 episode 完整 return $G_t$，**无偏**但高方差；需要 episode 终止才能更新
- **TD(0)** 用 bootstrap $r + \gamma V(s_{t+1})$，**有偏**（while $V \ne V^\pi$）但低方差；可以 online 更新

n-step / TD(λ) 在两者间调旋钮。
</details>

<details>
<summary><strong>9. Actor 和 Critic 各负责什么？</strong></summary>

**Actor** = 策略 $\pi_\theta$，靠 policy gradient 更新；**Critic** = 价值 $V_\phi$（或 $Q_\phi$），估 advantage 给 actor 做 baseline，降梯度方差。
</details>

<details>
<summary><strong>10. $\varepsilon$-greedy 为什么要退火 $\varepsilon$？</strong></summary>

早期探索价值高（环境不熟），需要 $\varepsilon$ 大；后期 $Q$ 准了，需要 exploit 多于 explore，所以 $\varepsilon$ 退火。Atari DQN：$1.0 \to 0.1$ 在前 1M 帧线性，评估时降到 $0.05$。
</details>

### L2 — 进阶（10 题）

<details>
<summary><strong>11. 为什么 PI 有限步收敛、VI 只是渐近收敛？</strong></summary>

PI 在 *策略空间* 走，策略是有限集合 $\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$，每步严格改进（policy improvement theorem），最多 $\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}$ 步收敛。VI 在 *价值空间* 走，是 γ-收缩，几何速率收敛但永远不到点（只能任意接近）。
</details>

<details>
<summary><strong>12. 证：baseline $b(s)$ 不改变 policy gradient 的均值。</strong></summary>

$$\mathbb{E}_a\!\bigl[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)\bigr] = b(s) \sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)$$

利用 $\nabla \log p = \nabla p / p$：

$$= b(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) = b(s) \nabla_\theta \!\!\sum_a \pi_\theta(a \mid s) = b(s) \nabla_\theta 1 = 0$$
</details>

<details>
<summary><strong>13. DDQN target 写出来，并解释为什么修 max bias。</strong></summary>

$$y^{\text{DDQN}} = r + \gamma Q_{\theta^-}\!\bigl(s',\;\arg\max_{a'} Q_\theta(s', a')\bigr)$$

**online 网选**，**target 网评**。$Q$ 含噪声时 $\max$ 操作引入正向偏差（Jensen），decouple 选择与评估**显著降低**正向偏差——注意不是完全消除（理论上要两估计器 *独立* 才能消，online/target 共享数据架构仍有残余相关；某些游戏甚至会 *underestimate*）。
</details>

<details>
<summary><strong>14. Dueling DQN 为什么要减 advantage 的均值？</strong></summary>

$V$ 和 $A$ 从 $Q$ 中不可识别——对任意常数 $c$，$(V + c, A - c)$ 给出同一 $Q$。强制 $\hat A(s, a) = A(s, a) - \tfrac{1}{\lvert\mathcal{A}\rvert}\sum_{a'} A(s, a')$（除以 *动作集合大小*）让 $\hat A$ 均值 0 → 唯一分解。论文也比较过减 max，实证减均值更稳。
</details>

<details>
<summary><strong>15. 写 PER 的优先级 + IS 权重公式，解释 β 退火的原因。</strong></summary>

$P(i) \propto p_i^\alpha$，$p_i = \lvert\delta_i\rvert + \epsilon$。修偏权重 $w_i = (N \cdot P(i))^{-\beta} / \max_j w_j$。**β 从 0.4 退火到 1.0**：早期 $Q$ 全错，bias 不是主矛盾；后期接近收敛，unbiased 梯度才能收到正确不动点。
</details>

<details>
<summary><strong>16. REINFORCE 为什么高方差？actor-critic 怎么帮？</strong></summary>

$G_t$ 是从 $t$ 到 episode 末所有 reward 的折扣和，方差随 horizon 增长。Actor-critic 用 bootstrap $V_\phi$ 替换 $G_t$ → 用 n-step 估计降方差；加 baseline $V^\pi$ 不改 mean 但降 var；GAE 再加 $\lambda$ 调 bias-variance。
</details>

<details>
<summary><strong>17. on-policy vs off-policy 对比，PPO 在哪儿？</strong></summary>

- **on-policy**：$\pi_b = \pi_t$（SARSA、A2C、vanilla PG）—— 每轮重采新数据
- **off-policy**：$\pi_b \ne \pi_t$（Q-learning、DQN、DDPG、SAC）—— replay 或专家数据
- **PPO** 是 *近似* on-policy：外层每轮从 $\pi_{\theta_{\text{old}}}$ 采一批，**内层多 epoch 在同一批上更新**（off-policy），用单步 IS 比 $r_t(\theta)$ 加 clip 控制偏移
</details>

<details>
<summary><strong>18. 用 log-derivative trick 推 policy gradient。</strong></summary>

$$\nabla_\theta J = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_\tau[R(\tau) \nabla_\theta \log p_\theta(\tau)]$$

$p_\theta(\tau) = p(s_0) \prod_t p(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t)$，只有 $\pi_\theta$ 依赖 $\theta$：

$$\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

$$\nabla_\theta J = \mathbb{E}_\tau\!\Bigl[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)\Bigr]$$

环境动力学 $p(s_{t+1} \mid s_t, a_t)$ 不依赖 $\theta$ → 消掉。
</details>

<details>
<summary><strong>19. IS 权重为什么随 horizon 方差爆炸？怎么修？</strong></summary>

$\rho_{0:T-1} = \prod_{t=0}^{T-1} \rho_t$ 是乘积；log-variance 随 horizon **线性增长**，所以方差指数增长。修法：① WIS（self-normalize，引入小 bias 换有界方差）；② truncation / clipping（Retrace cap at 1、V-trace 双阈值）；③ per-decision IS 用更短 prefix；④ doubly robust 引入 control variate。
</details>

<details>
<summary><strong>20. 为什么 Q-learning 在 $\varepsilon$-greedy 下也能学到 $Q^*$？</strong></summary>

bootstrap target 用 $\max_{a'}$ 与 behavior 无关——学的就是 greedy target 策略。形式条件：① behavior 策略保证每 $(s, a)$ **无限次访问**（ε-greedy 在 ergodic / communicating MDP + 无限训练时间下成立）；② Robbins-Monro 步长 $\sum_t \alpha_t = \infty,\;\sum_t \alpha_t^2 < \infty$；③ 奖励有界。三者齐全 → $Q_t \xrightarrow{a.s.} Q^*$（Watkins & Dayan 1992）。
</details>

### L3 — 顶级 lab（5 题）

<details>
<summary><strong>21. 证：Bellman 最优算子 $T^*$ 是 sup-norm 下的 γ-收缩。</strong></summary>

对任意 $V_1, V_2$ 和任意 $s$：

$$\lvert (T^* V_1)(s) - (T^* V_2)(s) \rvert = \Bigl\lvert \max_a [R(s, a) + \gamma \mathbb{E}_{s'} V_1(s')] - \max_a [R(s, a) + \gamma \mathbb{E}_{s'} V_2(s')]\Bigr\rvert$$

引理 $\lvert \max_a f(a) - \max_a g(a)\rvert \le \max_a \lvert f(a) - g(a)\rvert$：

$$\le \gamma \max_a \mathbb{E}_{s'} \lvert V_1(s') - V_2(s')\rvert \le \gamma \max_a \mathbb{E}_{s'} \lVert V_1 - V_2 \rVert_\infty = \gamma \lVert V_1 - V_2 \rVert_\infty$$

取 $\max_s$：$\lVert T^* V_1 - T^* V_2 \rVert_\infty \le \gamma \lVert V_1 - V_2 \rVert_\infty$。$\mathbb{R}^{\lvert S\rvert}$ 在 sup-norm 下完备 → Banach 不动点定理给出唯一 $V^*$，且 VI 几何收敛。
</details>

<details>
<summary><strong>22. 严格推导 policy gradient 定理（含状态分布 $d^\pi$）。</strong></summary>

用 **unnormalized 折扣状态访问** $d^\pi(s) = \sum_{t=0}^\infty \gamma^t \Pr(s_t = s \mid \pi_\theta, s_0)$（Sutton & Barto 2018 约定，不除 $1-\gamma$）。对 $V^\pi$ 求梯度：

$$\nabla_\theta V^\pi(s) = \nabla_\theta \sum_a \pi_\theta(a \mid s) Q^\pi(s, a) = \sum_a \Bigl[\nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a) + \pi_\theta(a \mid s) \nabla_\theta Q^\pi(s, a)\Bigr]$$

第二项展开 $Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$，得到对 $V^\pi(s')$ 的递归——unroll 后是 $V^\pi$ 在所有后继状态的几何级数。求和后：

$$\nabla_\theta J(\theta) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) Q^\pi(s, a) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\!\bigl[\nabla_\theta \log \pi_\theta(a \mid s) Q^\pi(s, a)\bigr]$$

关键：$\nabla_\theta Q^\pi$ 在递归展开后被吸收成 $d^\pi$ 的几何权重，最终公式只剩 $\nabla \log \pi$。**注意**：若改用 normalized $\tilde d^\pi = (1-\gamma) d^\pi$ 作概率分布，则公式右侧要补 $\dfrac{1}{1-\gamma}$ 因子；本文统一用 unnormalized。
</details>

<details>
<summary><strong>23. 为什么 DDQN 只 *渐近* 降 bias，不能完全消？</strong></summary>

DDQN 把 $\max$ 拆成"online 选 / target 评"。**理论保证**（独立 Double Q-learning, van Hasselt 2010 + DDQN 2016）：当两个估计器 *严格独立* 时能消除 max 引入的过估偏差；DDQN 是 *近似* 版本（online 和 target 共享数据 + 架构），所以 **显著降低但不消除** 偏差。实践原因：① 共享导致残余相关；② 函数近似误差（过参 / 优化器偏差）是系统性的，不是零均噪声；③ Jensen 仅在 *估计* 噪声上抵消，对 *逼近* 误差不修。结果：DDQN 在某些 Atari 游戏甚至 *underestimate*。

**针对 max bias 的进一步修法**：① **Clipped Double Q-learning**（TD3）取 twin Q 的 $\min$；② **Maxmin Q-learning** / **REDQ** 在 $N$ 个 Q 的最小值上做 target；③ **REM** 随机线性组合 ensemble。**注意区分**：distributional RL（C51 / QR-DQN）不直接修 max bias——它建模回报分布而非降低过估，正交方向；但实证可与 DDQN/TD3 叠加。
</details>

<details>
<summary><strong>24. actor-critic 用 TD-residual 当 advantage 与 PG 定理 + state-value baseline 的形式关系？</strong></summary>

PG 定理：$\nabla J = \mathbb{E}[\nabla \log \pi \cdot Q^\pi]$。加 baseline $V^\pi$：$\nabla J = \mathbb{E}[\nabla \log \pi \cdot A^\pi]$，$A^\pi = Q^\pi - V^\pi$。

用单步 TD 估 $A$：$\hat A_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t^V$。

**关键**：$\hat A_t$ **无偏 iff $V_\phi = V^\pi$**；否则被 critic 误差 $V_\phi - V^\pi$ 偏掉。所以 actor-critic 是 PG 的 *有偏* 估计——bias 换 variance 的工程取舍。GAE 用 $A_t^{\text{GAE}(\gamma, \lambda)} = \sum_l (\gamma\lambda)^l \delta_{t+l}$ 给 bias-variance 一个 $\lambda$ 旋钮，$\lambda = 1$ 给定准 critic 时回到 MC 无偏。
</details>

<details>
<summary><strong>25. 比较 Retrace(λ) / V-trace / PPO clip 三种 off-policy 修正——各自收敛到什么？</strong></summary>

三者都是截断 IS、bias-variance 折中：

- **Retrace(λ)** (Munos et al 2016)：每步权重 $c_s = \lambda \min(1, \rho_s)$，cap 在 1 → 对任意 behavior $\pi_b$ 是 γ-contraction，收敛到 $Q^\pi$；control 中收敛到 $Q^*$。**安全保证最强**。
- **V-trace** (IMPALA, Espeholt et al 2018)：两个截断 $\rho_t = \min(\bar\rho, \pi/\mu)$ 控 fixed-point，$c_t = \min(\bar c, \pi/\mu)$ 控 multi-step 方差，$\bar c \le \bar\rho$。**当 clipping 实际生效**（即至少部分样本被 $\bar\rho$ 截断），收敛到 *interpolated* 策略 $\pi_{\bar\rho}$，**不是原 $\pi$**；若 $\bar\rho$ 大到 clipping 从不触发，仍收敛到 $\pi$。IMPALA 默认 $\bar\rho = 1$ 通常 active。
- **PPO clip** (Schulman et al 2017)：$r_t(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$ clip 到 $[1-\varepsilon, 1+\varepsilon]$。**不是 return 校正**，而是 **trust-region surrogate**——阻止策略大跳。无 formal off-policy 收敛保证，但实证有效（因为 $\pi_\theta$ 紧跟 $\pi_{\theta_{\text{old}}}$）。

三者对应不同哲学：Retrace 求"严格 safe"，V-trace 求"distributed 高吞吐"，PPO 求"工程简单 + trust-region"。
</details>

## §A 附录：from-scratch 代码

### A.1 100 行 DQN on CartPole

```python
import gymnasium as gym, torch, torch.nn as nn, random, collections
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act),
        )
    def forward(self, x): return self.net(x)

env = gym.make("CartPole-v1")
obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
device = "cuda" if torch.cuda.is_available() else "cpu"

online = QNet(obs_dim, n_act).to(device)
target = QNet(obs_dim, n_act).to(device)
target.load_state_dict(online.state_dict())
opt = torch.optim.Adam(online.parameters(), lr=1e-3)

buf = collections.deque(maxlen=50_000)
gamma, batch, eps, eps_min, eps_decay = 0.99, 64, 1.0, 0.05, 0.995
target_sync, total_steps = 500, 0

def act(state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    with torch.no_grad():
        q = online(torch.as_tensor(state, dtype=torch.float32, device=device))
    return int(q.argmax().item())

for ep in range(500):
    state, _ = env.reset()
    ep_reward, done, truncated = 0.0, False, False
    while not (done or truncated):
        a = act(state, eps)
        nxt, r, done, truncated, _ = env.step(a)
        buf.append((state, a, r, nxt, done))
        state, ep_reward, total_steps = nxt, ep_reward + r, total_steps + 1

        if len(buf) >= batch:
            mb = random.sample(buf, batch)
            s, a_, r_, s2, d = zip(*mb)
            s  = torch.as_tensor(s,  dtype=torch.float32, device=device)
            s2 = torch.as_tensor(s2, dtype=torch.float32, device=device)
            a_ = torch.as_tensor(a_, dtype=torch.long,    device=device)
            r_ = torch.as_tensor(r_, dtype=torch.float32, device=device)
            d  = torch.as_tensor(d,  dtype=torch.float32, device=device)

            q_pred = online(s).gather(1, a_.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target(s2).max(dim=1).values
                y = r_ + gamma * q_next * (1.0 - d)
            loss = F.smooth_l1_loss(q_pred, y)  # Huber, 同 DQN Nature
            opt.zero_grad(); loss.backward(); opt.step()

            if total_steps % target_sync == 0:
                target.load_state_dict(online.state_dict())

    eps = max(eps_min, eps * eps_decay)
    if ep % 20 == 0:
        print(f"ep {ep:3d}  reward {ep_reward:6.1f}  eps {eps:.3f}")
```

**预期**：CartPole-v1 在 ~200 episode 内稳定到 ≥ 475 reward。关键组件对应 §6：① `buf` = experience replay；② `target` net + `target_sync` = target network；③ `smooth_l1_loss` = Huber（替代 reward clipping，CartPole 奖励本身 ≤ 1 不需 clip）。

### A.2 REINFORCE with baseline on CartPole

```python
import gymnasium as gym, torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.pi  = nn.Linear(hidden, n_act)   # actor head
        self.v   = nn.Linear(hidden, 1)        # critic head (baseline)
    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)

env = gym.make("CartPole-v1")
obs_dim, n_act = env.observation_space.shape[0], env.action_space.n
device = "cuda" if torch.cuda.is_available() else "cpu"

net = PolicyValueNet(obs_dim, n_act).to(device)
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
gamma = 0.99

def discounted_returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.insert(0, G)
    return out

for ep in range(1000):
    state, _ = env.reset()
    log_probs, values, rewards, entropies = [], [], [], []
    done, truncated = False, False
    while not (done or truncated):
        x = torch.as_tensor(state, dtype=torch.float32, device=device)
        logits, v = net(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        values.append(v)
        entropies.append(dist.entropy())                     # 整条轨迹的 entropy
        state, r, done, truncated, _ = env.step(int(a.item()))
        rewards.append(r)

    G = torch.as_tensor(discounted_returns(rewards, gamma),
                         dtype=torch.float32, device=device)
    V = torch.stack(values)
    advantage = (G - V).detach()              # 无偏需 detach
    pg_loss = -(torch.stack(log_probs) * advantage).mean()
    v_loss  = F.smooth_l1_loss(V, G)
    entropy = torch.stack(entropies).mean()                  # ← 不是只用最后一步
    loss = pg_loss + 0.5 * v_loss - 0.01 * entropy   # entropy bonus

    opt.zero_grad(); loss.backward(); opt.step()

    if ep % 50 == 0:
        print(f"ep {ep:4d}  return {sum(rewards):6.1f}")
```

**预期**：CartPole-v1 在 ~500 episode 收敛到 ≥ 450 reward。注意：
1. `advantage.detach()` —— baseline 仅用于降方差，**不传梯度** 到 critic（critic 用独立的 v_loss 更新）
2. **entropy bonus** 系数 0.01——和 A3C / PPO 一致；防止策略过早 collapse
3. 这是 REINFORCE + baseline + entropy bonus，距离完整 A2C 差的是 n-step return（这里用全 episode return）

## §B 一手资料

| 算法 / 概念 | 论文 / 资料 | 一作 | 年 | 出处 | arXiv / DOI |
|---|---|---|---|---|---|
| **教科书** | *Reinforcement Learning: An Introduction* (2nd ed.) | Sutton & Barto | 2018 | MIT Press | ISBN 978-0262039246 |
| Q-learning | *Learning from Delayed Rewards* (PhD 论文) | C. Watkins | 1989 | King's College, Cambridge | — |
| Q-learning 收敛 | *Q-Learning* | Watkins & Dayan | 1992 | *Machine Learning* 8:279-292 | DOI 10.1007/BF00992698 |
| SARSA | *On-line Q-Learning Using Connectionist Systems* (CUED/F-INFENG/TR 166) | Rummery & Niranjan | 1994 | Cambridge tech report | — |
| REINFORCE | *Simple Statistical Gradient-Following Algorithms* | R. J. Williams | 1992 | *Machine Learning* 8:229-256 | DOI 10.1007/BF00992696 |
| PG 定理 | *Policy Gradient Methods for RL with Function Approximation* | R. Sutton | 1999/2000 | NeurIPS | — |
| **DQN (workshop)** | *Playing Atari with Deep Reinforcement Learning* | V. Mnih | 2013 | NeurIPS DLW | arXiv:1312.5602 |
| **DQN (Nature)** | *Human-level control through deep reinforcement learning* | V. Mnih | 2015 | *Nature* 518:529-533 | DOI 10.1038/nature14236 |
| **Double DQN** | *Deep RL with Double Q-Learning* | H. van Hasselt | 2016 | AAAI | arXiv:1509.06461 |
| **Dueling DQN** | *Dueling Network Architectures for Deep RL* | Z. Wang | 2016 | ICML（best paper） | arXiv:1511.06581 |
| **PER** | *Prioritized Experience Replay* | T. Schaul | 2016 | ICLR | arXiv:1511.05952 |
| **Rainbow** | *Rainbow: Combining Improvements in Deep RL* | M. Hessel | 2018 | AAAI | arXiv:1710.02298 |
| A3C | *Asynchronous Methods for Deep RL* | V. Mnih | 2016 | ICML | arXiv:1602.01783 |
| **GAE** | *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | J. Schulman | 2016 | ICLR | arXiv:1506.02438 |
| NoisyNet | *Noisy Networks for Exploration* | M. Fortunato | 2018 | ICLR | arXiv:1706.10295 |
| Retrace(λ) | *Safe and Efficient Off-Policy RL* | R. Munos | 2016 | NeurIPS | arXiv:1606.02647 |
| V-trace / IMPALA | *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures* | L. Espeholt | 2018 | ICML | arXiv:1802.01561 |
| PPO | *Proximal Policy Optimization Algorithms* | J. Schulman | 2017 | arXiv | arXiv:1707.06347 |
| Per-decision IS | *Eligibility Traces for Off-Policy Policy Evaluation* | Precup, Sutton, Singh | 2000 | ICML | — |
| Entropy bonus | *Function optimization using connectionist RL algorithms* | Williams & Peng | 1991 | *Connection Science* 3:241-268 | DOI 10.1080/09540099108946587 |

**延伸阅读**：

- 教科书章节速查：Sutton & Barto Ch 3-5（MDP / DP / MC）→ Ch 6-7（TD / n-step）→ Ch 11（off-policy + 函数近似 + deadly triad）→ Ch 13（policy gradient）
- 工程对照：OpenAI Spinning Up（[spinningup.openai.com](https://spinningup.openai.com)）有所有算法的 from-scratch 实现 + 数学注释
- 经验回放 visualization：[Lil'Log – A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
