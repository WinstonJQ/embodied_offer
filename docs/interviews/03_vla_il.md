# 卷三 · VLA / 模仿学习 · 高频面试题

> 中文具身智能秋招高频面试题库 · **第三卷**
> 题源：牛客 / 知乎 / 小红书（教授级 AMA）/ CSDN / GitHub 公开面经
> 同义题合并后 **58 题**（频次 ≥3 主表 + N1/N2/N4 三题因来源是顶级 lab 教授集中 AMA 破例入选）

**难度分布**：L1（必会） **18** · L2（进阶） **31** · L3（顶级 lab） **9**

**使用方式**：题目默认折叠，点开看答案。建议先按 **L1 → L2 → L3** 顺序刷；同级内按频次从高到低。手机端原生支持。

---

## §1 模仿学习理论基础（8 题）

> IL 是 VLA 的入口；先把 BC / DAgger / 协变量偏移搞清楚，后面看 ACT / Diffusion Policy 才能融会贯通。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q01</b> · Behavior Cloning（BC）是什么？为什么会有 covariate shift？怎么缓解？</summary>

**答**：BC 把专家轨迹 $(s, a^*)$ 当监督学习样本，学 $\pi_\theta(a \mid s)$ 最小化 $\|a - a^*\|^2$（连续动作）或交叉熵（离散）。

**Covariate shift**（协变量偏移）：策略一旦小偏离专家分布，落入未见状态 → 误差累积 → 长程任务复利崩塌，T 步误差按 $O(\epsilon T^2)$ 增长。

**缓解**：① **DAgger** — 迭代收集策略访问态 + 专家标注，几乎消除 shift；② 数据扩增（轨迹噪声/镜像/回放）；③ **Action Chunking** — 一次预测 K 步，把"复利"压成 $T/K$ 次；④ Diffusion Policy 多模态分布建模；⑤ 加 RL 微调（IL + RL）。

**易错**：BC 不是"数据多就行"——数据多但仍偏专家分布，shift 不会自动消失；需要让策略实际访问的状态分布也在数据里。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×11</span> <b>Q02</b> · BC、DAgger、GAIL、IRL 的区别与适用场景？</summary>

**答**：
- **BC**：监督学习直接学 $\pi(a\mid s)$；简单、需要大量演示，有 shift。**场景**：数据足、任务短。
- **DAgger**：BC + 主动学习，迭代让策略访问的新状态由专家标注；几乎消除 shift。**场景**：专家在线可查询（仿真 / 人在回路）。
- **GAIL**：生成对抗 IL，判别器区分"策略 vs 专家"轨迹，policy 通过 RL 最大化欺骗判别器；无需 reward。**场景**：复杂行为、reward 难写。
- **IRL**：反推 reward function $R(s,a)$，再 RL 求解；可解释、可泛化。**场景**：想理解专家"为什么这么做"。

**易错**：DAgger 不是 BC 的"训练 trick"，是要持续访问专家；GAIL 训练不稳（GAN 通病），机器人上较少用。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q03</b> · DDPG / PPO / TD3 / SAC 的区别？哪些更适合机器人连续控制？</summary>

**答**：
- **DDPG**：off-policy actor-critic，确定性策略 + Q-learning；连续动作开山之作，但 Q 易过估、超参敏感。
- **TD3**：DDPG + **双 Q**（取 min 缓过估）+ **delayed actor update** + **target policy smoothing**；稳但保守。
- **SAC**：max-entropy 框架，policy 输出 $\pi(\cdot\mid s) = \mathcal{N}(\mu_\theta, \sigma_\theta)$ + **自动 temperature**；探索好、sample efficient。
- **PPO**：on-policy，clip ratio 限制策略更新；样本贵但稳，常用于 RLHF / VLA 微调。

**机器人推荐**：① 真机 sample 贵 → **SAC**（off-policy + 探索强）；② 仿真大规模 + 稳定性优先 → **PPO**；③ VLA + LLM 风格微调 → **PPO**（与 RLHF 一致）。

**易错**：SAC 的 entropy 不是"探索 trick"，是 max-entropy RL 的正则项；temperature α 自动调比手调好得多。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q04</b> · Mobile ALOHA 怎么解决"移动 + 双臂"协同？它和静态 ALOHA 的数据怎么联合训练？</summary>

**答**：**Mobile ALOHA** = 静态 ALOHA（2×7 DoF 双臂）+ **2 维移动底盘**（线速度 v、角速度 ω）= **16 维动作 / 步**。

**协同关键**：① 同步遥操（teleop master arms 控双臂 + 操作员脚踩 base）；② **Co-training**：把静态 ALOHA 1000+ demos（精细操作）与 Mobile 50 demos（移动操作）**混合训练**，让模型继承静态数据的精细操作能力；③ ACT 架构无需改动，把 base 动作当额外 2 维直接喂进去。

**结果**：Mobile-only 训练成功率 ~30%；静态 + Mobile co-train ~90%（50 demos 即可完成"擦桌子 / 关门 / 烹饪"等长程任务）。

**易错**：Mobile ALOHA 是 **16 维**，不是 14（容易漏算 base 的 v/ω）；50 demos 不少，但相比传统 IL 已经极少。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q05</b> · 解释 Markov 性质，为什么 RL 假设 MDP？POMDP 是怎么扩展的？</summary>

**答**：**Markov 性质**：$P(s_{t+1} \mid s_t, a_t, s_{<t}, a_{<t}) = P(s_{t+1} \mid s_t, a_t)$，即下一状态只依赖当前 $(s, a)$，与历史无关。

**MDP**：$\langle S, A, P, R, \gamma \rangle$，假设 **fully observable**（agent 看到的就是真实 state）；Bellman 方程可解，DP/RL 算法都依赖它。

**POMDP**：加观测函数 $O$，agent 收到 $o_t = O(s_t)$ 而非真实 $s_t$；用 **belief state** $b_t = P(s_t \mid o_{\leq t}, a_{<t})$ 替代 $s_t$，或用 RNN/Transformer 编码历史。

**机器人现实**：几乎都是 POMDP（相机有限视野、力觉噪声、遮挡）；这是为什么 VLA 多用 history 帧 + 序列模型，而非单帧。

**易错**：POMDP 不等于"不能 RL"，只是要换 belief 或 history-conditioning。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q06</b> · 值迭代和策略迭代的区别？最优值函数为什么直接对应最优策略？</summary>

**答**：
- **值迭代（VI）**：反复迭代 $V_{k+1}(s) = \max_a [R(s,a) + \gamma \mathbb{E}_{s'} V_k(s')]$ 直到收敛 $V^*$，最后 $\pi^*(s) = \arg\max_a Q^*(s, a)$。
- **策略迭代（PI）**：两步交替——① **评估**：解线性方程组求 $V^\pi$；② **改进**：$\pi'(s) = \arg\max_a Q^\pi(s, a)$。

**区别**：VI 每步只做"局部 max + 一步 lookahead"，便宜但需更多迭代；PI 每步评估完整 $V^\pi$，贵但收敛步数少。两者都收敛到 $V^*$。

**最优值 → 最优策略**：定义 $\pi^*(s) = \arg\max_a [R(s,a) + \gamma \,\mathbb{E}_{s' \sim P(\cdot \mid s,a)}\, V^*(s')]$，由 Bellman 最优方程，这就是 greedy w.r.t. $V^*$；所以 $V^*$ 解出来策略也"白送"。

**易错**：PI 的"评估"步在状态多时需要近似（GPI 即 partial PI）；VI 在大状态空间退化成 Q-learning。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q07</b> · model-based RL 和 model-free RL 的区别？哪种更适合 sample-expensive 的真机训练？</summary>

**答**：
- **Model-free**：直接学策略 $\pi$ 或值函数 $Q$，不显式建模环境（PPO/SAC/DQN）；简单、稳，但 sample efficiency 差。
- **Model-based**：先学环境动力学 $\hat{P}(s'\mid s,a)$ 与 $\hat{R}$，再用模型做规划（CEM/MPPI）或 imagination 训 policy（Dreamer/MuZero）；sample efficient 5-100×。

**真机首选 model-based**：真机数据贵（1 次抓取 30s + reset），model-based 用 1/10 数据达到 model-free 效果。

**但有坑**：① **Model error 累积**：长程 rollout 误差爆炸（→ short horizon / 不确定性正则）；② 真机分布与模型分布不一致（→ DAgger 思路收集 on-policy 数据回训模型）。

**实务路线**：仿真 model-free 预训练 → 真机 model-based 精调；或 offline RL 先吃所有历史数据，再 model-based 精调。

**易错**：Dreamer 不是 model-free，是 model-based + latent dynamics + imagination。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q08</b> · GAE（Generalized Advantage Estimation）是什么？为什么 PPO 离不开它？</summary>

**答**：**GAE** 是 advantage 估计器，在 high-variance MC 和 high-bias TD 之间用 $\lambda$ 插值：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\lambda=1$：纯 MC，方差大、无 bias；
- $\lambda=0$：纯 TD，方差小、有 bias；
- 实务 $\lambda \approx 0.95$, $\gamma \approx 0.99$ 是甜区。

**PPO 离不开**：PPO 的目标是 $\mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t)]$，**没有 $\hat{A}$ 就没有 PPO**。GAE 给 $\hat{A}$ 提供低方差估计，是 PPO 在机器人/RLHF 上稳定收敛的关键。

**易错**：GAE 不是"PPO 专属"，TRPO / A3C 都能用；但 PPO clip + GAE 是最常见组合。

</details>

---

## §2 VLA 模型架构时间线（9 题）

> RT-1 → RT-2 → OpenVLA → π0 → π0.5/0.6/0.7 的演进 + RDT / Octo / GR00T 平行路线。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×15</span> <b>Q09</b> · OpenVLA 和 RT-2 在架构上的主要区别？OpenVLA 用 7B Llama-2 凭什么打过 55B 的 RT-2-X？</summary>

**答**：
| | OpenVLA | RT-2 / RT-2-X |
|---|---|---|
| Backbone | Llama-2 **7B**（Prismatic 框架） | PaLI-X **55B** 或 PaLM-E **12B**（闭源） |
| 视觉 | **SigLIP + DINOv2 双路** | PaLI-X / PaLM-E 内置 ViT（含 ViT-22B 等） |
| 动作头 | 每维 256 bin 离散化 → 复用 LLM 词表最不常用 256 token | 同 |
| 训练数据 | Open-X-Embodiment **970K** 轨迹 | Google 内部 + Open-X |
| 开源 | ✅ 全开源 + LoRA-friendly | ❌ 闭源 |

**胜出原因**：① **DINOv2 补足空间几何**——CLIP/SigLIP 偏语义弱几何，DINOv2 自监督预训练对空间结构更敏感；② Llama-2 web 文本 + 代码预训练更稳；③ 全开源使社区微调迭代更快。

**易错**：OpenVLA 不是"小胜大"玄学——是同等 fine-tune 量上的工程整合优势；零样本新任务上 RT-2-X 仍可能更强。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q10</b> · RT-2 是怎么把动作编码成 token 的？为什么这种"动作即文本 token"方案有效但效率不高？</summary>

**答**：**方案**：动作向量 **7 维 = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]**，每维正态化到 $[-1, 1]$，再离散化为 **256 个 bin**；取 LLM 词表里**最不常用的 256 个 token** 复用为"动作 token"。一步动作 = **7 个 token**，与文本 token 同混训。

**有效**：① 复用 LLM 的 token embedding + 自回归解码，直接利用 web 知识；② co-finetune 文本 + 机器人数据，VLM 的常识能迁移；③ scale up 自然。

**低效**：① **7 token 串行 AR 解码**，单步推理 = 7 forward；chunk K 步则 $7K$ forward；OpenVLA 6 Hz 慢的根源。② 动作复用 LLM token 不是"自然"的——精度不足、跨任务泛化弱。

**改进**：① OpenVLA-OFT 用**并行解码** + 连续动作头；② π0-FAST 用 DCT + BPE 把 chunk 压成更少 token。

**易错**：256 bin 不是"够细"，是 VLM 复用的权衡；高精度任务（< 0.1 mm 插针）必失效。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×8</span> <b>Q11</b> · π0 / π0.5 / π0.6 三个版本的核心差异？KI（Knowledge Insulation）解决了什么？</summary>

**答**：
- **π0**（2024-10）：PaliGemma 3B（VLM）+ **Action Expert 300M**（独立 Flow Matching 动作头）+ 50 Hz 控制 + chunk **H = 50**。
- **π0.5**（2025）：核心是**开放世界泛化 + 多源 co-training**，让 π0 框架在更多任务/场景/embodiment 上稳定迁移；**KI（Knowledge Insulation）是 π0.5 框架的后续扩展**，隔离 VLM 与 Action Expert 梯度防止动作训练污染语言/视觉能力。
- **π0.6 / π*0.6**（2025-11）：核心创新 **RECAP**——把示教数据当 offline RL 用，advantage-conditioned 采样，强化"好动作 token"权重。
- **π0.7**（2026-04）：最新发布。

**KI 解决什么**：早期版本动作微调梯度反传会**破坏 VLM 的语言/视觉理解**（灾难性遗忘）；KI 用 stop-gradient + 双路梯度让两个模块各学各的。

**易错**：RECAP 属于 **π0.6**，不是 π0.5；π0.5 主体是"开放世界泛化"，KI 是配套扩展工作；π0.7 是 **2026-04** 发布。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q12</b> · RDT-1B、Octo、GR00T-N1 都是 VLA，它们的动作头分别长什么样？哪种结构更"重"？</summary>

**答**：
| 模型 | 总参 | 动作头 | "重量" |
|---|---|---|---|
| **RDT-1B**（2024-10） | 1.2B | 整个 1.2B Transformer 当 **diffusion denoiser** | **最重** |
| **Octo**（2024-05） | ~93M | 小 Transformer encoder + **小 diffusion head（~10M）** | **最轻** |
| **GR00T-N1**（2025） | 2.2B | VLM 1.34B + **Action Expert 0.86B**（Flow Matching） | 中重 |

**trade-off**：
- Head 太轻（Octo）→ 多模态动作分布建模能力受限，复杂任务难。
- Head 太重（RDT）→ 表达力强但推理慢、训练贵。
- GR00T-N1 中庸路线，借鉴 π0 的 backbone + expert 拆分思路。

**易错**：RDT-1B 不是"另一个 OpenVLA"——它没用 LLM token 离散化，直接 backbone 当 denoiser；这是与 OpenVLA / RT-2 范式不同的"端到端 diffusion VLA"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q13</b> · FAST tokenizer 是怎么做的？为什么 DCT + BPE 组合比传统离散化高效 10×？</summary>

**答**：**FAST = DCT → scale → round → flatten → BPE**，五步缺一不可。

1. **DCT**（Discrete Cosine Transform）：把 H 步 × D 维动作 chunk 转频域，主要能量集中低频。
2. **Scale**：把 DCT 系数缩放到合适整数范围（e.g. ±100）。
3. **Round**：离散化为整数。
4. **Flatten**：二维 (H × D) → 一维序列。
5. **BPE**：用预训练 BPE tokenizer 把常见短模式压成单 token。

**高效 10× 原因**：① DCT 集中能量，**高频系数大多 ≈ 0**，round 后变长串 0；② BPE 把这些重复模式压成 1-2 token；③ chunk 越长压缩比越高（H=50 时尤甚）。

**效果**：π0-FAST 在同样 chunk 下，token 数从 ~400 降到 ~40，推理速度 5-10× 提升。

**易错**：FAST 不是 "DCT 单独用"，BPE 是关键；少一步压缩比就不够。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q14</b> · OpenVLA 的动作是怎么离散化的（每维 256 个 bin）？这种方案在多大动作精度任务上会失效？</summary>

**答**：**离散化方案**：取每维动作（7 维（6-DoF 末端位姿 + 1 维 gripper））在训练集的 1%-99% 分位数区间，等分成 **256 个 bin**；超界截断。每个 bin 复用 LLM 词表里最不常用的 256 个 token。

**失效场景**：
1. **高精度任务**（< 0.1 mm 插针、拧螺丝）：256 bin 在 ±10 cm 量程下分辨率 ≈ 0.8 mm，精度不够。
2. **极端动作**：超 99% 分位被截断，剧烈动作不能表达。
3. **多模态动作**：256 bin 是单峰离散分布，难以表达"两条专家路径"这种多模态结构。

**改进**：① **OpenVLA-OFT** 换成**连续动作头 + L1 回归**；② π0 用 **Flow Matching** 连续动作建模；③ FAST tokenizer 提高压缩比同时保精度。

**易错**：256 bin 不是 OpenVLA 独创，RT-2 也是；这是 VLM 复用方案的固有局限。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q15</b> · RT-1 → RT-2 → RT-X 的演进逻辑是什么？每一代的关键创新是什么？</summary>

**答**：
- **RT-1**（2022）：EfficientNet + FiLM + Token Learner；**130K 真机数据**；首个 large-scale robot Transformer paradigm，但无 web 知识。
- **RT-2**（2023）：PaLI-X 55B / PaLM-E 12B + 动作 token + **co-finetune（web + robot data）**；**首次把 web 知识灌入 robot policy**。
- **RT-X**（2023-10）：RT-1 / RT-2 在 **Open-X-Embodiment**（~60 datasets / 22 embodiments / 1M+ trajectories）上跨形态训练（实验用 robotics mixture 子集），验证跨 embodiment 泛化。OpenVLA 后续用 OXE 中约 **970K** curated demos 微调。

**主线**：RT-1 证 paradigm → RT-2 引 web 先验 → RT-X 验证联合训练。OpenVLA 是 RT-X 路线的开源版。

**易错**：RT-2 核心是 **co-finetune** 引入 web 数据，不是"放大版 RT-1"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q16</b> · π0 用的是哪个 VLM backbone？为什么选 PaliGemma 而不是 Llama？</summary>

**答**：**π0 backbone = PaliGemma**（SigLIP-400M 视觉编码器 + **Gemma-2B** 语言模型）。

**选择原因**：
1. **已对齐的 VLM**：PaliGemma 是 Google 联合预训练好的 vision-language 模型，比 "Llama + 外挂 ViT" 拼接更稳。
2. **Gemma-2B 够小**：50 Hz 高频推理（20 ms / step）要求 backbone 轻量；Llama-2 7B 在消费级 GPU 上跑不到 50 Hz。
3. **Apache 2.0 协议**：商用友好，机器人公司容易落地。
4. **Multimodal projector 现成**：图像 → 文本 token 的投影层已训练好。

**不用 Llama**：① text-only，需要再接视觉；② 7B 太重；③ Meta 协议有限制。

**易错**：**PaliGemma ≠ Gemma**——PaliGemma 含 SigLIP 视觉；说成 "π0 用 Llama" 是常见错（OpenVLA 才用 Llama-2 7B）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q17</b> · π0 里 Action Expert 的 300M 参数为什么单独拆出来？和 backbone 怎么交互？</summary>

**答**：**π0 结构**：PaliGemma 3B（VLM，frozen 或 LoRA） + **Action Expert 300M**（独立 Flow Matching 网络）。

**拆出原因**：
1. **负载分层**：每次重算 chunk，VLM forward 一次产条件特征；Action Expert 走约 10 步 flow ODE。重算频率由 execution horizon / async 决定（不是固定 1 Hz）。
2. **训练目标不同**：VLM 是 AR 语义损失，Action Expert 是 Flow Matching 速度场回归；放一起训会互相干扰。
3. **隔离梯度**（KI 思想）：动作微调不应破坏 VLM 的语言/视觉理解。

**交互**：VLM 编码 → 条件特征 $c$ → Action Expert 接 $(c, \text{noisy chunk}, \tau)$ → 约 10 步 ODE 出干净 chunk。chunk H=50 在 50 Hz 下覆盖 ~1 秒。

**易错**：Action Expert 是 joint training（不是独立）；H=50 = **覆盖时长**，不等于"每秒重算一次"。

</details>

---

## §3 Diffusion Policy / Flow Matching（6 题）

> 动作分布建模的两条主流路线；π0 系列让 Flow Matching 从理论走到工业落地。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×11</span> <b>Q18</b> · Diffusion Policy 怎么把动作建模成"去噪过程"？相比 BC 直接回归动作有什么优势？</summary>

**答**：**建模**：把动作分布 $\pi(a \mid s)$ 用 score-based diffusion 建模。
- **训练**：在专家动作 $a^*$ 上加噪 $a_\tau = \sqrt{\bar\alpha_\tau} a^* + \sqrt{1-\bar\alpha_\tau}\,\varepsilon$，学噪声预测器 $\varepsilon_\theta(a_\tau, \tau, s)$。
- **推理**：从 $\varepsilon \sim \mathcal{N}(0, I)$ 起，反向 SDE/ODE 逐步 denoise 到 $a$（DDPM 50 步 / DDIM 10-20 步）。

**vs BC 优势**：
1. **多模态分布**：BC 回归会"平均"多专家路径输出"中间"不可执行动作；diffusion 显式建模多峰。
2. **chunk 一次生成**：天然支持 K 步动作 chunk（去噪整段而非单步）。
3. **长尾任务更稳**：噪声扰动训练相当于隐式数据增强。

**缺点**：① **推理慢**（10-50 step），高频控制需并行/蒸馏；② 训练目标对噪声调度敏感。

**易错**：Diffusion Policy 不是"在 BC loss 上加噪声"，是完全不同的 score-based 框架。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q19</b> · π0 为什么用 Flow Matching 而不是 Diffusion？50 Hz 控制频率背后的工程考量是什么？</summary>

**答**：**Flow Matching 优点**：
1. **训练更稳**：学速度场 $v_\theta$ 回归 $v^* = x_1 - x_0$，数值范围紧凑；diffusion 学 $\varepsilon$ 范围大、训练易抖。
2. **推理步数少**：10 步 ODE 即可 vs DDPM 50 步；省去 cosine/linear schedule 超参。
3. **路径显式**：$x_\tau = \tau x_1 + (1-\tau) x_0$，方差 $(1-\tau)^2 I$（不是 $(1-\tau) I$，常见笔误）。

**50 Hz 工程考量**：控制频率 50 Hz（20 ms / step）由底层执行端按 chunk 消耗动作。H=50 = **覆盖时长** ≈1 秒；**重规划频率**由 execution horizon / temporal ensemble / 异步队列决定，不是固定 1 Hz。

**易错**：50 Hz = 控制频率，H=50 = 覆盖时长，两者不要混成"每秒重算一次 chunk"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q20</b> · Diffusion 和 Flow Matching 有什么本质区别？为什么 Flow Matching 在机器人上越来越流行？</summary>

**答**：**本质区别**：
| | Diffusion | Flow Matching |
|---|---|---|
| 数学框架 | **SDE** 前向加噪 + 反向去噪 | **ODE** 沿运输路径走 |
| 学习目标 | 噪声 $\varepsilon$ 或 score $\nabla \log p_\tau$ | **速度场** $v_\theta$ |
| 路径定义 | 由 SDE 调度（cosine/linear）决定 | **显式给出** $x_\tau = \tau x_1 + (1-\tau) x_0$ |
| 推理 | SDE 求解器（慢） | ODE 求解器（快） |

**机器人流行原因**：
1. **推理快**：10 ODE step vs 50 SDE step，高频控制可行。
2. **训练稳**：速度场数值范围紧凑，loss 不爆炸。
3. **超参少**：无需选 cosine/linear 调度。
4. **理论统一**：Diffusion 是 FM 的特例（特定路径下）。

**易错**：FM 不是 Diffusion 的"加速版"，是更通用框架；Diffusion 可看作 FM 在特定路径上的实例。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q21</b> · Diffusion Policy 推理时用 DDPM 还是 DDIM？步数怎么选才能既快又准？</summary>

**答**：**训练同一 $\varepsilon_\theta$**，部署时按需选采样器。

- **DDPM**：reverse SDE，**stochastic**，50+ step；高质量但慢；适合离线生成/训练监督。
- **DDIM**：reverse ODE，**deterministic**，10-20 step 可达 DDPM 50 步质量；适合实时机器人。

**步数选择**：
- 实时 50 Hz 控制 → DDIM 10 步（≈ 20 ms / chunk）。
- 离线评估 / 高质量生成 → DDPM 50 步。
- 一般任务 → DDIM 20 步是甜区。

**DDIM 早期 OOD 坑**：DDIM 在去噪**早期步**（$\tau$ 接近 1，即接近纯噪声）对 OOD 状态敏感，可能输出非平滑动作；Reactive DP / 调度优化可缓解。

**易错**：DDIM 不是"绝对不如 DDPM"——它在 chunk 中段/后段表现一致，只是早期 OOD 步需要小心。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q22</b> · ACT 和 Diffusion Policy 各自适合什么任务？硬件 / 任务复杂度上谁更稳？</summary>

**答**：
| | ACT | Diffusion Policy |
|---|---|---|
| 结构 | CVAE + Transformer | Diffusion 模型 |
| 推理 | 单 forward（快） | 10-50 step（慢） |
| 多模态 | 通过 latent z 建模 | 天然多模态 |
| 适合任务 | **精细 manipulation**（插针、烹饪、ALOHA） | **多解 / 长程任务**（推杆、堆叠、避障） |
| 硬件 | 单卡可跑实时 | 需并行解码或蒸馏 |

**口诀**：**"ACT 出拳快，DP 内功深"**。

**选择**：
1. 任务有多专家路径（多解）→ DP。
2. 任务单解但精度高 → ACT。
3. 实时 50 Hz 严苛 → ACT。
4. 多任务通用 backbone → DP（配合并行加速）。

**易错**：DP 不是"通用更好"；在 ALOHA 一类精细任务上 ACT 仍是 SOTA 之一。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q23</b> · Reactive Diffusion Policy / Slow-Fast Diffusion Policy 是怎么解决 DP 推理慢的？</summary>

**答**：**问题**：DP 推理 10-50 step，高频接触控制不可行。

**Reactive Diffusion Policy（RDP）**：双层架构——**slow planner**（重型 visual-tactile diffusion，低频生成 chunk）+ **fast reactive policy / controller**（轻量、高频，根据实时触觉/力觉做微调）。针对接触丰富任务，关键是低延迟反馈。

**Slow-Fast Diffusion Policy**：思路类似——slow 大模型离线/低频生成动作 chunk 候选，fast 小 policy 在线选择 + 调整；与双系统 VLA 的"S1 / S2"同构。

**共同思路**：把"慢的多模态生成"和"快的反应执行"解耦，重 DP 推理移到低频通路，高频通路用轻量 head；接触任务上比纯 DP 显著更稳。

**易错**：RDP 不只是"并行去噪 / streaming"——核心是 slow planner + fast reactive controller 的双层；模型权重也分两套。

</details>

---

## §4 Action Chunking & ACT（4 题）

> 一次预测多步动作是 IL 长程任务的关键；ACT 是 ALOHA 团队的范式，π0 / OpenVLA 都借鉴。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q24</b> · 什么是 Action Chunking？为什么 VLA 要预测一个动作 chunk 而不是单步动作？块大小怎么选？</summary>

**答**：**Action Chunking**：策略一次预测 K 步连续动作 $a_{t:t+K}$ 而非单步 $a_t$。

**执行方式**：
- **Open-loop**：执行整个 chunk 再重新规划。
- **Receding-horizon**：执行 1 步重规划，类似 MPC。
- **Ensemble**：多 chunk 重叠用 EMA / 投票（ACT 用法）。

**好处**：
1. **减少高频推理开销**：K 倍。
2. **平滑动作**：相邻步内插值好，减少抖动。
3. **显式建模时序相关性**：尤其 diffusion / flow 类策略受益。
4. **缓解复合误差**：T 步任务的"高层决策点"从 T 降到 $T/K$，整体偏离专家分布的次数变少；但 chunk 内仍是 open-loop，不能严格消除 covariate shift。

**块大小**：
- ALOHA / 精细操作：K = 8-16（160-320 ms）。
- π0：K = 50（50 Hz 下 1 秒，长程任务）。
- 任务变化慢可更长；环境多变需更短。

**易错**：chunk 太长牺牲反应性（环境突变前已锁死动作）；4-16 在 manipulation 上是甜区。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q25</b> · ACT（Action Chunking with Transformers）的结构是什么？它用 VAE / CVAE 解决了什么问题？</summary>

**答**：**ACT 结构**（ALOHA 团队 2023）：Encoder-Decoder Transformer + **CVAE**。

- **Encoder**：图像 + proprio + 文本指令 → 上下文特征。
- **Decoder**：自回归 / 一次性输出 K 步动作 chunk。
- **CVAE**：训练时额外 encoder 把 (动作 chunk, 观测) → latent $z$，KL 正则到 $\mathcal{N}(0, I)$；decoder 用 $(z, \text{观测})$ 还原 chunk。
- **推理时不用 encoder**：直接 $z = 0$（mean）→ decoder 出 chunk。

**CVAE 解决的问题**：**多模态动作分布**。直接 MSE 回归 chunk 会把"专家 A 走左路 / 专家 B 走右路"平均成"中间路"（物理不可执行）。CVAE 让 $z$ 编码"走哪条路的意图"，推理 $z=0$ 输出**某条代表性路径**而非 mean。

**易错**：ACT 推理用 $z=0$ 不是"取所有路径平均"，而是"取 prior 的 mode"，仍是某条具体路径。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q26</b> · 模仿学习里"复合误差"指什么？Action Chunking 是怎么缓解它的？理论解释是什么？</summary>

**答**：**复合误差**（compounding error）：BC 在每步犯小错 $\epsilon$，T 步后误差累积为 $O(\epsilon T^2)$（在最坏情况下二次增长）；策略一旦偏离专家分布，错误像复利一样雪崩。

**Action Chunking 缓解**：一次决策 K 步，等价于把 T 步轨迹划成 $T/K$ 个"决策点"，**chunk 内动作 joint 预测，不会因单步偏离累积**；只在 chunk 之间的决策点处可能累积。

**直觉**：高层重规划次数从 $T$ 降到 $T/K$，复合误差量级随之缩小；但 chunk 内部仍是 open-loop，环境扰动或专家不一致仍会让单 chunk 内误差累积，因此并非"严格平方降阶"。

**实证**：ACT 论文中较大 chunk 在插针等高精度任务上显著优于单步；最佳 K 与任务变化率相关，过长牺牲反应性。

**易错**：chunking 不解决"OOD 状态"本身——只减少"高层决策点数量"；仍需 DAgger / 数据多样性来覆盖 OOD。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q27</b> · ACT 推理时把 $z$ 固定为 0 有什么局限？为什么不随机采样？KL 权重 β 怎么调？</summary>

**答**：**$z = 0$ 的局限**：取 prior 的 mode → 在多模态任务上退化为"代表性单解"路径，丢失探索性；如果训练分布里多策略权重接近，$z = 0$ 可能落在"平均路"附近，物理仍不可执行。

**为什么不随机采样 $z \sim \mathcal{N}(0, I)$**：连续步骤独立采样 → 策略不连续（前 1 秒走左、后 1 秒走右）。解法是 **chunk 边界冻结 $z$**（同一 episode 内固定）或 **temporal ensemble** 平均；纯随机采样实务上比 $z = 0$ 还差。

**KL 权重 β 调参（β-VAE 视角）**：
- β 过大 → **posterior collapse**，$z$ 失活（编码器输出近 $\mathcal{N}(0, I)$，无信息），decoder 退化为 MSE。
- β 过小 → $z$ 太"任性"，推理 $z = 0$ 不再"代表性"。
- ACT 实务：β 取 $\sim$10 或更小；动作 reconstruction loss 权重 1.0；常监控 KL 项是否近 0（collapse 信号）。

**易错**：KL collapse 是 ACT 训练最常见陷阱——KL → 0 不等于"训练好了"，是 latent 失活；要看 reconstruction 是否真的依赖 $z$。

</details>

---

## §5 视觉-语言对齐 & 数据（7 题）

> Open-X 是 VLA 走通的数据基石；视觉编码器选择决定空间几何能力上限。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q28</b> · 介绍一下 Open-X-Embodiment 数据集，它解决了什么问题？数据多样性对 VLA 泛化的影响？</summary>

**答**：**Open-X-Embodiment（OXE）**：2023-10 Google 联合 21 个 lab 发布，约 **60 个数据集 / 22 种 embodiment / 1M+ 轨迹**（OpenVLA 训练用其中 curated **~970K** 子集）。是迄今最大的跨实验室、跨形态机器人数据集。

**解决的问题**：
1. **单 lab 数据量太小**（< 100K），scale 不起来。
2. **跨形态无数据**：每 lab 用不同机器人，无法对比/合并。
3. **任务覆盖窄**：单数据集多为 pick & place，缺多样性。

**对 VLA 泛化的影响**：
- RT-X 论文实证：在 OXE 上联合训练比单数据集训练，跨场景成功率显著提升。
- 数据多样性（任务 / 场景 / embodiment）比数据量本身更关键。
- OpenVLA / π0 都以 OXE 为预训练基础。

**易错**：OXE ≠ 一个统一格式的数据集，是 ~60 个原始数据集 + RLDS 统一接口；动作空间标准化到 7 维末端 + gripper（不足补 0）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q29</b> · OpenVLA 的视觉编码器为什么用 SigLIP + DINOv2 双路融合？只用 CLIP 行不行？</summary>

**答**：**SigLIP** 与 **DINOv2** 各有所长：
- **SigLIP**（image-text contrastive）：偏**语义**（"杯子" → 杯子图像），对自然语言强对齐。
- **DINOv2**（自监督）：偏**空间几何**（深度、表面纹理、空间结构），对场景结构敏感。

**双路融合**：两个编码器各出 patch tokens，**channel-wise concat**（不是 cross-attention），喂给 LLM。

**只用 SigLIP 不行**：
1. SigLIP / CLIP 对**空间几何弱**——能识别"杯子"但不知"杯子距夹爪 5 cm 偏左"。
2. 机器人操作高度依赖空间推理（抓取位姿、避障）。
3. **Prismatic VLM** 论文实证：双路 > SigLIP > DINOv2 > CLIP，在 robotics benchmark 上差距明显。

**易错**：双路融合不是 ensemble，是同时给 LLM 输入两套 tokens；LLM 自己学如何用。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q30</b> · VLM 和 VLA 的区别是什么？VLA 多了什么模块？token 化方案怎么变？</summary>

**答**：
| | VLM（Vision-Language Model） | VLA（Vision-Language-Action） |
|---|---|---|
| 输入 | 图像 + 文本 | 图像 + 文本指令 + 历史 proprio |
| 输出 | 文本 token | **动作**（连续 / 离散 token） |
| 训练数据 | web 图文 | web + **机器人轨迹** |
| 多出模块 | — | **Action head**（离散 token 或 Flow Matching 头） |

**Token 化方案**：
- VLM：仅 text token（如 32K vocab）。
- VLA：text token + **action token**——
  - 离散方案（RT-2 / OpenVLA）：每维动作 256 bin，复用 LLM 词表最不常用 256 token；一步 7 维 = 7 token。
  - 连续方案（π0 / RDT）：action 不 token 化，直接由独立 Action Expert（Flow Matching / Diffusion）输出。
  - FAST：DCT + BPE 把 chunk 压成更少 token，介于两者之间。

**易错**：VLA 不是 "VLM 加个 head" 就完事——训练时要 co-finetune 机器人数据，否则 VLM 在动作分布上零样本不行。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q31</b> · 多模态对齐里 Q-Former、projection、cross-attention 这几种 connector 的区别？</summary>

**答**：**Connector** = 把视觉特征对齐到 LLM 输入空间的模块。

| Connector | 代表 | 机制 | token 数 |
|---|---|---|---|
| **Projection** | LLaVA | MLP 直接映射 patch → token | = patch 数（多） |
| **Q-Former** | BLIP-2 | 可学 query 通过 cross-attn 提取关键视觉特征 | **可压缩**（少） |
| **Cross-Attention** | Flamingo | 在 LLM 每层插 cross-attn 与视觉特征交互 | 不增 input token，但**改 LLM 结构** |

**VLA 实务**：
- **Projection（OpenVLA / Prismatic）**：简单、稳；token 数多（256-576 / 图）但 LLM 能扛。
- **Q-Former**：复杂、训练不稳；VLA 上较少用。
- **Cross-Attention**：需改 LLM 架构；不便于直接用现成 VLM 权重。

**易错**：Q-Former 不是"更先进"——LLaVA 系列证明 projection 简单 + 数据多反而更好。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q32</b> · DROID 和 Bridge 数据集的特点分别是什么？它们和 Open-X 是什么关系？</summary>

**答**：
- **Bridge / BridgeData V2**（2023）：多场景多任务，约 **60K 轨迹**；早期跨任务 IL 基础数据集；任务集中在 pick & place + 简单 manipulation。
- **DROID**（2024）：约 **564 个场景 / 350 小时 / 76K demonstrations / ~86 tasks**（arXiv 摘要写 84，项目页 86，版本口径差异）；多样性 SOTA，被誉为"野外"机器人数据集。

**Open-X 关系**：**Bridge V2 是 OXE 子集**；**DROID 不在 OXE 原始发布的约 60 个数据集中**——OXE（2023-10）早于 DROID（2024）发布，但二者常被联合用于后续训练 / fine-tune。

**训练用法**：
- OpenVLA / π0：先在 OXE 上预训练（含 Bridge 等）。
- DROID 常作为后续大规模 fine-tune 源（场景泛化）。
- Bridge 常作为评估 baseline。

**易错**：DROID 不是"OXE 子集"——它是后续发布、独立维护的数据集；常与 OXE 联合用但出处不同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q33</b> · 怎么解决 VLA 训练里"多任务相互干扰 / 灾难性遗忘"？跨域联合训练怎么做？</summary>

**答**：**现象**：fine-tune VLA on task A 后，task B 性能掉；或 web 知识被动作微调"洗掉"。

**解法**：
1. **LoRA** — 冻结大部分参数，只训低秩适配；保留预训练知识。
2. **Co-training（联合训练）** — 把所有任务 / 多数据集混训，共享 backbone；OpenVLA 在 OXE ~60 datasets 上的做法。
3. **Rehearsal** — 持续学习时保留旧任务数据，定期 replay 防遗忘。
4. **KI（Knowledge Insulation，π0.5）** — 隔离 VLM 与 Action Expert 的梯度，VLM 不被动作训练污染。
5. **Task-conditional LoRA** — 每任务一组 LoRA 权重，推理时按 task ID 加载。

**跨域联合训练**：① 按比例采样（按 dataset 大小开根号防大数据集 dominate）；② Embodiment token 告诉模型当前数据集；③ Action 标准化到统一 7 维末端 + gripper（不足补 0）。

**易错**：LoRA 不能完全避免遗忘——LoRA 矩阵也会"污染"前向；真正抗遗忘还需配合 KI 或 frozen backbone。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q34</b> · 跨 embodiment 训练（手臂数不同 / 自由度不同）怎么对齐动作空间？Octo / RT-X 怎么做的？</summary>

**答**：**问题**：不同机器人 DoF 不同（**7-DoF Franka/Panda**、**6-DoF UR5e**、**14-DoF ALOHA 双臂**、**16-DoF Mobile ALOHA**）。

**RT-X / OpenVLA（单臂主流）**：canonicalize 到 **7 维 end-effector** [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]——不同关节配置通过 IK 映射到同一末端空间。

**双臂 / 移动**：保留原维度（14 / 16），用 **embodiment-specific action head** 或 **padding mask** 兼容；**不能截断到 7D**，否则丢第二臂或底盘动作。

**Octo**：可变长 action token + padding mask。

**关键技巧**：① embodiment-conditional token 告诉模型形态；② 动作按数据集 1%-99% 分位归一化；③ dataset 按大小开根号采样。

**易错**：单臂场景才能 7D canonicalization；双臂/移动需 embodiment-specific head + mask。

</details>

---

## §6 推理优化 & 控制频率（6 题）

> VLA 落地最大瓶颈是推理速度——50 Hz 控制频率把 6 Hz 的 OpenVLA 卡在实验室。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q35</b> · OpenVLA 推理速度只有约 6 Hz（4090 GPU 上），为什么这么慢？有哪些主流加速方案？</summary>

**答**：**慢的原因**：
1. **AR 串行解码**：每步动作 = 7 token 串行 forward（chunk K 步 = $7K$ forward）。
2. **Llama-2 7B 单 forward** 在 4090 上 30-50 ms。
3. **无 KV cache 复用**：多帧观测每次重算。

**主流加速方案**：
1. **OpenVLA-OFT**：并行解码 + 连续动作头 + L1 回归，**5-10× 提速**。
2. **TinyVLA / SpatialVLA**：蒸馏到 1-2B；牺牲精度换速度。
3. **量化**：INT8 / INT4，2-4× 提速，精度损失 < 5%。
4. **推理引擎**：vLLM / TensorRT / FlashAttention-2。
5. **KV cache 复用**：多帧观测共享 KV，减少重复计算。
6. **Chunking + 异步**：chunk 内 open-loop 执行，下一 chunk 重叠推理。

**易错**：OpenVLA 是 **6 Hz**（4090，未优化），不是 7 Hz——网传 7 Hz 是常见错传。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q36</b> · 如果模型推理只能跑 5 Hz 而机器人需要 50 Hz 控制，怎么办？说说工程方案。</summary>

**答**：**4 套方案**：

1. **Action Chunking + open-loop**：
   - 模型 5 Hz 输出 chunk K = 10（覆盖 0.2 秒）；
   - 机器人按 50 Hz 顺序执行 chunk 内动作 + 插值。
   - 简单稳，是 π0 / OpenVLA 默认做法。

2. **双系统 VLA**：
   - **System 2** 大 VLM 5 Hz 输出高层"语义子目标"；
   - **System 1** 小 policy 50 Hz 出 low-level action。
   - 代表：Figure Helix、GR00T N1。

3. **异步流水线**：
   - 当前 chunk 执行中，下一 chunk 已开始推理（overlap）；
   - 推理 latency 被掩盖在执行时间里。

4. **底层控制器兜底**：
   - 模型出 desired joint pos / velocity，底层 **PID / impedance** 控制器在 50 Hz / 1 kHz 跟踪；
   - 模型只管"目标"，控制器管"如何到达"。

**实务组合**：① + ④ 最常见——chunking + 底层控制器。

**易错**：不要试图"把模型做到 50 Hz"——50 Hz × 7B 模型 = 350 GFLOPS / s，4090 也吃力；架构上把高频部分剥离才对。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q37</b> · LoRA 微调一个 VLA 有哪些注意点？冻结哪部分？为什么 OpenVLA 推荐 LoRA？</summary>

**答**：**LoRA**：在 attention 的 $Q/K/V/O$ 矩阵上加低秩适配 $\Delta W = BA$（$r \ll d$）。

**冻结策略**（VLA 实务）：
- **全冻结**：视觉编码器（SigLIP / DINOv2）+ LLM backbone。
- **训练**：LoRA 矩阵 + **action head**（含 Action Expert 等）+ optional 视觉 projector LoRA。

**超参经验**：
- rank $r$ = 4-16；
- alpha = $2r$（缩放因子）；
- 学习率 1e-4 ~ 5e-4；
- 目标模块：attention 的 $Q, K, V, O$（不一定全加）。

**OpenVLA 推荐 LoRA 原因**：
1. **消费级硬件友好**：7B 全参微调需 8×A100；LoRA 单 4090（24 GB）可微调。
2. **不破坏 web 知识**：大部分参数 frozen，VLM 通识保留。
3. **社区迭代快**：每任务 ~MB 级 LoRA 权重，易共享。
4. **可组合**：多任务 LoRA 可加权组合。

**易错**：LoRA rank 不是越大越好；rank 太大 = 全参微调，丢了"防遗忘"的优势。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q38</b> · PaliGemma 为什么用 prefix-LM 注意力（图像 token 互看，动作 token 因果）？</summary>

**答**：**Prefix-LM attention**：
- **Prefix 部分**（图像 token + 系统 prompt）：**bidirectional**，互相可见。
- **Suffix 部分**（生成 / 动作 token）：**causal mask**，只能看前面。

**对比**：
- 纯 **causal LM**（GPT 风格）：所有 token 都因果；图像 token 只能看到自己之前的，视觉理解受限。
- 纯 **bidirectional**（BERT 风格）：不能 AR 生成。
- **Prefix-LM**：兼顾——图像 token 间信息充分流通，生成 token 仍是 AR。

**对 VLA 的价值**：
1. 图像 patch token 互相可见 → 空间推理更强。
2. 动作 token 因果 → 仍是 AR 解码，与 LLM 训练范式兼容。
3. π0 借鉴此设计：visual + text prefix（双向） + action token（因果）。

**易错**：Prefix-LM 不是"所有 token 都双向"——只有 prefix 部分双向；suffix 仍是 causal，否则没法 AR 生成。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q39</b> · OpenVLA-OFT 加速 VLA 的三大关键设计是什么（并行解码 / 连续动作 / L1 回归）？</summary>

**答**：**OFT recipe** = **parallel decoding + action chunking + continuous actions + L1 regression**（OpenVLA-OFT, arXiv:2502.19645）。

1. **Parallel Decoding**：抛弃 AR 串行（原 OpenVLA: 7 token/步），一次 forward 同时出 chunk 全部 $K \times 7$ 个动作分量（[ACTION] 占位 + bidirectional attention）。
2. **Action Chunking**：与 parallel 协同放大加速。
3. **Continuous Action Head**：放弃 256-bin，小 MLP 直接回归连续动作，无量化误差。
4. **L1 Loss**：比 L2 对 outlier / 噪声专家动作更鲁棒。

**论文报告**：action 生成吞吐 ≈ **26×**、单 step latency ≈ **3×** 改善；精度也提升。

**易错**：OFT 重训了 action head，不能直接套用 OpenVLA checkpoint。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q40</b> · OpenVLA 输出是离散 token，怎么在推理时还原成连续动作？反量化误差怎么处理？</summary>

**答**：**还原流程**：
1. LLM 输出动作 token → 查表得 bin index $i \in [0, 255]$。
2. **Bin 中心还原**：每维动作在训练集 1%-99% 分位区间等分 256 bin，token $i$ → bin 中心值 $\hat{a}_{\text{norm}} = -1 + (i + 0.5)/256 \times 2$。
3. **反归一化**：用训练集的 $(\min, \max)$ 还原到物理量纲 $a = \hat{a}_{\text{norm}} \cdot (\max - \min)/2 + \text{mid}$。

**反量化误差**：
- 每维 ~ $1/256 \approx 0.39\%$ 量程；
- 在 ±10 cm 末端位置上 ≈ 0.78 mm 精度；
- 对粗操作够用，对插针 / 拧螺丝（< 0.1 mm）不够。

**处理方法**：
1. 任务允许就直接用。
2. 高精度任务换 OpenVLA-OFT 连续头。
3. 加 **PID / impedance** 后端 closed-loop 兜底（模型出 desired，控制器精确跟踪）。

**易错**：256 bin 不是"够细"，是 VLM 复用的权衡；精度需求高必须换连续头。

</details>

---

## §7 Sim2Real（2 题）

> 卷四（世界模型 / Sim2Real）会展开；这里只覆盖 VLA 落地视角的高频题。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q41</b> · Domain Randomization 是什么？在 VLA / Sim2Real 中怎么用？有哪些常见随机化维度？</summary>

**答**：**Domain Randomization（DR）**：训练时**随机化仿真参数**，强迫 policy 在分布外参数下仍鲁棒；本质是数据增强 + 隐式 domain adaptation。

**核心理念**：让仿真分布 **覆盖** 真实分布，policy 在 sim 上鲁棒 = 在真机也鲁棒。

**常见维度**：
| 类别 | 随机化项 |
|---|---|
| **视觉** | 纹理、光照、相机内外参、背景、颜色 |
| **物理** | 质量、摩擦系数、关节阻尼、外力扰动、重力 |
| **控制** | 动作延迟、控制器增益、动作噪声 |
| **传感器** | IMU 偏置、相机噪声、关节编码器分辨率 |

**VLA / Sim2Real 用法**：
1. Sim 大规模 DR 训练（pretrain）；
2. 真机 demos fine-tune（domain adaptation）；
3. 与 **RMA** 配合：RMA 引入 latent 编码当前 domain，online 推断；DR + RMA 比纯 DR 强。

**易错**：DR 不是"越多越好"——过强会使任务无解，训练发散；需 curriculum（先弱后强）。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q42</b> · Sim-to-Real Gap 主要来自哪？怎么缩小？在你项目里是怎么处理的？</summary>

**答**：**Gap 来源**：① 视觉（渲染 vs 真实噪声/光照）；② 物理（摩擦/接触/软体/关节弹性仿不准）；③ 控制（仿真无延迟 vs 真机 5-20 ms 延迟）；④ 传感器（IMU 漂移、力觉噪声）。

**缩小方法**：
1. **DR**（sim 端扩分布）；
2. **System ID**（真机数据校准仿真参数）；
3. **Real-world fine-tune**：sim 预训练 + 真机 200-500 demos 精调（最稳）；
4. **Visual transfer**（GAN / NeRF 渲染 photo-realistic）；
5. **Privileged + Distillation**（sim 用真值训，蒸馏给只用图像的 student，RMA 思路）。

**项目实务**：Isaac Gym + DR 预训练（~100 GPU 小时）→ 真机 ~300 demos fine-tune（~1 小时）。

**易错**：sim 训好不能直接用；真机 fine-tune 几乎必须，差异只是 demos 数量。

</details>

---

## §8 双系统 / 系统观 / 前沿判断（10 题）

> L2/L3 开放讨论题；二面 / 终面常考。N1/N2/N4 来自顶级 lab 教授小红书 AMA，破例入选。

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q43</b> · 双系统（System 1 / System 2）VLA 是什么思路？解决了什么问题？快慢系统怎么协同？</summary>

**答**：源自 Kahneman 双系统理论：S1 直觉快、S2 推理慢。

**VLA 映射**：
- **System 1**：小模型（< 1B），**50-100 Hz**，执行 atomic skills。
- **System 2**：大 VLM（3-10B），**1-5 Hz**，做高层任务分解 / 推理 / 重规划。

**协同**：① S2 → S1 输出语义子目标（"抓杯子"）或 latent goal；② S1 检测异常 / 完成 → trigger S2 重规划；③ S2 在 S1 执行间隙异步推理，streaming 喂给 S1。

**代表**：Figure Helix、GR00T-N1、FiS、Hume。

**解决**：单端到端 VLA 无法同时满足"long-horizon planning + high-freq control"——双系统兼顾推理深度 + 控制频率。

**易错**：双系统是**分工 + 异步**，不是集成学习；S1 / S2 是完全不同的模型，频率差 10-50×。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q44</b> · 谈一下 VLA 在自动驾驶里的两种范式（模仿学习派 vs 模拟器 RL 派）的优劣？</summary>

**答**：
| | 模仿学习派 | 模拟器 RL 派 |
|---|---|---|
| 代表 | Tesla FSD、小鹏 XNGP、Wayve | Waymo（部分）、CARLA 研究 |
| 数据源 | **海量真人驾驶视频** | 仿真器 rollout |
| 训练 | BC / VLA 端到端 | RL（PPO / SAC） |
| 优点 | scale 好；数据现成 | 可控、安全、可枚举边界场景 |
| 缺点 | 长尾安全难（demos 不覆盖罕见事件） | Sim2Real gap 严；仿真行为模型不真 |

**主流**：当前模仿学习派**规模化能力更强**（Tesla / 小鹏量产）；但安全和长尾上 hybrid 仍是趋势——主体 IL + 关键场景 RL 兜底。

**与机器人 VLA 区别**：
- 自动驾驶动作维度小（2-4 维：转向 + 油门 + 刹车）；机器人 7-16+ 维。
- 自动驾驶任务结构化（道路 / 路标 / 信号灯），机器人开放场景。
- 自动驾驶决策周期长（秒级），机器人毫秒级。

**易错**：自动驾驶 VLA 和机器人物理操作 VLA 是**两个赛道**，技术栈重叠但落地路径完全不同。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q45</b> · 为什么"缺乏实机经验的多模态大模型研究"在具身岗竞争力有限？VLA 求职者除了懂模型还要懂什么？</summary>

**答**：**核心矛盾**：纯模型研究者擅长 sim / dataset SOTA，但具身岗的真正需求是**"模型 + 系统 + 落地"三栖人才**。

**真机部署的隐藏成本**：
1. **控制频率约束**——50 Hz 控制 + 6 Hz 推理怎么协调。
2. **硬件接口**——ROS / 自研 SDK / EtherCAT 通讯。
3. **数据采集 pipeline**——teleop / VR / kinematic retargeting。
4. **长程任务稳定性**——多步串联时累积误差。
5. **安全机制**——力限 / 紧急停止 / 错误恢复。

**求职者要懂**：
1. **真机数据采集**（teleop 至少跑通一次）。
2. **部署 pipeline**（ROS 基础 + 至少一种实时通讯）。
3. **控制基础**（PID / impedance / MPC 概念）。
4. **至少一个 closed-loop 真机 demo**（哪怕是 LeRobot 复现的简单任务）。

**面试官的潜台词**：能不能让模型在真机上跑起来 ≠ paper SOTA。

**易错**：在简历里写"OpenVLA / π0 实现"但没真机验证，面试官会重点拷打你的 demo 视频。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q46</b> · VLA 的 scaling law 存在吗？数据 / 模型 / 算力哪个维度最难 scale？</summary>

**答**：**现状**：**没有公认的 robot scaling law**——曲线远不如 LLM Chinchilla 清晰。

**论据**：
- RT-2（55B）比 OpenVLA（7B）提升幅度有限；
- 数据量（OXE 970K）远小于 LLM 训练量（10T+ tokens）；
- closed-loop 评估稀缺，scaling 曲线难稳定测量。

**最难 scale 的维度**：
1. **数据 >> 模型 >> 算力**——
2. **数据**：① 真机数据采集 1 人 1 天 100 demos，scale 到 10M+ 需上千人年；② 跨 embodiment / 任务多样性是数据，不是数量；③ 缺乏 self-supervised pretrain 路径（视频是 action-free）。
3. **模型**：参数好加，但小模型 + 多数据当前回报更高。
4. **算力**：单次训练成本可控（OpenVLA 数千 GPU 小时）；瓶颈在数据。

**关键观察**：scaling 的回报在 **"跨 embodiment / 跨任务"** 上比 "单任务参数量" 明显——这是为什么 OXE 比 single-lab 数据有用。

**易错**：不要照搬 Chinchilla 20:1 token:param——机器人 token 信息密度远低于自然语言。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q47</b> · RECAP（π0.6）的核心是什么？怎么把示教数据当 offline RL 用？advantage-conditioned 怎么实现？</summary>

**答**：**RECAP** = **RL with Experience and Corrections via Advantage-conditioned Policies**（π*0.6，2025-11，arXiv:2511.14759）。

**核心三件事**：
1. **异构数据**：demos + on-policy rollouts + **teleoperated corrections**。
2. **Advantage-conditioned 训练**：估算 $A(s,a)$，把 advantage 作为 condition 喂回 policy（类似 DT 的 return-conditioned）。
3. **推理时 condition on high advantage**：生成高质量动作。

**易错**：RECAP 不是 reward-weighted regression、也不是 reward-model + PPO；是**条件生成式 advantage-conditioned VLA RL**；数据上不只 offline demos，还吃 corrections 与 on-policy rollouts。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q48</b> · 现在的通用具身大模型（OpenVLA / π0 / GR00T）还差什么才能"落地"？说说你的判断。</summary>

**答**：**核心差距 = 数据 > 模型 > 算力 > 评估**。

**具体差距**：
1. **数据规模**：OXE 970K 远不够覆盖家庭长尾；需 100M+ trajectories（约现量 100×）。
2. **数据多样性**：场景多样性、任务多样性、长尾事件覆盖（罕见物体、突发干扰）。
3. **推理速度**：6 Hz 不够工业 50 Hz；硬件加速 + 模型蒸馏待突破。
4. **闭环评估**：sim 评估不靠谱；真机 benchmark 太贵；缺统一标准。
5. **安全机制**：紧急停止 / 错误恢复 / 力觉异常未集成到端到端 VLA。
6. **长程任务**：> 30 秒的多步任务稳定性差，需 hierarchical 或 World Model。
7. **经济性**：单台机器人成本 + 训练成本 > 量产替代单价时才商用。

**最难突破的是数据**——硬件 / 模型 / 算力都可"用钱解决"，数据靠人力时间无法压缩；这也是为什么 RoboNet / DROID / OXE 这种共享数据集是行业关键。

**易错**：不要只说"模型还不够大"——这是 LLM 思维；具身的瓶颈一直在数据。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q49</b> · 你怎么看 VLA "走端到端"还是"分层规划 + 低层 policy"？工业界主流路线是什么？</summary>

**答**：
- **端到端 VLA**：one model 直接 obs → action（OpenVLA / π0 / RDT）；优点：简单、scale 友好；缺点：长程任务稳定性差、可解释性弱。
- **分层规划**：High-level LLM 输出子任务（SayCan / Code-as-Policy）+ Low-level policy 执行 atomic skills；优点：可解释、模块化、可调试；缺点：层间接口设计难、容易堆栈耦合。

**工业界主流**：**Hybrid**——
1. 端到端 VLA 做 atomic skill 执行（< 30 秒任务）；
2. LLM 做高层任务分解（如 "做早餐" → "打开冰箱 / 取鸡蛋 / 开火..."）；
3. 双系统 VLA（Helix / GR00T N1）正是这种思路的端到端化。

**π0 自己也提分层**——虽然 π0 模型本身是端到端，但 Physical Intelligence 公开 blog 提到"未来 hierarchical + VLA"是落地必由路径。

**判断关键**：
- 数据极丰富 + 短程任务 → 端到端；
- 长程 + 多步推理 → 分层 / 双系统。

**易错**：不要把"端到端"和"分层"看成对立——它们正在融合（双系统 VLA）。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×2</span> <b>Q50</b> · 端到端 VLA 真的能跑通吗？什么场景下你会优先选 World Model + 分层规划而不是端到端 VLA？</summary>

**答**：**端到端 VLA 挑战**：数据需求海量（OXE 970K 仍不够）、closed-loop 评估难、长程任务复合误差爆炸。

**World Model 路线**：学环境动力学 $\hat{P}(s' \mid s, a)$，用于规划（CEM / MPPI）或 imagination 训 policy（Dreamer / TD-MPC2）。

**优先 WM + 分层场景**：
1. **真机 sample 极贵**——WM 用 1/10-1/100 数据达到 RL 效果。
2. **需 explicit reasoning**——latent 空间可做反事实推理。
3. **长程规划**——规划与执行分层，避免端到端复合误差。
4. **可解释性需求**——WM latent 可观察、可干预。

**端到端 VLA 更好**：数据丰富、反应式毫秒级、多任务 co-train 通用 backbone。

**易错**：WM 不是"过时路线"——工业界两条路线并行，WM 进入决策环路被广泛看好。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>Q51</b> · 具身智能里"操作（manipulation）"和"运动控制（locomotion）"哪个落地更难？为什么操作是"前途无量"但卡点更多？</summary>

**答**：
| | Locomotion | Manipulation |
|---|---|---|
| 落地成熟度 | **已商用**（Boston Dynamics、宇树 Go2） | **实验室阶段**（少量量产） |
| 物理建模 | 浮动基座动力学，仿真较准 | 接触 / 摩擦 / 软体物，仿真极难 |
| 数据稀缺度 | 中（步态数据有共性） | **极高**（每类操作都需新数据） |
| 闭环反馈 | IMU + 关节位置足够 | 需视觉 + 力觉，多模态对齐难 |
| 任务多样性 | 行走 / 跳跃 / 跑步 | **几乎无限**（抓 / 放 / 拧 / 插 / 切...） |

**Locomotion 已是"基础设施"**：Sim2Real + RL 是成熟 pipeline；Unitree、宇树等量产；技术风险已收敛。

**Manipulation 是"前途无量但卡点多"**：
- 卡点：物体多样、接触建模难、数据稀缺、长尾任务无穷。
- 前途：家庭服务 / 工业精细装配 / 医疗 / 物流的核心。

**求职建议**：
- 想长期需求大 → **Manipulation**（容量未饱和）。
- 想短期商用 → **Locomotion**（已成熟，需求集中在工程优化）。

**易错**：Locomotion 不是"简单"——只是"已知怎么做"；Manipulation 是"前沿但慢"。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>Q52</b> · 具身智能的"泛化性"为什么必须靠多模态大模型撑？纯 RL 或纯 IL 路线为什么不够？</summary>

**答**：**具身泛化的四个维度**：跨任务 / 跨场景 / 跨物体 / 跨形态——传统 RL / IL 训练分布外几乎崩。

**多模态大模型的核心价值**：
1. **Web-scale 视觉 + 语言先验**——VLM 见过千万张图、亿级文本；新场景仍能特征匹配。
2. **语言 grounding**——"操作螺丝刀" 这种新任务可通过语言映射到学过的"用细长工具旋转"知识上。
3. **跨任务知识共享**——任务在自然语言空间天然相近，VLM 自动迁移。

**纯 RL 不够**：
- 真机 sample 贵，无法覆盖长尾。
- 探索效率低，复杂任务 reward 难设。
- 跨任务零迁移。

**纯 IL 不够**：
- 演示数据有限（OXE 970K），远不能覆盖家庭长尾。
- 分布外协变量偏移崩。
- 跨场景泛化弱。

**VLA 解法**：VLM backbone 提供"通识 + 语义对齐"，IL / RL 提供"动作生成"——两者结合是当前唯一可见的泛化路径。

**易错**：VLM ≠ 替代物理推理——它补足"语义理解 + 先验"，不能直接生成精确动作；必须配合 action head。

</details>

---

## §A 项目拷打 / 工程实现（6 题）

> 面试官会用项目题考察"系统观 + 真机经验"；答案要套用自己的项目讲，下面是标准答题框架。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q53</b> · 强化学习和模仿学习如何结合？在你的项目里是怎么用的？</summary>

**答**：**5 种主流组合**：
1. **IL 预训练 → RL fine-tune**：BC/DAgger 学初始 → PPO/SAC 精调；工业首选最稳。
2. **联合 loss**：$L = \alpha L_{\text{BC}} + \beta L_{\text{RL}}$；适合 demos 少 + 有 reward。
3. **Offline RL**：CQL / IQL / AWAC 从静态 demos 学，无需在线交互。
4. **GAIL**：discriminator 对抗，无需 reward。
5. **RLHF / RLAIF**：偏好 → reward model → PPO。VLA 后训练也走 RL-from-experience；其中 RECAP 是 **advantage-conditioned 变体**，不是典型 PPO。

**项目答题模板**：BC 预训练 → 定义 reward = task + smoothness → PPO + GAE fine-tune → 关键是 **early stopping + KL 约束** 防偏离 BC 太远。

**易错**：纯 RL 真机 sample 贵；纯 IL 长程崩；hybrid 是工业现实。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q54</b> · 你的项目里奖励函数怎么设计？为什么这么设？尝试过哪些其他形式？</summary>

**答**：**设计原则**：
1. **Dense vs Sparse**：dense 收敛快但易 reward hacking；sparse 难学但目标明确。
2. **Shaping**：加 distance / progress 等中间信号引导。
3. **Multi-component**：`R = task_reward + safety_penalty + smoothness_penalty`，相加形成总奖励。
4. **Curriculum**：早期 dense 加快学习，后期切 sparse 防 hacking。

**项目答题框架（套你的项目讲）**：
- 任务奖励：成功完成给大 +R，关键 sub-event（接触 / 接近）给小正奖励。
- 安全惩罚：碰撞 / 关节限位 / 力超限给负奖励。
- 平滑惩罚：$-\alpha \|\Delta a\|^2$ 防抖动；trade-off：α 大动作平滑但响应慢。
- 尝试过的其他形式：纯 sparse 收敛慢、纯 dense 易 hacking；最终选 hybrid。

**易错**：reward hacking 不靠 reward 曲线判断——要看 rollout 视频是否真完成任务（模型常找"接近但不抓"刷分）。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q55</b> · 你怎么收集数据？真机示教 vs 仿真 vs 人类视频，各自的优劣和落地代价？</summary>

**答**：
| 来源 | 优点 | 缺点 |
|---|---|---|
| **真机示教**（teleop/VR） | 保真度最高、含接触反馈 | 1 demo 30-60s，人工最贵（ALOHA / DROID / Bridge） |
| **仿真** | 无限量、可标注 | Sim2Real gap，接触/软体仿不准（RLBench / ManiSkill / Isaac） |
| **人类视频** | 海量、多样 | 无 action label，需 retargeting（Ego4D / Something / Epic-Kitchens） |

**实务路线**：① 人类视频做 representation 预训练（R3M / VC-1 / VIP）→ ② 仿真大规模 DR pretrain → ③ 真机 ~300 demos fine-tune。

**易错**：不要单一数据源——三者结合才 scale；纯真机太贵，纯仿真 gap 大。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q56</b> · 末端力控有噪声怎么办？阻抗控制 / 导纳控制怎么选？</summary>

**答**：
- **阻抗控制**（Impedance）：输入**期望位置** $x_d$，输出 **force/torque**；行为类似弹簧 + 阻尼，$F = K(x_d - x) + D \dot{x}$。
- **导纳控制**（Admittance）：输入**外力** $F_{\text{ext}}$，输出**位置变化**；行为是"外力推 → 位置响应"。

**选择**：
| 场景 | 选哪个 |
|---|---|
| 机器人**主动**接触（打磨、装配、插针） | **阻抗** |
| 人**主动**引导机器人（co-bot 教学） | **导纳** |
| 高刚度任务（精确定位） | 阻抗（高 K） |
| 高顺应性任务（抓软物） | 阻抗（低 K）或导纳 |

**力觉噪声处理**：
1. **低通滤波**（一阶 IIR / 卡尔曼）去高频抖动；
2. **死区**：小于阈值视为 0；
3. **力 / 力矩传感器标定**：去重力 + 偏置；
4. **冗余传感融合**：关节扭矩 + 末端 F/T 双重估算。

**易错**：阻抗 / 导纳是**对偶**关系，不是同义；阻抗高刚 / 导纳高柔，相反。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×3</span> <b>Q57</b> · 你的观测空间为什么这么设？图像 + proprio + force 多模态怎么对齐 / 归一化？</summary>

**答**：**典型 VLA 观测**：
- **图像**：1-4 帧 RGB，224×224 或 256×256，多视角（手腕相机 + 第三人称）。
- **Proprio**：关节位置（rad）+ 关节速度（rad/s）+ 末端位姿（SE(3)）。
- **Force/Torque**：末端 F/T 传感器 6 维 + 关节扭矩 N 维（可选）。

**对齐 / 归一化**：
1. **图像**：CHW float，预训练 mean/std 归一化（如 ImageNet 的 `[0.485, 0.456, 0.406]`）；多视角 concat 在 channel 维。
2. **Proprio**：每维 $(x - \mu) / \sigma$，$\mu / \sigma$ 来自训练集统计。
3. **Force**：通常归一化到 $[-1, 1]$（±10 N / 1 Nm 量程）；力觉噪声大需先低通滤波。

**时序同步**：相机 30 Hz、关节 1 kHz、F/T 500 Hz——按**最低频率**（如 30 Hz）对齐，其它做插值或最近邻。

**历史窗口**：常用 4-8 帧图像 + 同步 proprio 序列；用 Transformer / RNN 编码时序。

**易错**：图像忘记归一化（直接喂 [0, 255]，权重崩）；proprio 不归一化（不同关节量纲差 100 倍，训练不稳）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q58</b> · 在双臂 / 移动操作场景里，你怎么设计动作空间？关节空间 vs 末端空间，trade-off 在哪？</summary>

**答**：**两种动作空间**：
| | 关节空间（joint） | 末端空间（end-effector） |
|---|---|---|
| 输出 | N 维关节位置 / 速度 / 扭矩 | 6-DoF SE(3) 末端位姿 + gripper |
| 优点 | 直接给电机，无 IK 奇异 | 任务直观（"抓 cup"），数据效率高 |
| 缺点 | 模型需学复杂关节耦合 | 需 IK 解算，奇异点问题 |
| 数据需求 | 高（耦合复杂） | 低（任务对齐自然） |

**VLA 实务**：大多数 VLA 用**末端空间**——
- OpenVLA / RT-2：7 维（6-DoF 末端位姿 + 1 维 gripper）。
- π0：同上 + 多形态适配。
- 双臂：两套末端 = 14 维（ALOHA）。
- Mobile + 双臂：+ 2 维 base = **16 维**（Mobile ALOHA）。

**Trade-off**：
- **数据效率**：末端空间任务对齐好，少 demos 也能学（ACT 50 demos）。
- **物理稳定**：关节空间避免 IK 奇异，但模型要学更复杂映射。
- **跨 embodiment**：末端空间易标准化；关节空间因 DoF 不同难统一。

**易错**：末端空间的 IK 奇异 / 关节限位问题需要**底层控制器 + 安全检查**兜底，不能让模型完全裸跑。

</details>

---

## §H 手撕代码（8 题）

> IL / VLA 手撕段——与上文"概念+答案"题区分开。本节每题给"考察点 / 实现 / 易错"——附 ≤30 行 Python 实现。

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>✍ H01</b> · 手撕 Behavior Cloning（连续 / 离散动作）</summary>

**考察点**：BC 本质是监督学习；连续 MSE / Gaussian NLL、离散 CE；根本限制是 distribution shift（无 reward 信号）。

**实现**：

```python
import torch
import torch.nn.functional as F

def bc_loss_mse(pi, s, a_star):
    # 连续 deterministic：action 须先归一化到 [-1,1]，否则量纲不一压不住
    return F.mse_loss(pi(s), a_star)

def bc_loss_gaussian(pi, s, a_star, log_std_min=-5.0, log_std_max=2.0):
    # 连续 stochastic：模型出 μ, log_σ；clamp log_σ 防数值发散
    mu, log_std = pi(s)
    log_std = log_std.clamp(log_std_min, log_std_max)
    # -log N(a*; μ, σ) 的可降项：0.5·((a-μ)/σ)² + log σ + const
    return (0.5 * ((a_star - mu) / log_std.exp()).pow(2) + log_std).sum(-1).mean()

def bc_loss_ce(pi, s, a_token):
    # 离散（OpenVLA-style）：a_token 是离散 bin id
    return F.cross_entropy(pi(s), a_token)
```

**易错**：把 BC 当 RL；忘 action 归一化；`log_σ` 不 clamp 数值发散；离散用 MSE（应 CE）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×8</span> <b>✍ H02</b> · 手撕 Diffusion Policy 训练 step（DDPM ε-prediction）</summary>

**考察点**：`x_t=√ᾱ_t·x_0+√(1-ᾱ_t)·ε`；ε-prediction 目标；视觉 obs 作为 cross-attn condition。

**实现**：

```python
import torch
import torch.nn.functional as F

def make_alpha_bar(T=1000):
    # cosine schedule (Nichol & Dhariwal)：ᾱ_t = f(t)/f(0)，比 linear 末段更稳
    steps = torch.arange(T + 1) / T
    f = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2).pow(2)
    return (f / f[0])[1:]  # 长度 T，单调递减，保证 1-ᾱ_t ∈ (0,1)

def dp_train_step(model, x0, cond, alpha_bar):
    # x0: [B, H, action_dim] 未来 H 步 action chunk
    # cond: [B, D_cond] 视觉特征（不能漏，否则退化为无条件采样）
    B = x0.size(0)
    t = torch.randint(0, len(alpha_bar), (B,), device=x0.device)
    ab = alpha_bar[t].view(B, 1, 1)  # 广播到 [B,H,action_dim]
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps  # forward q(x_t|x_0)
    eps_pred = model(xt, t, cond=cond)
    return F.mse_loss(eps_pred, eps)  # ε-prediction 目标
```

**易错**：漏掉视觉 cond 进 UNet；把 action 当图像 spatial 维；推理 K 步太多（DDIM 10-50 即可）；不用 EMA 生成抖。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>✍ H03</b> · 手撕 ACT action chunking loss</summary>

**考察点**：chunk 长度 k 缓解 distribution shift；CVAE latent z 建模动作多模态；temporal ensemble 推理。

**实现**：

```python
import torch
import torch.nn.functional as F

def act_loss(decoder, encoder, qpos, action_chunk, cond, beta=10.0):
    # action_chunk: [B, k, action_dim]；cond: 视觉特征
    # encoder 输入 [qpos, action_chunk]，[CLS] token 出 μ, log_σ²
    mu, logvar = encoder(qpos, action_chunk)             # [B, z_dim] each
    std = (0.5 * logvar).exp()
    z = mu + std * torch.randn_like(std)                  # 训练时采样 (reparam)
    # decoder：cross-attn(query=fixed pos, k/v = cond ⊕ qpos ⊕ z)
    a_hat = decoder(qpos=qpos, cond=cond, z=z)            # [B, k, action_dim]
    # 重建损失用 L1（ACT 论文实验上比 MSE 更稳）
    recon = F.l1_loss(a_hat, action_chunk)
    # KL(q‖N(0,I)) 闭式
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
    return recon + beta * kl

def act_inference(decoder, qpos, cond, z_dim):
    # 推理时 z=0 不再采样，避免 latent 抖动
    z = torch.zeros(qpos.size(0), z_dim, device=qpos.device)
    return decoder(qpos=qpos, cond=cond, z=z)
```

**易错**：chunk 太短退化为 BC；β 太大压死 latent；推理仍从 latent 采样导致抖动；recon 用 MSE 反而不如 L1 稳。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>✍ H04</b> · 手撕 OpenVLA 风格 action token CE loss</summary>

**考察点**：连续动作离散到 256 bin → LM head next-token CE；用 quantile 切分（非 min/max）匹配数据分布。

**实现**：

```python
import torch
import torch.nn.functional as F

def fit_bins_quantile(actions, n_bin=256):
    # actions: [N, D]；按 quantile 而非 min/max 切分，防长尾压缩
    qs = torch.linspace(0, 1, n_bin + 1, device=actions.device)
    # 每维独立切：返回 [D, n_bin+1] 的边界
    return torch.stack([torch.quantile(actions[:, d], qs) for d in range(actions.size(1))])

def discretize(action, bins, vocab_offset):
    # action: [B, D]；bins: [D, n_bin+1]
    # 用 bucketize 找落在哪个区间 → token id（偏移到词表预留段）
    token_ids = torch.stack([
        torch.bucketize(action[:, d], bins[d, 1:-1])  # 去掉首尾哨兵
        for d in range(action.size(1))
    ], dim=1)
    return token_ids + vocab_offset  # 落到词表预留的 action token 段

def openvla_loss(logits, target_ids, action_mask):
    # logits: [B, L, V]；target_ids: [B, L]；action_mask 仅在 action token 处为 1
    # 屏蔽 prompt/image token 的 loss，只学 action 段
    return F.cross_entropy(logits.transpose(1, 2), target_ids,
                           reduction='none').mul(action_mask).sum() / action_mask.sum().clamp(min=1)
```

**易错**：用全局 min/max 切 bin 长尾压缩；mask 后 target_ids 仍要是合法 vocab id（用 `ignore_index=-100` 时改写法）；推理用 sampling 而非 argmax 致动作抖。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>✍ H05</b> · 手撕 CLIP / InfoNCE 对比损失</summary>

**考察点**：对称 CE；温度 τ 用可学的 `exp(t)` + clamp；为何 i→t 与 t→i 各算一次（对齐两个方向）。

**实现**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, init_t=0.07):
        super().__init__()
        # 学 log(1/τ)；用 exp 保证 τ>0；clamp 防梯度爆炸
        self.log_inv_t = nn.Parameter(torch.tensor(1.0 / init_t).log())

    def forward(self, img_feat, txt_feat):
        # 必须 L2 归一化，否则点积≠cosine
        img_emb = F.normalize(img_feat, dim=-1)
        txt_emb = F.normalize(txt_feat, dim=-1)
        # 对照官方 CLIP：clamp log(1/τ) ≤ ln(100) ≈ 4.605
        scale = self.log_inv_t.clamp(max=4.605).exp()
        logits = scale * img_emb @ txt_emb.T  # [N, N]
        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        # 对称：i→t 沿行做 softmax；t→i 沿列做 softmax
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2
```

**易错**：忘 L2 归一化点积≠cosine；τ 未 clamp 梯度爆炸；batch 太小负样本不足；只算单向 CE 失对称性。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>✍ H06</b> · 手撕 DAgger（数据聚合）训练流程</summary>

**考察点**：解决 BC 的 distribution shift；每轮 rollout 拿专家标注新数据回灌；β-mix 早期靠专家、后期靠 π。

**实现**：

```python
import random

def dagger_train(pi, expert, env, train_fn, n_iter=10, rollout_T=200):
    # train_fn(pi, dataset) -> 新 pi；专家与环境为黑箱
    dataset = expert.collect()  # D_0：纯专家轨迹（先 BC 起步）
    pi = train_fn(pi, dataset)
    for i in range(1, n_iter + 1):
        beta = max(0.05, 0.9 ** i)  # 退火不要太快，保留早期专家引导
        obs = env.reset()
        traj_obs, traj_act = [], []
        for _ in range(rollout_T):
            # β-mix rollout：以 β 用专家动作走、否则用 π；标注始终来自专家
            a_exec = expert.act(obs) if random.random() < beta else pi.act(obs)
            traj_obs.append(obs)
            traj_act.append(expert.act(obs))  # 关键：标注用专家在该 obs 的动作
            obs, _, done, _ = env.step(a_exec)
            if done:
                break
        dataset.extend(list(zip(traj_obs, traj_act)))  # 只加不替换
        pi = train_fn(pi, dataset)
    return pi
```

**易错**：标注用执行动作而非专家在该 obs 的动作（破坏 DAgger 本质）；β 退火太快首轮就全靠 π；数据集替换而非累加；忽略 query 选择策略致标注爆炸。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>✍ H07</b> · 手撕 cross-attention（视觉-语言融合）</summary>

**考察点**：与 self-attn 唯一区别 = K/V 来自另一序列；输出长度跟 Q；mask 形状 `[L_q, L_kv]`。

**实现**：

```python
import math
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.Wq = nn.Linear(d_q, d_model)
        self.Wk = nn.Linear(d_kv, d_model)  # K/V 从另一序列投影
        self.Wv = nn.Linear(d_kv, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv, mask=None):
        # x_q: [B, Lq, d_q]; x_kv: [B, Lkv, d_kv]
        B, Lq, _ = x_q.shape
        Lkv = x_kv.size(1)
        q = self.Wq(x_q).view(B, Lq, self.h, self.d_k).transpose(1, 2)
        k = self.Wk(x_kv).view(B, Lkv, self.h, self.d_k).transpose(1, 2)
        v = self.Wv(x_kv).view(B, Lkv, self.h, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:  # mask: [B, 1, Lq, Lkv]
            scores = scores.masked_fill(mask, float('-inf'))
        out = F.softmax(scores, -1) @ v  # 输出长度 = Lq
        return self.Wo(out.transpose(1, 2).contiguous().view(B, Lq, -1))
```

**易错**：把 K/V 也从 `x_q` 投影（变 self-attn）；mask 形状按 self-attn 设；V 忘换序列。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>✍ H08</b> · 手撕 KV cache（增量解码）</summary>

**考察点**：prefill 后单步只算新 token 的 Q；旧 K/V 复用而非重算；显存随序列线性增长。

**实现**：

```python
import math
import torch
import torch.nn.functional as F

def attn_with_kv_cache(q_new, k_new, v_new, kv_cache, layer_idx):
    # q_new/k_new/v_new: 当前 step token 的 q,k,v，形状 [B, h, 1, d_k]
    # kv_cache[layer_idx] = (K_past, V_past)，shape [B, h, L_past, d_k]
    if kv_cache[layer_idx] is None:
        K, V = k_new, v_new  # prefill 首步
    else:
        K_past, V_past = kv_cache[layer_idx]
        # 沿序列维 concat；旧 K/V 不再重算（这是 cache 核心）
        K = torch.cat([K_past, k_new], dim=2)
        V = torch.cat([V_past, v_new], dim=2)
    kv_cache[layer_idx] = (K, V)  # 写回
    d_k = q_new.size(-1)
    # decode 单步只算 [B,h,1,L_new] 的 scores，不需要 causal mask
    scores = q_new @ K.transpose(-2, -1) / math.sqrt(d_k)
    return F.softmax(scores, -1) @ V  # [B, h, 1, d_k]
```

**易错**：每步把整段重算（失去 cache 意义）；忘对新 K 也加 RoPE（位置错）；显存爆但没 paged 管理；并发请求共享同一 cache。

</details>

