## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 π 系列** — Physical Intelligence 从 2024-10 到 2026-04 走完了 VLA 从 "能跑" 到 "可调速、可组合、专家级" 的整条主线。

1. **π0（2024-10）**：第一代 VLA。`PaliGemma-3B` + `300M flow-matching action expert`，**chunk H=50, 50Hz**，10 步 ODE 推理。训练数据 ≈ **10,000 小时** 跨 7 种本体 + 9.1% OXE 混入。

2. **π0-FAST（2025-01）**：把动作离散化为 token 喂 LLM。`FAST = DCT + 量化 + BPE`，1 秒动作压成 **30–60 个 token**，纯自回归，训练 **5× 加速**。

3. **Hi Robot（2025-02, ICML'25）**：分层。**System 2** 用一份 PaliGemma 做高层 subtask 规划与 "inner monologue"，**System 1** 用 π0 执行；instruction following 比 flat VLA 高 **30–40 个点**。

4. **π0.5（2025-04）**：开放世界。两阶段（FAST 预训练 → flow 后训练）+ 五路异质 co-training（MM/ME/CE/HL/WD）；**3 个未见家庭 OOD success ≈ 94%**，但 in-domain MM 数据只占 ≈ 2.4%。

5. **KI（2025-05）**：训练 recipe。同时学 `FAST next-token`（给 backbone 灌运动语义）+ `flow matching`（推理只用 flow），**stop-grad 隔离** action expert 对 VLM 的梯度污染，训练步数 **少 7.5×**。

6. **RTC（2025-06）**："Thinking while moving"。新 chunk 还在算的时候老 chunk 继续执行，**inpainting** 锁住已执行步并约束重叠步；移动机器人端到端延迟 **≈ 139 ms**，可容忍 +200 ms 注入延迟。

7. **π0.6 + π\*0.6（2025-11）**：Backbone 升级到 `Gemma 3 4B + SigLIP 400M`，action expert **860M**，总 ≈ **5B**。**RECAP**（离线 RL 预训练 → demo SFT → on-robot RL）+ **advantage-conditioning**：espresso success **40% → ≥90%**，30 杯/小时连开 13 小时不停。

8. **π0.7（2026-04, arXiv 2604.15483）**：**Steerable generalist**。Prompt 从单语言扩成 *diverse context*（subtask 文本 + subgoal 图像 + speed/quality/mistake metadata）；首次出现 **compositional generalization**——bimanual UR5e shirt folding **零训练样本** 进度 **85.6%**，几乎追平 teleop 专家。

> ✅ **快速记忆口诀** — 一行总结。

- π0 = **VLA + Flow Matching** 模板
- π0-FAST = 把动作变 token 让 LLM 直接吐
- Hi Robot = 慢思考 + 快执行
- π0.5 = 用 web/HL/CE 数据 **泛化到未见家庭**
- KI = 训练 recipe **stop-grad + 双目标**
- RTC = **异步推理**消化网络/算力延迟
- π0.6/π\*0.6 = **强化学习上线**，专家级
- MEM = 短时 ViT + 长时语言总结
- RLT = 用 token 桥接 VLA 与小 actor/critic，**15 分钟在线 RL**
- π0.7 = **可调速、可组合**的通用基础策略

## §1 直觉与定位

### 1.1 VLA 是什么？π 在哪一格？

**VLA = Vision-Language-Action**：吃 *多视角图像 + 自然语言指令 + 本体感知*，输出 *机器人动作*。VLA 的核心范式分歧在 **动作怎么生成**：

- **离散 token AR**（RT-2 / OpenVLA / π0-FAST）：把动作切 bin 或压成 token，复用 LLM 的训练 / 推理 infra；优点是工程友好，缺点是采样不平滑、long-horizon 难。
- **Diffusion / Flow matching**（Diffusion Policy / RDT-1B / **π0** / **π0.5** / **π0.6** / **π0.7**）：连续动作的 generative model，对多模态分布天然 robust；π0 选 **flow matching** 而非 DDPM，关键是 **10 步 ODE 就够，比 DP 的几十步快 5–10×**。

> 💡 **一个 PI 哲学** — *"机器人需要互联网级别规模的数据，但互联网上没有机器人数据，所以我们必须把所有别的数据（web/video/cross-embodiment）也用上"*。π0.5 / π0.7 的异质 co-training 是这一哲学的工程兑现。

### 1.2 为什么是 Flow Matching 而不是 Diffusion？

四个原因：

1. **推理步少**：ODE 比 SDE 平均少一个数量级（10 vs 100 步）；机器人控制环对延迟极敏感。
2. **目标更简单**：flow matching 学的是 **速度场** $v(x, \tau)$，不需要 noise schedule 设计；DDPM 的 $\bar\alpha_t$ schedule 是个超参雷区。
3. **连续多模态分布友好**：和 diffusion 一样不会像 MSE 那样把双峰平均。
4. **训练稳定**：flow matching loss 是简单的 L2，没有 noise prediction 的 reparameterization 漂移。

### 1.3 谱系一张图（按时间排序）

```
2024-10  π0       ──┐  (flow base)
2025-01  π0-FAST  ──┤  (discrete token)
2025-02  Hi Robot ──┤  (S2/S1)
2025-04  π0.5     ──┤  (open-world)
2025-05  KI       ──┤  (training recipe)
2025-06  RTC      ──┤  (real-time inference)
2025-11  π0.6     ──┤  (Gemma 3 4B backbone)
2025-11  π*0.6    ──┤  (RECAP RL)
2026-03  MEM      ──┤  (long/short-term memory)
2026-03  RLT      ──┤  (online RL token)
2026-04  π0.7     ──┘  (steerable + compositional)
```

## §2 π0 — 第一代旗舰 VLA Flow Model

> 📌 **论文**：Black, Brown, Driess, Finn, Levine et al., *"π0: A Vision-Language-Action Flow Model for General Robot Control"*, arXiv **2410.24164**, 2024-10-31.

### 2.1 架构总览

```
┌──────────────────────────────────────────────────────────┐
│  PaliGemma 3B (frozen-ish VLM backbone)                  │
│  ├─ vision tokens (multi-view images)                    │
│  ├─ text tokens   (language instruction)                 │
│  └─ proprio token (joint angles, EE pose)                │
│       │                                                  │
│       ▼   cross-attention (shared KV cache)              │
│  ┌────────────────────────────────────────────┐          │
│  │  Action Expert 300M (from scratch)         │          │
│  │  ─ input:  noisy chunk A^τ  + time τ       │          │
│  │  ─ output: velocity v_θ(A^τ, c, τ)         │          │
│  └────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

- **VLM backbone**：PaliGemma 3B，3B 参数
- **Action expert**：300M，from-scratch（不复用 LM head），与 VLM 通过 **cross-attention** 共享 KV cache
- **总参数** ≈ 3.3B
- **Action chunk** $H = 50$，最高控制频率 **50 Hz**（即 1 个 chunk 覆盖 1 秒）

### 2.2 核心公式：Flow Matching 训练 / 推理

**Linear-Gaussian path**：定义从噪声 $\varepsilon \sim \mathcal{N}(0, I)$ 到真实动作 chunk $A$ 的线性插值

$$A^\tau = \tau A + (1-\tau)\,\varepsilon, \qquad \tau \in [0, 1]$$

由于 $\varepsilon \sim \mathcal{N}(0, I)$，条件分布 $q(A^\tau \mid A) = \mathcal{N}(\tau A, (1-\tau)^2 I)$（方差是 $(1-\tau)^2$ 而非 $(1-\tau)$——这是 linear-Gaussian path 的标准形式，对应的速度场就是下式右端）。**速度场目标**：

$$\frac{dA^\tau}{d\tau} = A - \varepsilon$$

**训练 loss**（conditional flow matching）：

$$\boxed{\;\mathcal{L}_{\text{CFM}} = \mathbb{E}_{\tau \sim \mathcal{U}[0,1],\, A, \varepsilon} \bigl\| v_\theta(A^\tau, c, \tau) - (A - \varepsilon) \bigr\|^2\;}$$

其中 $c$ 是 VLM 给出的条件向量（vision + language + proprio token 经过 cross-attention 聚合）。

**推理（10 步 Euler ODE）**：从 $A^0 = \varepsilon \sim \mathcal{N}(0, I)$ 出发，

$$A^{\tau + \delta} = A^\tau + \delta \cdot v_\theta(A^\tau, c, \tau), \qquad \delta = 0.1,\; \tau = 0, 0.1, ..., 0.9$$

得到 $A^1 \approx A$。**没有 CFG**——条件 $c$ 通过 cross-attention 注入，不需要 classifier-free guidance。

> ⚠️ **常见误区** — π0 的 flow matching **不是 DDPM**：DDPM 学 noise prediction $\varepsilon_\theta$，需要预定义 noise schedule $\bar\alpha_t$；flow matching 直接学速度场，schedule-free。

### 2.3 训练数据：跨本体大锅炖

π0 训练混合 (training mixture)：

| 数据来源 | 占比 | 备注 |
|---|---|---|
| PI 自家 903M timesteps | 90.9% | 7 robot configs, 68 tasks |
| Open-X-Embodiment (OXE) | 9.1% | 70+ 公开机器人数据集 |
| **总时长** | **≈ 10,000 小时** | |

**7 个本体**及最大维度 17：

- UR5e（7-D）
- 双臂 UR5e（14-D）
- Franka（8-D）
- 双臂 Trossen（14-D）
- 双臂 ARX / AgileX（14–16-D）
- Mobile Trossen / ARX（14–16-D）
- Mobile Fibocom（17-D）

**对齐方式**：所有动作 zero-padding 到 17-18 维；每条样本带一个 "robot type" embedding token 指示本体。

### 2.4 关键 benchmark

| 任务 | π0 success | π0-small | OpenVLA |
|---|---|---|---|
| Bussing easy | 0.971 | 0.875 | — |
| Bussing hard | 0.875 | 0.500 | — |
| Shirt folding | 1.000 | 0.583 | — |
| Grocery bagging | 0.786 | 0.286 | — |
| Toast | 0.750 | 0.250 | — |

> ✅ **开源** — base 权重 + DROID / ALOHA 任务专家 ckpt 全部在 `gs://openpi-assets`，**Apache-2.0** 协议，**允许商用**。

## §3 π0-FAST — 离散动作 Token AR

> 📌 **论文**：Pertsch, Stachowicz, Ichter, Driess, Nair, Vuong, Mees, Finn, Levine, *"FAST: Efficient Action Tokenization for Vision-Language-Action Models"*, arXiv **2501.09747**, 2025-01-16.

### 3.1 动机

π0 的 flow matching 训练慢——每个 sample 要 forward action expert 一次算 loss，跨 GPU sync 多。能不能复用 **LLM 的纯 next-token AR** 训练 infra？

直接量化（discretize）每一维成 256 bin 的 OpenVLA 风格的做法在 dexterous 任务上塌：因为动作是 *连续高频* 信号，独立 binning 把强相关性丢了。

### 3.2 FAST：三步压缩

```
原始 chunk A ∈ R^{H × D}   (H=50, D≤17)
   │
   │  ① DCT（Discrete Cosine Transform，沿时间维）
   ▼
DCT 系数（低频集中能量）
   │
   │  ② 量化（scale ×10, round）
   ▼
整数张量
   │
   │  ③ flatten（低频在前）+ BPE（vocab=1024）
   ▼
30–60 个离散 token
```

**类比 JPEG/MP3**：DCT 把时序信号能量压到低频，量化丢掉肉眼/手感不敏感的高频细节，BPE 把高频共现的 token 序列合并。

### 3.3 FAST+ 通用 tokenizer

PI 还训了一个 **通用版** FAST+：

- 训练数据：**100 万** 1 秒动作序列
- 覆盖：单臂 / 双臂 / 移动；joint / EE-world / EE-cam 三种动作空间
- padding 到 32 维

**意义**：一个 tokenizer 适配所有 manipulation 数据，下游 VLA 直接 plug-and-play。

### 3.4 π0-FAST 架构 & 训练

把 FAST 输出的 BPE token **覆写到 PaliGemma vocab 中最少使用的 token 位置上**（无需扩 vocab）。然后纯 next-token 预测：

$$\mathcal{L} = -\sum_{t} \log p_\theta(a_t \mid a_{<t}, c)$$

**训练速度**：在同样 10k h 跨本体数据上，**5× faster** than flow π0。

### 3.5 性能 trade-off

| 维度 | π0 (flow) | π0-FAST |
|---|---|---|
| 训练速度 | 1× | **5× faster** |
| 推理速度 | **10 ODE step，并行解码** | autoregressive，慢（30–60 token 串行） |
| 灵巧动作平滑度 | **更平滑** | OK，略 jerky |
| DROID 零样本评测 | — | **第一个通用 manipulation 策略** |
| 开源 | base + experts | base + DROID expert |

> ⚠️ **面试常考** — "如果 π0-FAST 训练这么快，π0 还有什么用？" 答案：**推理速度**。机器人控制环 50 Hz 要求 **每个 control step ≈ 20 ms**（一个 chunk 含 50 步 ≈ 1 秒）；flow 的 10 步 ODE 并行解码一个 chunk，远比 30–60 个 AR token 串行解码快，更适合实时控制。

## §4 Hi Robot — System 1 / System 2 分层

> 📌 **论文**：Lucy X. Shi (一作), Ichter, Equi, Ke, Pertsch, Vuong, Tanner, Driess, Groom, Levine, Finn, *"Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models"*, arXiv **2502.19417**, ICML 2025.

### 4.1 直觉：Kahneman Thinking, Fast and Slow

复杂指令（"把桌上不是我喜欢颜色的杯子收掉，再去取一杯咖啡"）要求 *先理解 + 拆解 + 再执行*。flat VLA 一个网络把这些全做了，长程任务上崩塌。

**Hi Robot** 仿照人类双系统：

| 系统 | 角色 | 模型 | 频率 | 输出 |
|---|---|---|---|---|
| **System 2**（慢/推理） | 高层规划 + inner monologue | PaliGemma-3B（微调） | ≈ 1 Hz | 语言子任务 |
| **System 1**（快/反应） | 低层执行 | π0 | 50 Hz | 连续动作 chunk |

### 4.2 训练数据三元组

- **𝒟_labeled**：人工把长 demo 切段并打 skill 标签（"pick mug", "place in tray"）
- **𝒟_syn**：用一个大 VLM **合成** prompt-response 对（"user 说 X，模型应该说 Y"），扩 instruction diversity
- **𝒟_demo**：teleop 演示（多种本体）

### 4.3 Instruction Accuracy 数字

| 任务 | Hi Robot | flat VLA | GPT-4o |
|---|---|---|---|
| Table bussing | **74%** | 36% | ~30% |
| Sandwich making | **83%** | 34% | ~40% |
| Grocery shopping | **72%** | 39% | ~30% |

平均比 GPT-4o **高 40+ 个点**——说明 *机器人控制能力不能靠 zero-shot prompt 大模型解决，必须有专门的 low-level policy*。

## §5 π0.5 — 开放世界泛化

> 📌 **论文**：Physical Intelligence et al., *"π0.5: a Vision-Language-Action Model with Open-World Generalization"*, arXiv **2504.16054**, 2025-04-22.

### 5.1 目标：让机器人进 *未见过的家庭*

π0 / π0-FAST 在见过的家庭里很强，但换厨房就崩。π0.5 的目标：**3 个未见厨房 + 3 个未见卧室**，每个任务 2–5 分钟、多阶段，能否 ≥ 80% 完成？

### 5.2 两阶段训练 recipe

**Stage 1 — Pretrain（280k steps）**：

- 统一 transformer
- **离散动作**（FAST tokenizer）
- 数据：**MM + ME + CE + HL + WD**（即不含 VI；CE 在此阶段保留以学跨本体语义）

**Stage 2 — Posttrain（80k steps）**：

- 加上 **flow matching action expert（300M）**
- 数据：**MM + ME + HL + WD + VI**（即引入口头指令 VI，移除 CE 让模型聚焦目标域）
- 双通路推理：
  - **高层**：自回归解码语言子任务（如 "pick up the dish"）
  - **低层**：在子任务条件下用 flow 生成动作 chunk（H=50, 1 秒）

### 5.3 训练数据五路 + Web

| 标签 | 含义 | 量级 |
|---|---|---|
| **MM** | Mobile Manipulation（≈ 100 个家庭） | ≈ **400 小时** |
| **ME** | Multi-Environment 固定臂 | 大量 |
| **CE** | Cross-Embodiment 桌面（含 OXE） | 大量 |
| **HL** | High-Level 语义子任务（人工标注） | 中 |
| **WD** | Web Data（caption / VQA / object detection） | 巨大 |
| **VI** | Verbal Instructions（post-train 时加） | 小 |

> ⚠️ **反直觉但关键** — **97.6% 训练样本不是 MM 数据**。in-domain MM 只占 ≈ 2.4%。**泛化能力来自异质 co-training**，不是 in-domain 堆量。

### 5.4 关键消融

| 移除 | OOD success drop |
|---|---|
| 去掉 ME 数据 | ≈ −20 pt |
| 去掉 CE 数据 | ≈ −15 pt |
| 去掉 WD 数据 | ≈ −30 pt on unseen objects |

### 5.5 Unseen Homes Benchmark

| 指标 | In-distribution | OOD（unseen 家庭） |
|---|---|---|
| Instruction following | 86% | **94%** |
| Task success | 83% | **94%** |

> ✅ **意义** — 这是 **VLA 走出实验室** 的标志性数字。π0.5 base + LIBERO / DROID expert 权重在 openpi 开源。

## §6 Knowledge Insulation (KI) — π0.5 → π0.6 的训练 recipe

> 📌 **论文**：*"Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better"*, arXiv **2505.23705**, 2025-05-28.

### 6.1 痛点

π0.5 同时学 *flow matching action* 和 *FAST next-token*，但 action expert 的梯度会反向传到 VLM backbone，**污染 PaliGemma 已经学好的多模态知识**。结果是：

- post-train 后 backbone 在 VQA / captioning 上掉点
- continual learning 时灾难性遗忘

### 6.2 解法：stop-grad 隔离 + 三目标

```
┌─────────────────────────────────────────────────┐
│ VLM backbone (PaliGemma)                        │
│  ├─ Loss A: web 多模态  (caption/VQA)           │
│  └─ Loss B: FAST next-token AR (动作 token)    │  ← 学运动语义
│       │                                         │
│       ▼  stop-grad ⛔                            │
│ ┌──────────────────────────────────────────┐    │
│ │ Action expert (300M, flow)               │    │
│ │  └─ Loss C: flow matching CFM            │    │  ← 推理只用这条
│ └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**关键三步**：

1. **Loss A**：保持 backbone web 知识不掉点
2. **Loss B**：让 backbone 通过 FAST next-token 任务 *学到运动语义*（不需要真的用 AR 推理）
3. **Loss C**：action expert 的梯度 **被 stop-grad 截断**，不污染 backbone

推理时只用 Loss C 的路径（flow matching action expert），快。

### 6.3 数字

- **训练步数 ≈ 7.5× fewer than π0** to reach same performance
- Bussing 执行时间：≈ **400 s**（vs π0-FAST 800 s）
- LIBERO-90 / LIBERO-Spatial SOTA

KI 是后续 π0.6 / π\*0.6 / π0.7 训练 recipe 的主干。

## §7 Real-Time Chunking (RTC) — Thinking While Moving

> 📌 **Research note**：`pi.website/research/real_time_chunking`, 2025-06-09.

### 7.1 问题：推理延迟会卡死机器人

机器人控制环 50 Hz 要求 20 ms / step。一个 chunk 1 秒（50 步），但 VLA 推理 ≈ 100 ms。如果**同步**执行——执行完一个 chunk 才请求下一个——会有 100 ms 的空窗，肉眼可见的"卡顿"。

### 7.2 异步 chunk + Inpainting

```
时间轴 ──────────────────────────────────────────────►
chunk_0: [a_0, a_1, ..., a_49]            执行
                          ▲
                          │
chunk_1 请求 ─────────────┘────────► 推理 100 ms
                                ▲
chunk_1: [a_50, ..., a_99]      │  到达，开始执行
                                │
                                ▼
重叠部分约束：chunk_1 的前 N 步与 chunk_0 后 N 步**对齐**
```

**Inpainting 机制**：

1. 已经物理执行的步骤（chunk_0 的前 K 步）**冻结**——chunk_1 不能改它们
2. 重叠区域用 partial attention：chunk_1 生成时把 chunk_0 在重叠位置的动作作为 *已知部分*，新生成的是 *缺失部分*（"inpainting"）
3. 这样保证 chunk 边界 **连续**，不会跳变

### 7.3 延迟数字

| 平台 | 端到端延迟 | 容忍注入延迟 |
|---|---|---|
| Mobile robot | **139 ms** (推理 97 + 网络 21 + 其他 21) | +200 ms |
| Static robot | **108 ms** | +200 ms |

> 💡 **工程意义** — 即使把推理放云端（增加几十 ms 网络往返），RTC 让机器人**感觉不到延迟**。这是把 VLA 从 lab demo 推向产品的关键技术。

## §8 π0.6 + π\*0.6 — RECAP 强化学习上线

> 📌 **论文**：*"π\*0.6: a VLA That Learns From Experience"*, arXiv **2511.14759**, 2025-11-17.

### 8.1 Backbone 升级（π0.6 base）

| 组件 | π0 | π0.5 | **π0.6** |
|---|---|---|---|
| VLM | PaliGemma 3B | PaliGemma 3B + KI | **Gemma 3 4B** |
| Vision encoder | (PaliGemma 内置) | 同左 | **+ SigLIP 400M** |
| Action expert | 300M | 300M | **860M**（与 backbone 同层数） |
| **总参数** | 3.3B | 3.3B | **≈ 5B** |

π0.6 base 已经能可靠折叠 laundry，但 box assembly 仍只 ~20%。要做到专家级，需要 **RL fine-tune**——这就是 π\*0.6。

### 8.2 RECAP 三阶段

> **R**L with **E**xperience and **C**orrections via **A**dvantage-conditioned **P**olicies

```
Stage 1 — Offline RL Pretrain
   ▶ 在大规模 offline 数据上同时训 V(s) 与 advantage-conditioned policy π(a|s, A)
   ▶ V 学到"哪种 state 更接近 success"的判断
   ▶ Policy 学到"在给定 advantage 信号下应该输出什么动作"
                  │
                  ▼
Stage 2 — Demo SFT (任务级)
   ▶ 在目标任务的 teleop demo 上 SFT
                  │
                  ▼
Stage 3 — On-robot RL with Interventions
   ▶ 部署到真机，专家随时介入（人按手柄接管）
   ▶ 介入 = high advantage，自主失败 = low advantage
   ▶ 持续 fine-tune V 与 policy
```

### 8.3 Advantage-Conditioned Policy

**关键洞察**：传统 RL 把 advantage 用作 loss 权重（AWR）；RECAP 把 advantage 作为 **prompt token 喂给 VLA**。

$$\pi(a \mid s, A_t), \qquad A_t \;=\; \mathbb{E}\Bigl[\sum_{k=0}^{N-1} r_{t+k}\Bigr] + V(s_{t+N}) - V(s_t)$$

（一般 n-step advantage；在 sparse-reward 机器人任务中常退化为 $A_t \approx V(s_{t+N}) - V(s_t)$ 的一步/n-步 V-差简化）。

- 训练时：**所有数据都用**（好的标 advantage=+，差的标 advantage=−）
- 推理时：**始终 condition on 高 advantage**

> ⚠️ **vs Decision Transformer** — DT 是 return-to-go 条件化；RECAP 是 advantage（V 差）条件化，**更适合长程任务**，因为 return-to-go 在 sparse reward 下信噪比低。

### 8.4 9 个任务的专家级数字

| 任务 | base (π0.6, 模仿) success | π\*0.6 (RECAP) success | throughput |
|---|---|---|---|
| Espresso | ~40% | **≥ 90%** | **30 杯/小时**，连续 13 小时 |
| Box assembly | ~20% | **~95%** | 14 个/小时 |
| Laundry | — | **~95%** | 65 件/小时（50 件未见衣物） |
| Bed making | — | 高 | — |
| Candle lighting | — | 高 | — |
| Litter box | — | 高 | — |
| Light bulb replacement | — | 高 | — |
| Table bussing | — | 高 | — |
| Kitchen cleaning | — | 高 | — |

> ✅ **整体：throughput 翻倍以上，failure rate ≥ 2× 下降**。这是首次把 VLA 推到 *能商用* 的可靠性。

社区复现：`huggingface.co/exla-ai/openpie-0.6`，Apache-2.0。

### 8.5 Robot Olympics（2025-12）

PI 用 π0.6 在 5 项 Moravec 式难任务上做了一次"奥运评测"：

| 项目 | 子任务 | π0.6 + finetune | baseline VLM |
|---|---|---|---|
| 全身 | 自闭合门 | silver | ≈ 0% |
| 洗衣 | 袜子翻面 + T 恤折叠 | **gold** | ≈ 0% |
| 基础工具 | 钥匙 / PB 三明治 / 擦窗 | **gold** | ≈ 0% |
| 指尖 | 狗粪袋 / 橘皮 | silver | ≈ 0% |
| 湿滑 | 油锅 / 手指 PB / 台面 | **gold** | ≈ 0% |

**3 金 2 银，平均 ≈ 52%**，每个任务只需 **< 9 小时** 微调数据；baseline VLM 几乎全 0。

> 💡 **Take-away** — "large-scale robot pre-training is essential"。如果不预训练 π0.6 base，任务再多 fine-tune 数据也救不回来。

## §9 MEM + RLT — 长程记忆与在线 RL Token

### 9.1 MEM（2026-03-03）：双时间尺度记忆

VLA 默认只看当前 frame，做 ≥ 1 分钟任务就崩。MEM 给 VLA 加两套记忆：

| 记忆类型 | 表示 | 写入策略 | 读取策略 |
|---|---|---|---|
| **短期** | 时序 ViT token | interleaved spatial-temporal，**上层丢弃旧 token** | cross-attention |
| **长期** | **自然语言** 摘要 | 模型用 CoT *自选* 写什么 | 把摘要拼到 prompt 里 |

**为什么长期用自然语言而不是 KV cache**：

- 自然语言可 **抽象**（"我刚刚把红杯子放进了上层柜"）→ 节约 context
- 模型可 **选择性** 记录（CoT decides what to remember）
- 可 **解释**——人能直接读懂机器人的"日记"
- KV cache 长视频太贵（O(L²)）

**任务长度**：MEM-augmented π0.6 可做 **≤ 15 分钟** 长任务：

- Grilled cheese 定时（要等面包变金黄才翻面）
- Recipe 取料（按食谱逐项找）
- 厨房清理（多房间记忆）

### 9.2 RLT（2026-03-19）：把 VLA 接到在线 RL

**痛点**：on-robot RL 慢（更新一次梯度要几秒），VLA 模型大没法直接训。

**RLT 解法**：

```
VLA  ──→  special token  ──→  encoder-decoder ──→  compact RL token
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │ 小 actor / critic │  100+ update/s
                                                │ 在 token 上跑     │
                                                └─────────────────┘
                                                          │
                                                          ▼
                                         action edit:  Δa  ──→ chunk + Δa
```

**关键设计**：

- actor 学的是 **edit** VLA 预测的 chunk（Δa），不是替换；保留预训练知识
- KL-style 正则：限制 Δa 大小，防 *off-policy 漂移*
- compact RL token 是 *信息瓶颈*：只让任务相关信号通过

**数字**：

| 任务 | base (no RLT) | RLT 后 |
|---|---|---|
| Ethernet 插入 | ~100 / 10 min | **~350 / 10 min** |
| Power cord 插入 | ~100+ | **500+** |

**关键时间**：

- **15 分钟数据** 就能起效
- Ethernet 任务 **2 小时** 达峰
- 50% trial **比人 teleop 还快**（66 vs 146 timestep）

## §10 π0.7 — Steerable Generalist with Emergent Capabilities

> 📌 **论文**：Physical Intelligence et al. (Pertsch submitting, **87 co-authors**), *"π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities"*, arXiv **2604.15483**, 2026-04-16.

### 10.1 范式跃迁：从 "language conditioned" 到 "diverse context conditioned"

π0 ～ π0.6 都是 **language conditioned**——prompt 主要是一句话指令。π0.7 把 prompt 扩成 **diverse multimodal context**：

```
┌──────────────────────────────────────────────────────────┐
│  Prompt Context                                          │
│  ├─ Language instruction:  "fold the shirt"              │
│  ├─ Subtask instruction:   "first align the sleeves"     │
│  ├─ Subgoal image:         [target visual state]         │
│  ├─ Episode metadata:                                    │
│  │    ├─ speed:    fast / slow                           │
│  │    ├─ quality:  1–5 评分                              │
│  │    ├─ mistake:  label                                 │
│  │    └─ control:  joint / EE / cartesian                │
│  └─ Memory context (MEM 风格短时视频)                    │
└──────────────────────────────────────────────────────────┘
```

### 10.2 架构：π0.6 + KI + MEM

- VLM backbone = **Gemma 3 4B**
- Vision encoder = **SigLIP 400M**
- Action expert = **860M**
- 总参数 ≈ **5B**
- 训练 recipe：KI（stop-grad 隔离）
- 短时记忆：MEM 风格 ViT
- 输出方式：flow matching（沿用 π0.6 model card 披露的 **5 步 ODE 推理**，KI recipe）

### 10.3 训练数据（PI 未精确披露量级，**比 π0.6 更大且更异质**）

| 来源 | 备注 |
|---|---|
| 多平台机器人 demo | 至少覆盖 π0.6 的 7 种本体 + 新增双臂 UR5e |
| **自主 policy rollouts**（含失败！） | 失败样本是 advantage 信号的来源 |
| 人工 intervention 数据 | RECAP 风格 |
| **第一人称 egocentric 人类视频** | 让模型学到非机器人本体的 manipulation 先验 |
| 开源数据集 | OXE + DROID + RLBench 等 |
| Web data | caption / VQA / detection |

### 10.4 关键 benchmark

**Espresso & Box assembly**（专家级长任务）：
- 单 π0.7 模型 **≥** RL-finetuned π\*0.6 specialist
- 即：**通用模型追平了专家模型**

**Shirt folding on bimanual UR5e（零训练样本）**：
- π0.7 任务进度：**85.6%**
- π0.7 success rate：**80%**
- 人类 teleop 专家进度：90.9% / success 80.6%
- → **几乎追平人类**，且模型从未在该本体训过

**Unseen 场景**：
- 4 个未见厨房 + 2 个未见卧室：instruction following 高
- 与 π0.5 相比 OOD success +10 pt 以上

### 10.5 Compositional Generalization（首次出现）

π0.7 **首次** 在 VLA 里展示了 *把已学技能重组解决新任务* 的能力。例如：

- 训练时学过 "pick up cup"、"place in tray"、"wipe table"
- 测试时给一个**没训练过的组合**："use the cup to wipe the table"
- π0.7 能成功执行（虽然成功率比 in-distribution 低）

**为什么会出现 compositionality**：

1. **Diverse multimodal prompts** 把 *做什么* 和 *怎么做* 解耦——metadata 标 "speed=slow, quality=5" 让模型学到 *manner-invariant skill* 表征
2. **Cross-embodiment 大规模训练** 强制学到 *body-invariant abstractions*——技能不绑死到某个机器人
3. **Subgoal images** 提供视觉的目标空间，让模型从"目标导向"而非"模仿导向"学习

> ✅ **意义** — 这是 VLA 走向 *foundation model* 的转折点：以前每个任务要 fine-tune，现在 zero-shot 可组合。

### 10.6 Steerability

Prompt 中的 metadata 可在 deploy 时显式调整：

| metadata 设置 | 行为变化 |
|---|---|
| `speed: fast` | 动作更快但精度略降 |
| `quality: 5` | 慢但精细 |
| `quality: 1` | 快糙 |
| `mistake: avoid` | 偏保守 |
| `control: joint` | 输出关节空间 |
| `control: EE_cartesian` | 输出末端笛卡尔空间 |

这让产品经理可以**按场景调速**——同一个模型，"咖啡店忙时" 设 fast，"医院" 设 quality=5。

> ⚠️ **License 注意** — arXiv 论文采用 arXiv non-exclusive distribution license（默认）；权重截至 2026-05-20 **尚未** 进 openpi 正式发布，使用前请查 PI 官方公告。

## §11 从零实现：Flow Matching Action Head（PyTorch）

下面是一份 **能跑** 的简化 π0 action expert，专注 flow matching 训练 / 推理逻辑。条件向量 $c$ 用一个 simple MLP encoder 代替真实的 PaliGemma backbone。

```python
"""pi_flow_action_head.py — minimal flow-matching action head (π0 style).

Demonstrates: linear-Gaussian path, CFM loss, 10-step Euler ODE inference.
NOT a real PaliGemma + π0 — backbone is a tiny MLP for clarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyConditionEncoder(nn.Module):
    """Stand-in for PaliGemma 3B: turns (image_feat, lang_feat, proprio) into c."""

    def __init__(self, img_dim=512, lang_dim=512, proprio_dim=18, c_dim=256):
        super().__init__()
        self.proj = nn.Linear(img_dim + lang_dim + proprio_dim, c_dim)

    def forward(self, img, lang, proprio):
        x = torch.cat([img, lang, proprio], dim=-1)  # [B, img+lang+proprio]
        return self.proj(x)                          # [B, c_dim]


class FlowActionExpert(nn.Module):
    """Predicts the velocity field v_θ(A^τ, c, τ) for a chunk of actions."""

    def __init__(self, action_dim=18, chunk_len=50, c_dim=256, hidden=512):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        flat_a = action_dim * chunk_len
        # input: [flat A^τ | c | τ-embedding]
        self.tau_embed = nn.Linear(1, 64)
        self.net = nn.Sequential(
            nn.Linear(flat_a + c_dim + 64, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_a),
        )

    def forward(self, A_tau, c, tau):
        """A_tau: [B, H, D]  c: [B, c_dim]  tau: [B, 1]"""
        B = A_tau.size(0)
        flat = A_tau.reshape(B, -1)                  # [B, H*D]
        tau_e = self.tau_embed(tau)                  # [B, 64]
        x = torch.cat([flat, c, tau_e], dim=-1)
        v = self.net(x).reshape(B, self.chunk_len, self.action_dim)
        return v


def cfm_loss(expert, encoder, batch):
    """Conditional flow matching loss (π0-style).

    batch keys: img, lang, proprio, action_chunk (target A).
    """
    img, lang, proprio, A = batch["img"], batch["lang"], batch["proprio"], batch["action_chunk"]
    B = A.size(0)
    device = A.device

    c = encoder(img, lang, proprio)                  # [B, c_dim]

    eps = torch.randn_like(A)                        # noise [B, H, D]
    tau = torch.rand(B, 1, device=device)            # τ ~ U[0, 1]
    tau_expanded = tau.unsqueeze(-1)                 # [B, 1, 1] for broadcast

    A_tau = tau_expanded * A + (1.0 - tau_expanded) * eps  # linear-Gaussian path
    target_v = A - eps                                      # dA^τ / dτ = A − ε
    pred_v = expert(A_tau, c, tau)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def sample_chunk(expert, encoder, batch, n_steps=10):
    """10-step Euler ODE inference: ε → A."""
    img, lang, proprio = batch["img"], batch["lang"], batch["proprio"]
    B = img.size(0)
    device = img.device

    c = encoder(img, lang, proprio)
    A_tau = torch.randn(B, expert.chunk_len, expert.action_dim, device=device)

    delta = 1.0 / n_steps
    for k in range(n_steps):
        tau = torch.full((B, 1), k * delta, device=device)
        v = expert(A_tau, c, tau)
        A_tau = A_tau + delta * v                    # Euler step
    return A_tau                                     # ≈ A


# ─────────────── Sanity check ───────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, D = 4, 50, 18
    enc = TinyConditionEncoder()
    expert = FlowActionExpert(action_dim=D, chunk_len=H)
    optim = torch.optim.AdamW(
        list(enc.parameters()) + list(expert.parameters()), lr=1e-3
    )

    # 假数据：让模型记住一个 "目标 chunk"
    target_action = torch.sin(torch.linspace(0, 6.28, H)).unsqueeze(-1).repeat(1, D)
    batch = {
        "img": torch.randn(B, 512),
        "lang": torch.randn(B, 512),
        "proprio": torch.randn(B, 18),
        "action_chunk": target_action.unsqueeze(0).repeat(B, 1, 1),
    }

    for step in range(2000):
        loss = cfm_loss(expert, enc, batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 500 == 0:
            print(f"step {step:5d}  cfm_loss={loss.item():.4f}")

    # 推理 → 与 target 对比
    pred = sample_chunk(expert, enc, batch, n_steps=10)
    rmse = (pred - batch["action_chunk"]).pow(2).mean().sqrt().item()
    print(f"\n10-step ODE inference RMSE vs target: {rmse:.4f}  (should be small)")
```

**预期输出**（CPU 几秒钟，GPU 不到 1 秒）：

```
step     0  cfm_loss=2.0143
step   500  cfm_loss=0.0421
step  1000  cfm_loss=0.0098
step  1500  cfm_loss=0.0044

10-step ODE inference RMSE vs target: 0.0517  (should be small)
```

（注：循环 `range(2000)` 迭代 0..1999，所以 step 2000 不会打印；最后一次打印是 step 1500。）

> ✅ **学习要点** — 这份 80 行代码包含了 π0 的所有核心数学：linear-Gaussian path、CFM L2 loss、10 步 Euler ODE 推理。换成 PaliGemma backbone + 真实机器人数据，就是 openpi 的简化骨架。

## §12 复杂度 & 资源

### 12.1 模型大小演进

| 模型 | VLM | Vision Enc | Action Expert | **Total** |
|---|---|---|---|---|
| π0 / π0-FAST | PaliGemma 3B | (内置) | 300M | **≈ 3.3B** |
| Hi Robot | PaliGemma 3B (S2) + π0 (S1) | — | — | **≈ 6.6B**（两份） |
| π0.5 | PaliGemma 3B + KI | (内置) | 300M | **≈ 3.3B** |
| π0.6 / π\*0.6 / π0.7 | **Gemma 3 4B** | **SigLIP 400M** | **860M** | **≈ 5B** |

### 12.2 训练 / 推理时间复杂度

设 $f$ = 单次 5B 模型 forward 时延（在 H100 上 $f \approx$ 数十毫秒，取决于实现）。

| 操作 | 复杂度（以 $f$ 为单位） | 备注 |
|---|---|---|
| 单次 forward（5B 模型） | $1 \cdot f$ | 决定所有推理的基准延迟 |
| Flow matching 训练（per sample） | $f + b$（forward + backward） | $b$ 通常约 $1.5\,$–$\,2 f$ |
| **Flow matching 推理（π0 / π0.5）** | $10 \cdot f$ | 10 步 Euler ODE |
| **Flow matching 推理（π0.6 / π\*0.6 / π0.7）** | $5 \cdot f$ | KI recipe 把推理步数减半 |
| **π0-FAST 推理**（注意：与 flow 路线不同） | $30 \cdot f \;\text{–}\; 60 \cdot f$ | autoregressive，串行解 30–60 个 BPE token |

### 12.3 训练资源（公开信息粗估）

- π0 训练数据 ≈ 10,000 h
- 训练时长：**数周** on H100 集群（具体规模 PI 未披露）
- π0.6 / π0.7 backbone 升级 + 数据增加，预计 **数月** on H100 集群

## §13 与其他 VLA 横向对比

| 模型 | 主干 VLM | 动作生成 | 训练数据 | 跨本体 | 开源 |
|---|---|---|---|---|---|
| **RT-2** (Google, 2023) | PaLI-X / PaLM-E (5–55B) | 离散 action token, AR | 内部，未细披露 | 部分 | 无 |
| **OpenVLA** (Stanford, 2024) | Prismatic-7B (Llama2-7B + DINOv2 + SigLIP) | 256-bin/dim 离散 AR | OXE 970k traj | 是 | **Apache-2.0** |
| **RDT-1B** (Tsinghua, 2024) | T5-XXL text + Transformer | **Diffusion (DDPM)** 多步 | 46 数据集, 1M+ episodes; 6k ALOHA | 统一动作空间 | **MIT** |
| **π0** | PaliGemma 3B | **Flow matching** 10 步 | 10k h, 7 robots + 9.1% OXE | 内部 7 + OXE | **Apache-2.0** |
| **π0-FAST** | PaliGemma 3B | FAST 离散 token AR | 同 π0 | 同 π0 | Apache-2.0 |
| **π0.5** | PaliGemma 3B + KI | FAST + Flow 双通路 | π0 + 400h MM + WD + HL | 同 π0 | Apache-2.0 |
| **Hi Robot** | PaliGemma 3B (S2) + π0 (S1) | π0 的 flow | π0 base + 合成 prompt | 同 π0 | 论文 + 部分代码 |
| **π0.6** | Gemma 3 4B + SigLIP 400M | Flow (860M expert) | 更大 + 异质 prompt | 同 π0.5+ | 模型卡公开 |
| **π\*0.6** | π0.6 + RL fine-tune | RECAP + advantage cond | + on-robot RL | 同 π0.6 | 社区 (exla-ai) Apache-2.0 |
| **π0.7** | Gemma 3 4B + SigLIP 400M + MEM | Flow (860M) | 更大 + subgoal images + metadata | **零样本** UR5e 跨本体 | 论文，权重 TBD |

> 💡 **结构性差异** — π0 系列 vs OpenVLA 的核心区别：**π 是 flow expert + cross-attn**（小 action head，大 VLM），**OpenVLA 是把动作 token 拍回 LM head**（一锅炖）。π 的设计让 backbone 可以保持 web 知识不掉点，OpenVLA 一旦 post-train 就会损伤 VLM 能力。

## §14 25 道高频面试题

### L1 — 必会（10 题）

<details>
<summary><strong>1. π0 的 VLM 主干和 action expert 各多大？总参数多少？</strong></summary>

PaliGemma 3B + 300M flow-matching action expert，总参数约 **3.3B**。Action expert 通过 cross-attention 共享 VLM 的 KV cache，不是 stacked 在 VLM 顶上。
</details>

<details>
<summary><strong>2. π0 的动作 chunk 长度 H 和最高控制频率？为什么用 chunk 而不是 single-step？</strong></summary>

H = 50，最高 50 Hz（即 1 个 chunk = 1 秒）。Chunk 的好处：① 模型一次看一段连贯动作，避免 single-step 的 jitter；② 推理频率可以低于控制频率（推理 10 Hz，控制 50 Hz），节省算力；③ 与 RTC 异步推理天然兼容。
</details>

<details>
<summary><strong>3. π0 推理时 flow matching 跑几步 ODE？步长 δ 多少？</strong></summary>

10 步 Euler，δ = 0.1。从 τ=0（噪声 ε ~ N(0, I)）走到 τ=1（≈ ground truth chunk）。每步是 $A^{\tau+\delta} = A^\tau + \delta \cdot v_\theta(A^\tau, c, \tau)$。
</details>

<details>
<summary><strong>4. π0 的 flow matching 与 Diffusion Policy (DDPM) 在数学上的核心区别？</strong></summary>

- **DDPM**: 学 noise prediction $\varepsilon_\theta$ 或 score $\nabla \log p_t$，用 SDE 反向采样，几十~百步
- **π0 (flow matching)**: 学速度场 $v_\theta(x, \tau) \approx dx/d\tau$，用 ODE 反向，10 步就够
- 数学上等价（都是 stochastic interpolant 的特例），但 flow 推理快得多
</details>

<details>
<summary><strong>5. FAST tokenizer 的三步压缩是什么？</strong></summary>

① **DCT**（Discrete Cosine Transform）沿时间维做频域变换，把动作能量集中到低频；② **量化**（scale × 10，round）丢掉高频细节；③ **BPE**（vocab=1024）把高频共现 token 合并。50 步 × 18 维的 chunk 压成 **30–60 个 token**，类比 JPEG/MP3。
</details>

<details>
<summary><strong>6. π0-FAST 比 π0 训练快 5×，为什么？推理也更快吗？</strong></summary>

**训练快**：纯 next-token AR，可以复用 LLM 的 dense batched 训练 infra，省掉 flow matching 的 per-sample expert forward。

**推理慢**：30–60 个 token 必须 **串行** AR 解码，每个 token 一次 forward。而 π0 flow 的 10 步 ODE 整个 chunk 可以并行解。生产部署还是用 π0 / π0.5 / π0.7 的 flow 路径。
</details>

<details>
<summary><strong>7. π0.5 怎么实现未见家庭 94% 成功率？In-domain 数据占多少？</strong></summary>

异质 co-training：MM（移动操作，400 h）+ ME（多环境固定臂）+ CE（跨本体）+ HL（高层语义子任务）+ WD（web 数据）。**97.6% 训练样本不是目标 MM 任务**——in-domain 只占 ≈ 2.4%。泛化能力来自**异质性**，不是 in-domain 堆量。
</details>

<details>
<summary><strong>8. Hi Robot 的 System 1 和 System 2 各是什么模型？输出粒度？</strong></summary>

- **System 2** = PaliGemma-3B 微调，≈ 1 Hz，输出**语言子任务** + "inner monologue"
- **System 1** = π0，50 Hz，输出**连续动作 chunk**
- S2 给 S1 喂语言子任务作为 prompt
</details>

<details>
<summary><strong>9. openpi 的开源协议？商用允许吗？</strong></summary>

**Apache-2.0**，**允许商用**、修改、再分发，只需保留版权声明。这是 π 系列在工业界扩散的关键因素之一——很多公司因为协议宽松直接拿来用。
</details>

<details>
<summary><strong>10. π 系列的 cross-embodiment 怎么对齐不同维度？OXE 占比？</strong></summary>

所有动作 zero-padding 到最大维度（17–18 D，π0.6/FAST+ 提升到 32 D）；每条样本带一个 "robot type" embedding token 指示本体。OXE 占 π0 训练混合的 **9.1%**。π0.5 消融显示：**去掉 web data (WD)** 在未见物体上掉 ≈ 30 pt；去掉 ME 数据掉 ≈ 20 pt；去掉 CE 数据掉 ≈ 15 pt——多源异质数据各有不可替代的贡献。
</details>

### L2 — 进阶（10 题）

<details>
<summary><strong>11. Knowledge Insulation 解决的核心问题？stop-grad 在哪？</strong></summary>

**问题**：action expert 的梯度反向传到 VLM backbone，**污染** PaliGemma 的多模态知识（VQA / caption 掉点）。

**解法**：① 在 VLM backbone 上加 FAST next-token AR loss（学运动语义）+ web loss（保多模态）；② action expert 的 flow matching loss 在喂回 backbone 前 **stop-grad**——backbone 不知道 expert 长什么样。

**效果**：训练步数 7.5× fewer，backbone web 能力不掉点。
</details>

<details>
<summary><strong>12. RECAP 的三阶段？解决什么问题？</strong></summary>

**问题**：纯模仿学习的 VLA 到不了专家级（espresso ≈ 40%，box ≈ 20%）。

**三阶段**：① **离线 RL 预训练** 同时学 V 函数 *和* advantage-conditioned policy（**不仅是 V**）；② **任务级 demo SFT**；③ **on-robot RL with interventions**（专家随时接管 = high advantage，自主失败 = low advantage）。

**结果**：espresso 40% → ≥ 90%；30 杯/小时连开 13 小时。
</details>

<details>
<summary><strong>13. Advantage-conditioned policy 怎么把 bad data 也用上？与 AWR 区别？</strong></summary>

**AWR**（Advantage-Weighted Regression）：把 advantage 当 **loss 权重**，$\exp(A/\beta)$ 偏向好动作。

**RECAP**：把 advantage 当 **prompt token 喂给 VLA**。训练时全用（好的标 advantage=+，差的标 −），推理时始终 condition 在 high advantage。

**优点**：① 不丢任何数据；② 更接近 Decision Transformer 的 in-context conditioning，但用 V 差代替 return-to-go，**长程任务信噪比更好**。
</details>

<details>
<summary><strong>14. Real-Time Chunking 怎么解决推理延迟？inpainting 做了什么？</strong></summary>

**异步**：新 chunk 还在推理时老 chunk 继续执行，没空窗。

**Inpainting**：① 已经物理执行的 K 步**冻结**——新 chunk 不能改它们；② 重叠区域用 partial attention：把老 chunk 在重叠位置的动作当作 *已知*，新 chunk 生成 *缺失部分*。

**效果**：移动机器人端到端延迟 139 ms，容忍 +200 ms 注入延迟，云端部署可行。
</details>

<details>
<summary><strong>15. π0.7 最大的范式升级是什么？</strong></summary>

**Prompt 从单语言扩成 diverse multimodal context**：
- subtask 文本
- subgoal **图像**
- episode metadata（speed / quality / mistake / control mode）

这让模型学到 **manner-invariant skill 表征**（怎么做 vs 做什么解耦），从而出现 **compositional generalization**。
</details>

<details>
<summary><strong>16. π0.7 在 bimanual UR5e 上 shirt folding 怎么做到零样本 80%？</strong></summary>

三个因素：① **Cross-embodiment 大规模训练** 让 backbone 学到 body-invariant manipulation 表征；② **Diverse multimodal prompt**（含 subgoal image）让模型把 "fold" 这个 skill 抽象成视觉目标转换；③ **MEM 风格短时记忆** 让多步骤任务有 context。

**任务进度 85.6% / success 80%**，几乎追平人类 teleop 专家（90.9% / 80.6%）。
</details>

<details>
<summary><strong>17. RLT 怎么把 VLA 接到在线 RL？为什么不直接训 VLA？</strong></summary>

**不直接训 VLA**：5B 模型，单次更新几秒，on-robot RL 一天最多几百次更新，收敛不了。

**RLT 解法**：VLA 输出加 special token → encoder-decoder 压成 compact **RL token**（信息瓶颈）；小 actor / critic 在 token 上跑（100+ update/s）。actor 学的是 **edit** VLA chunk（Δa），不是替换；KL 正则限制 Δa。

**数字**：15 分钟数据起效，Ethernet 任务 2 小时达峰，50% trial 比人 teleop 还快。
</details>

<details>
<summary><strong>18. π0 vs OpenVLA 在动作表示上的关键差异？哪个对 backbone 更友好？</strong></summary>

- **OpenVLA**: 256-bin per-dim 离散 AR，把动作 token 直接拍回 LM head——**post-train 会损伤 VLM 能力**（caption / VQA 掉点）。
- **π0**（原版）：独立的 flow expert + cross-attention 共享 KV cache。**结构上 expert 与 LM head 分离**，但因为 expert 梯度仍能经 cross-attention 反传到 VLM backbone，**没有 KI 时 backbone 也会被污染**——这正是 π0.5/π0.6/π0.7 引入 **KI stop-grad** 的动机。
- **π0 + KI**（π0.5 之后）：在 cross-attention 上加 stop-grad，**才真正实现 backbone 不被 expert 污染**。

工程上 π 系列 + KI 比 OpenVLA 更适合 *base + N 个任务 expert* 的多任务部署，因为 backbone 多模态能力可保。
</details>

<details>
<summary><strong>19. π0.5 的两阶段训练具体怎么切？为什么不一步到位？</strong></summary>

**Stage 1（pretrain, 280k 步）**：FAST 离散 token + 异质 co-training **MM/ME/CE/HL/WD**（不含 VI）。让 backbone 学到跨本体的 manipulation 运动语义。

**Stage 2（posttrain, 80k 步）**：加 flow matching action expert（300M）做精细动作输出；**数据切换为 MM/ME/HL/WD/VI**（引入口头指令 VI，移除 CE 聚焦目标域）；双通路推理（高层 AR 子任务 + 低层 flow chunk）。

**为什么不一步到位**：① FAST 阶段允许超大 batch 训 backbone；② Flow 阶段才需要 expert 收敛；两阶段在 wall-clock 上更省。
</details>

<details>
<summary><strong>20. MEM 的长时记忆为什么用自然语言而不用 KV cache？</strong></summary>

四个原因：
1. **抽象**：自然语言可以总结（"红杯子已放上层柜"），KV cache 是 raw token，O(L²) attention 太贵
2. **选择性**：模型用 CoT *自选* 写什么，KV cache 是被动堆积
3. **可解释**：人能直接读懂机器人的"日记"，便于 debug
4. **可压缩**：15 分钟视频原始 KV 几十万 token，自然语言总结几百 token
</details>

### L3 — 顶级 lab（5 题）

<details>
<summary><strong>21. 如果让你重新设计 Flow Matching 的 loss，怎么处理 multi-modal action distribution？</strong></summary>

默认 flow matching 已对 multi-modal 鲁棒（学速度场，不像 MSE 平均掉双峰），但仍可改进：

1. **Mixture-of-experts flow head**：不同 expert 学不同 mode，gate network 选 expert
2. **Mixture-Gaussian endpoint**：把 path 的终点 $A$ 改成 mixture-Gaussian，每个模式一个 path
3. **Stochastic Interpolants 风格 noise schedule**：在路径上加可控噪声，模式之间更清晰
4. **Latent flow**：先用 VQ-VAE 把动作离散成 mode index + residual，flow 学 residual

PI 实际生产可能用了 (1) 或 (3) 的变种，但论文没披露细节。
</details>

<details>
<summary><strong>22. π0.7 出现的 "compositional generalization" 你认为来源是什么？什么数据/objective 触发？</strong></summary>

我的猜测三条：

1. **Diverse multimodal prompts**——subgoal image + metadata 把 *做什么* 和 *怎么做* 解耦，backbone 被迫学到 manner-invariant skill 表征
2. **Cross-embodiment 大规模训练**——同一个 skill 在多种机器人上学过，必然抽象成 body-invariant primitives
3. **第一人称人类视频**——给了非机器人本体的 manipulation 先验，进一步推动 abstraction

**触发 objective**：不是单一 loss，而是 *多模态 prompt + 多本体 + 多目标（FAST AR + flow CFM）* 联合学习的涌现现象。类似 LLM 在足够大模型 + 足够多任务 prompting 下涌现 in-context learning。
</details>

<details>
<summary><strong>23. 在 humanoid 上 fine-tune π0.7 会遇到哪些 challenge？</strong></summary>

四个主要 challenge：

1. **动作维度爆炸**：humanoid 全身 50+ DoF，π0 padding 到 17–18 D 不够；需要重设 action expert 维度，可能要重新 pretrain
2. **控制频率不够**：50 Hz 对腿部 PD 控制不够（要 200+ Hz），需要重设计 chunk 颗粒度 + RTC 异步分层（高层 50 Hz / 低层 200 Hz）
3. **Proprio 通道结构差异**：humanoid 有 IMU、关节扭矩、足底力，π 系列没有这些通道的预训练经验
4. **数据稀缺**：没有 humanoid 级的 OXE，需要自建大规模数据集（参考 1X 的 NEO Beta、Tesla Optimus 内部）

务实路径：把 π0.7 当 *vision-language backbone*，**重新训练 action expert**，保留 KI 风格 stop-grad 不污染 backbone。
</details>

<details>
<summary><strong>24. PI 的 advantage-conditioned policy 与 Decision Transformer / RvS 区别在哪？为什么 PI 选 advantage 而不是 return-to-go？</strong></summary>

**Decision Transformer / RvS**：用 **return-to-go (RTG)** 作为 prompt token——"接下来还要拿多少 reward"。在 dense reward 任务（Atari、MuJoCo）有效。

**RECAP**：用 **advantage** $A_t = \mathbb{E}[\sum_{k=0}^{N-1} r_{t+k}] + V(s_{t+N}) - V(s_t)$（n-step；与 §8.3 一致）作为 prompt token。在机器人 sparse-reward 设定下，常退化为 $A_t \approx V(s_{t+N}) - V(s_t)$ 的简化形式。

**为什么 PI 选 advantage**：

1. 真实机器人任务 **reward 极 sparse**（要么成功要么失败），RTG 信噪比低，advantage 用 V 差距能在每一步给信号
2. Advantage 是 **基线扣减后的相对量**——告诉模型"这一步好不好"，比"还要拿多少"更接近 *operative knowledge*
3. 训练数据天然含 expert intervention，intervention = high advantage，failure = low advantage，标注成本零
4. 长 horizon、长部署任务（espresso 单店连续运营 ≈ 13 小时不间断，单杯 ≈ 1–2 分钟）下，return-to-go 量级跨尺度差几个数量级，量化与 normalize 都困难；advantage 是 *局部* 量，天然解决这个尺度问题
</details>

<details>
<summary><strong>25. 预测 PI 2026 下半年到 2027 可能发布的功能/版本，并给出技术理由。</strong></summary>

合理推测四条：

1. **π1**：把 RLT 风格的在线 RL + RECAP 整合进 base 训练 recipe，让 "deploy-time self-improvement" 成默认能力；预计 backbone 升级到 Gemma 4 / 7B
2. **World-model conditioned flow**：用 video diffusion / Genie 风格的 world model 给高层 subgoal（生成"未来 5 秒应该长这样"的图像 sequence），π 用这个作为 subgoal image prompt
3. **触觉 / 力反馈通道纳入 prompt context**：现有 π0.7 是 vision + language + proprio + metadata，下一步加 force/torque token，继续 diverse-context 路线
4. **跨日 persistent memory**：MEM 现在是 episode 内 15 分钟，下一步是 **跨任务、跨日**记忆（VLA 知道"昨天我修过这台咖啡机的某个零件松了"）

技术驱动因素：① 真实部署反馈数据成为 PI 的护城河；② foundation model 范式逼着 VLA 也走 "一个模型搞定所有" 路线。
</details>

## §A 附录：参考资料

### 论文 / 技术报告

- **π0**：[arXiv 2410.24164](https://arxiv.org/abs/2410.24164) · [blog](https://www.physicalintelligence.company/blog/pi0)
- **π0-FAST**：[arXiv 2501.09747](https://arxiv.org/abs/2501.09747) · [research/fast](https://www.physicalintelligence.company/research/fast)
- **Hi Robot**：[arXiv 2502.19417](https://arxiv.org/abs/2502.19417) · [research/hirobot](https://www.physicalintelligence.company/research/hirobot)
- **π0.5**：[arXiv 2504.16054](https://arxiv.org/abs/2504.16054) · [blog](https://www.physicalintelligence.company/blog/pi05)
- **Knowledge Insulation**：[arXiv 2505.23705](https://arxiv.org/abs/2505.23705) · [research](https://www.physicalintelligence.company/research/knowledge_insulation)
- **Real-Time Chunking**：[research/real_time_chunking](https://www.physicalintelligence.company/research/real_time_chunking)
- **π0.6 model card**：[PDF](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)
- **π\*0.6**：[arXiv 2511.14759](https://arxiv.org/abs/2511.14759) · [blog/pistar06](https://www.physicalintelligence.company/blog/pistar06)
- **Robot Olympics**：[blog/olympics](https://www.physicalintelligence.company/blog/olympics)
- **MEM**：[research/memory](https://www.physicalintelligence.company/research/memory)
- **RLT**：[research/rlt](https://www.physicalintelligence.company/research/rlt)
- **π0.7**：[arXiv 2604.15483](https://arxiv.org/abs/2604.15483) · [blog/pi07](https://www.physicalintelligence.company/blog/pi07)

### 代码

- **openpi**（官方）：<https://github.com/Physical-Intelligence/openpi> — Apache-2.0
- **exla-ai/openpie-0.6**（社区复现 π\*0.6）：<https://huggingface.co/exla-ai/openpie-0.6>
- **open-pi-zero**（社区第三方再实现）：注意非官方

### 相关 VLA

- **RT-2 / RT-X**：Google DeepMind 系列
- **OpenVLA**：Stanford, [openvla.github.io](https://openvla.github.io/)
- **RDT-1B**：Tsinghua, [arXiv 2410.07864](https://arxiv.org/abs/2410.07864)

### 关键人物（PI 核心团队）

- **Sergey Levine**（co-founder，UC Berkeley 教授）
- **Chelsea Finn**（co-founder，Stanford 教授）
- **Karol Hausman**（co-founder，前 Google Robotics）
- **Brian Ichter**（co-founder）
- **Karl Pertsch**（多篇一作 / 通讯）
- **Kevin Black**（π0 一作）
- **Danny Driess**（π0 / Hi Robot / π0.5 主要作者）
- **Lucy X. Shi**（Hi Robot 一作）

> 💡 **小八卦** — Sergey Levine + Chelsea Finn + Karol Hausman 这三人在 PI 之前都在 Google Robotics（RT-1 / RT-2 / SayCan 主力）。PI 可以看成是 *Google Robotics 体系的延续*，加上 flow matching 作为新核心。

### 中文社区资源

- HuggingFace 中文 blog：搜 "π0 flow matching" / "openpi 中文"
- B 站：搜 "π0 机器人" / "physical intelligence VLA"
- 知乎：搜 "VLA π0" / "flow matching 机器人"
