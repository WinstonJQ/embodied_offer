# 具身智能岗位 · 手撕代码题调研

> **调研时间**: 2026-05-22
> **用户研究方向**: VLN (Vision-Language Navigation) / 零样本 ObjectNav / Embodied VLM
> **用途**: 后续可能融入 `embodied-interview-qa` 现有 7 卷题库（vol-1..7 共 320+ 题，全为"概念+答案"型，**完全无手撕代码内容**），或新开 vol-8 手撕专题。本笔记**只调研，不改题库**。
> **数据来源**: 牛客网、知乎、CSDN、博客园、GitHub、阿里云开发者、华为云、Glassdoor、InterviewQuery、Medium、Reddit。详细来源见 §5。

---

## 0. TL;DR（一页速览）

- **总调研到 105 题（手撕代码主表 65 题 + LeetCode 高频原题 30 题 + 系统设计 10 题）**。
- **三个最高频考察方向**:
  1. **Transformer 家族**（self-attention / MHA / RoPE / LayerNorm / mask）——具身岗几乎 100% 必考至少 1 道，因为 VLA / VLM / 导航 policy 普遍 Transformer-based
  2. **RL 核心公式**（PPO clipped loss / GAE / Bellman / Q-learning）——RL/VLA 岗、运控岗的"分水岭题"
  3. **LeetCode Hot 100 中频段题**（无重复字符最长子串、三数之和、反转链表、二叉树层序、最小栈、接雨水、滑动窗口最大值）——所有公司一面/二面均考，无差别
- **给用户的插入建议（高层）**: **推荐 Option C 混合方案**。手撕代码按主题贴入现有卷的"项目拷打"段（vol-1/2/3/7 各加 5-10 题），LeetCode + 系统设计开新卷 vol-8。理由见 §4。
- **信心评估**:
  - 信心高: Transformer 手撕、RL 公式手撕、LeetCode 高频题（题源 ≥10）
  - 信心中: Diffusion Policy / ACT / VLA 专项手撕（题源 ~5，但具身岗权威面经支持）
  - 信心低: 具身公司（Figure / 1X / 银河通用 / 星海图）的精确原题——这些公司面经绝对量小，多为定性描述

---

## 1. 手撕代码题主表

### 1.1 Transformer 家族（频次 ≥3，共 14 题）

#### T1 · 手撕 scaled dot-product attention 🔥×15 · L2
**考察点**: Attention(Q,K,V) = softmax(QK^T / √d_k) V 公式 + PyTorch/NumPy 实现 + 为何要 scale
**关键实现要点**:
- Q@K.transpose(-2,-1) / sqrt(d_k)
- softmax 沿最后一维 (`dim=-1`)
- mask 加在 softmax 之前（加性 -inf）
- 最后 @V 得到 [B, ..., L, d_v]
**易错**: 忘记 transpose 维度；忘记 √d_k（变体: 写成 d_k 平方根，常忘开根号）；mask 用乘性 0/1 而非加性 -inf 导致 softmax 数值不准
**为何考**: VLA 模型最底层组件，Diffusion Policy / OpenVLA / RT-2 内部全是它
**来源**: [MLM tutorial](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/) · [TensorTactics interview Q](https://www.tensortactics.com/machine-learning-interview-questions/scaled-dot-product-attention) · [CSDN 手撕](https://blog.csdn.net/qq_44949041/article/details/128087174) · [华为云手撕](https://bbs.huaweicloud.com/blogs/475209) · [zhihu 手撕 transformer](https://zhuanlan.zhihu.com/p/16064503712)

#### T2 · 手撕 multi-head attention 🔥×12 · L2
**考察点**: 拆头/合头的 view/transpose/contiguous；为什么多头比单头好
**关键实现要点**:
- `x.view(B, L, h, d_k).transpose(1,2)` → `[B, h, L, d_k]`
- 各头独立做 SDPA
- 合并: `transpose(1,2).contiguous().view(B, L, h*d_k)` → 最后过 out_proj
- 4 个 Linear: W_q / W_k / W_v / W_o
**易错**:
- `embed_dim % num_heads != 0` 导致维度错
- 忘 `contiguous()` 后 `view()` 报 stride 错
- mask 广播形状（应为 `[B, 1, 1, L]` 或 `[B, h, L, L]`）
**为何考**: VLN/VLA/VLM 的 cross-modal attention 全靠它
**来源**: [zhimi 手写](https://zhimi.vercel.app/multi_head_20240311_zh-cn.html) · [CSDN 多头NumPy](https://blog.csdn.net/weixin_43114209/article/details/150490531) · [zhihu 手撕MHA](https://zhuanlan.zhihu.com/p/13166045093) · [楚千羽手撕(一)](https://www.cnblogs.com/chuqianyu/p/18048501)

#### T3 · 手撕 sinusoidal positional encoding 🔥×6 · L1
**考察点**: PE(pos, 2i) = sin(pos/10000^(2i/d))；为什么不用 learnable embedding
**关键实现要点**:
- 用 broadcast 一次性生成 [L, d] 矩阵
- 偶数列 sin / 奇数列 cos
- 注册为 buffer，不参与梯度
**易错**: 维度 d 是奇数时切片错；忘记加到 token embedding 后再过 dropout
**来源**: [transformers.run §3](https://transformers.run/c1/attention/) · [CSDN 旋转编码](https://blog.csdn.net/BIT_666/article/details/133696553)

#### T4 · 手撕 RoPE 旋转位置编码 🔥×11 · L2
**考察点**: 复数旋转视角；为什么 RoPE 能编码相对位置；与 absolute PE 区别
**关键实现要点**:
- 预算 freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
- 每对 (q_2i, q_2i+1) 看作复数 q_2i + i·q_2i+1，乘 e^(i·pos·θ)
- 写法: q_rot = q * cos + rotate_half(q) * sin
- 只对 Q/K 应用，不对 V 应用
**易错**:
- rotate_half 写错（应 `torch.cat([-x2, x1])`）
- d_model 是奇数无法成对
- 序列超过预计算长度需要外推
**为何考**: LLaMA/Qwen/π0 都用 RoPE，VLA 大模型几乎必用
**来源**: [CSDN RoPE 代码](https://blog.csdn.net/BIT_666/article/details/133696553) · [博客园 RoPE](https://www.cnblogs.com/mudou/p/18307600) · [labml.ai RoPE](https://nn.labml.ai/zh/transformers/rope/index.html) · [zhihu 20行 RoPE](https://zhuanlan.zhihu.com/p/684666015) · [苏剑林 RoPE](https://kexue.fm/archives/10862)

#### T5 · 手撕 LayerNorm 🔥×10 · L1
**考察点**: 沿哪个维度 norm；γ / β 的形状；为什么 Transformer 用 LN 不用 BN
**关键实现要点**:
- 沿最后一维（特征维）算 mean/var
- `(x - mean) / sqrt(var + eps) * gamma + beta`
- `gamma, beta` 的形状是 `normalized_shape`
**易错**: 把 var 写成 std；eps 忘加；与 BN 维度混淆（BN 沿 batch+spatial，LN 沿 feature）
**来源**: [zhihu LN vs BN vs RMS](https://zhuanlan.zhihu.com/p/694909672) · [hwcoder 神经网络篇](https://hwcoder.top/Manual-Coding-2) · [CSDN 详解三种](https://blog.csdn.net/wxc971231/article/details/139925707)

#### T6 · 手撕 RMSNorm 🔥×8 · L1
**考察点**: 与 LN 区别（去掉 mean）；为何 LLaMA/Mistral/π0 都用 RMSNorm
**关键实现要点**:
- `x / sqrt(mean(x^2) + eps) * gamma`
- 没有 β（不 re-center）
- 比 LN 省 7%-64% 运算
**易错**: 记错为还要减 mean
**来源**: [阿里云 BN/LN/RMS](https://developer.aliyun.com/article/1645185) · [CSDN RMSNorm 详解](https://blog.csdn.net/m0_37586991/article/details/149251473)

#### T7 · 手撕 BatchNorm（推理 vs 训练区别）🔥×7 · L2
**考察点**: train 用 mini-batch 统计、eval 用 running_mean/running_var；BN 在 batch=1 失效
**关键实现要点**:
- train: 用当前 batch mean/var + 滑动平均更新 running_*
- eval: 用 running_mean/var
- gamma/beta 是 [C] 形状，沿 C 维 norm
**易错**: 忘 `model.eval()` 切换；momentum 方向（PyTorch 是 0.1，TF 是 0.99）
**来源**: [zhihu BN/LN 实现](https://zhuanlan.zhihu.com/p/172185048) · [zhihu BN/LN/IN/GN](https://zhuanlan.zhihu.com/p/630720228) · [snailcoder 博客](https://snailcoder.github.io/2024/05/01/batchnorm-and-layernorm.html)

#### T8 · 手撕 causal mask / padding mask 🔥×8 · L2
**考察点**: 区分两种 mask 的目的（causal: 不看未来 / padding: 忽略 pad）；如何组合
**关键实现要点**:
- causal: `torch.triu(torch.ones(L,L), diagonal=1).bool()` 取 mask
- 或更优: `torch.arange(L)[:,None] < torch.arange(L)[None,:]`
- padding: `attention_mask` 形如 `[B, L]`, 扩展到 `[B, 1, 1, L]`
- 在 logits 上 `masked_fill(mask, -inf)`
**易错**:
- mask 形状广播失败
- causal mask 方向反了（应屏蔽未来不是过去）
- 用乘性 0/1 mask（错；softmax 后非 0 项不是 0）
**来源**: [zhihu LLM手撕(2)](https://zhuanlan.zhihu.com/p/2022092403721479678) · [CSDN attention mask](https://blog.csdn.net/weixin_43889416/article/details/115301178) · [博客园 Mask](https://www.cnblogs.com/wevolf/p/12484972.html)

#### T9 · 手撕 KV cache（增量解码）🔥×7 · L2
**考察点**: 为何 prefill 后只算新 token 的 q；K/V 怎么 concat；显存瓶颈
**关键实现要点**:
- decode 步: q 形状 `[B, 1, d]`，K_cache/V_cache 形状 `[B, L_past, d]`
- `K_new = cat([K_cache, k_t], dim=1)`，存回
- 计算 `attn = softmax(q @ K_new.T / √d) @ V_new` 只输出最后一步
- 显存: `2 * num_layer * num_head * L * d` per request
**易错**: 写成每步重算全部历史 K/V（变 O(N²) → 不算 cache）；忘了 RoPE 也要对 K_cache 更新位置
**为何考**: 推理服务设计常考；Qwen2-7B 4K 序列 KV cache 1.6GB
**来源**: [zhihu 面试官KV cache](https://zhuanlan.zhihu.com/p/1965823665364014863) · [zhihu 看图学KV](https://zhuanlan.zhihu.com/p/662498827) · [华为面试每日](https://zhuanlan.zhihu.com/p/18748221598)

#### T10 · 手撕 cross-attention 🔥×4 · L2
**考察点**: 与 self-attn 的唯一区别（K/V 来自另一个序列）；何时用（图像+文本融合 / encoder-decoder）
**关键实现要点**:
- Q from 序列 A, K/V from 序列 B
- `attn = softmax(Q_A @ K_B.T / √d) @ V_B`
- 输出形状跟 A 一样长
**易错**: 把 Q/K/V 都从一个序列拿；mask 维度跟 self-attn 不一样（应是 `[L_A, L_B]`）
**为何考**: VLN/VLA 视觉-语言融合的核心
**来源**: [zhihu Attention 面经](https://zhuanlan.zhihu.com/p/379033238) · [datawhalechina](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%8D%81%E7%AB%A0/Transformer%20%E8%A7%A3%E8%AF%BB.html)

#### T11 · 手撕 ViT patch embedding 🔥×6 · L1
**考察点**: 用 Conv2d 一步切+映射的技巧；为什么 patch 大小决定序列长度
**关键实现要点**:
- `nn.Conv2d(in_c, dim, kernel_size=patch, stride=patch)`
- 输出 `[B, dim, H/p, W/p]` → flatten(2).transpose(1,2) → `[B, N, dim]`，N=(H/p)*(W/p)
- 加 CLS token + position embedding
**易错**: H/W 不能被 patch 整除；忘加 CLS token；PE 长度跟 patch 数不一致
**为何考**: DINOv2 / SigLIP 这些 VLA backbone 都是 ViT
**来源**: [CSDN ViT原理](https://blog.csdn.net/fulva/article/details/121045938) · [datawhalechina ViT](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%8D%81%E7%AB%A0/ViT%E8%A7%A3%E8%AF%BB.html) · [CSDN patch embedding](https://blog.csdn.net/qq_42740834/article/details/124994344)

#### T12 · 手撕 FlashAttention 思想（不要求完整 CUDA）🔥×4 · L3
**考察点**: tiling / online softmax / IO-aware；为什么 FLOPs 没变但快了几倍
**关键实现要点**（描述，不让真撕 CUDA）:
- 分块 (tile size 取决于 SRAM 容量 M)
- online softmax: 维护 running max + running sum
- kernel fusion 减少 HBM 读写
- 复杂度: 不是降 FLOPs 而是降 HBM 访问 O(N²) → O(N²·d/M)
**易错**: 把 FlashAttention 当成"近似 attention"（错，是精确的）；说降低了 FLOPs（错，是降低了 IO）
**来源**: [zhihu Flash 全解析](https://zhuanlan.zhihu.com/p/1953761827025584899) · [zhihu Flash 简单解读](https://zhuanlan.zhihu.com/p/655448380) · [zhihu Flash 含代码](https://zhuanlan.zhihu.com/p/676655352)

#### T13 · 手撕完整 Transformer Block（MHA+FFN+残差+LN）🔥×6 · L2
**考察点**: pre-norm vs post-norm；FFN 用 GELU 还是 SwiGLU；残差顺序
**关键实现要点**:
- pre-norm: `x = x + MHA(LN(x))`, `x = x + FFN(LN(x))`
- post-norm（原论文）: `x = LN(x + MHA(x))`, `x = LN(x + FFN(x))`
- FFN: `Linear(d, 4d) → GELU → Linear(4d, d)` 或 SwiGLU
**易错**: 残差顺序反了；FFN 隐藏维度（一般 4×d）；忘 dropout
**来源**: [腾讯云一晚上 transformer](https://cloud.tencent.com/developer/article/1890943) · [arthur chiao 600行](https://arthurchiao.art/blog/transformers-from-scratch-zh/)

#### T14 · 手撕 BPE tokenizer 🔥×3 · L2
**考察点**: 字节对编码原理；merge rule；和 WordPiece 区别（频率 vs 似然）
**关键实现要点**:
- 初始化: 字符级词表
- 统计相邻字符对频率
- 合并最高频对 → 新 token
- 重复直到词表达到目标大小
**易错**: 没处理空格/边界；merge 顺序在 decode 时重要
**来源**: [zhihu BPE/WordPiece/Unigram](https://zhuanlan.zhihu.com/p/620508648) · [CSDN BPE 实践](https://blog.csdn.net/2401_85325397/article/details/139472993)

---

### 1.2 强化学习核心公式（频次 ≥3，共 11 题）

#### R1 · 手撕 Bellman equation（V/Q/advantage 三个版本）🔥×9 · L2
**考察点**: V(s) = E[r + γV(s')], Q(s,a) = E[r + γ max_a' Q(s',a')], A(s,a) = Q-V；区分 expectation/optimality 两种形式
**关键实现要点**:
- V^π(s) = Σ π(a|s) Σ p(s',r|s,a)[r + γV^π(s')]
- 最优 V*: 把外层 Σπ 换成 max_a
- Q-learning 用 max（off-policy），SARSA 不用 max（on-policy）
**易错**: 把 expectation 跟 optimality 搞混；忘 γ 的折扣意义
**来源**: [hrl.boyuai 动手学RL](https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B/) · [CSDN Q-learning](https://blog.csdn.net/qq_30615903/article/details/80739243) · [zhihu RL教程05](https://zhuanlan.zhihu.com/p/18067879673)

#### R2 · 手撕 Q-learning update（TD(0)）🔥×8 · L1
**考察点**: 更新规则；α、γ 含义；为什么是 off-policy
**关键实现要点**:
- `Q[s,a] += α * (r + γ * max(Q[s']) - Q[s,a])`
- ε-greedy 探索
- 表格法 vs 函数逼近 (DQN)
**易错**: 收敛慢与 α 太大；终止状态 V(s_T)=0 没置零；连续动作无法 max → 应改 DDPG/SAC
**来源**: [CSDN Q-learning](https://blog.csdn.net/qq_30615903/article/details/80739243) · [博客园 Q-learning Python](https://www.cnblogs.com/hhh5460/p/10134018.html) · [CSDN GridWorld 实现](https://blog.csdn.net/2501_91624122/article/details/147916483) · [CSDN 完整指南](https://blog.csdn.net/qq_36603091/article/details/147522177)

#### R3 · 手撕 DQN（含 target net + replay）🔥×6 · L2
**考察点**: 经验回放打破时序相关性；target network 稳定 bootstrap
**关键实现要点**:
- 主网络 Q(s,a;θ)，target Q(s,a;θ⁻) 每 N 步同步
- replay buffer 存 (s,a,r,s',done)
- loss = MSE(Q(s,a) - (r + γ * max Q_target(s', a') * (1-done)))
**易错**: target net 没 detach 导致梯度回传；done=True 时还加 γV(s')；replay 容量太小
**来源**: [hrl.boyuai DQN改进](https://hrl.boyuai.com/chapter/2/dqn%E6%94%B9%E8%BF%9B%E7%AE%97%E6%B3%95/) · [zhihu D3QN](https://zhuanlan.zhihu.com/p/490163865)

#### R4 · 手撕 PPO clipped objective 🔥×14 · L2/L3
**考察点**: clip ratio 1±ε；min(r·A, clip(r,1-ε,1+ε)·A)；为什么用 min 而不是 max
**关键实现要点**:
- ratio = exp(log_π_new(a|s) - log_π_old(a|s))
- L = -E[min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)] - c1·VF_loss + c2·H(π)
- 一般 ε=0.2, c1=0.5, c2=0.01
- old logprob 在采样时存好（不能用现网络重算）
**易错**:
- ratio 没 detach old logprob → 重复梯度
- 忘了负号（最大化 → loss 取负）
- entropy bonus 符号错
- Advantage 没做 mean-std 归一化
- 多次 epoch 更新时 ratio 偏离过远
**为何考**: 这是 RL/RLHF/VLA 岗第一难题，几乎必考
**来源**: [zhihu 强化学习手撕PPO](https://zhuanlan.zhihu.com/p/2012930266168046909) · [CSDN PPO Loss](https://blog.csdn.net/weixin_36378508/article/details/152085552) · [CSDN PPO实现](https://blog.csdn.net/smartcat2010/article/details/144467884) · [hwcoder RLHF篇](https://hwcoder.top/Manual-Coding-6)

#### R5 · 手撕 GAE 优势估计 🔥×9 · L2
**考察点**: λ 在 0/1 之间的 bias/variance 权衡；从后往前递归计算
**关键实现要点**:
- δ_t = r_t + γV(s_{t+1}) - V(s_t)
- A_t^GAE = δ_t + γλ·A_{t+1}^GAE
- 反向遍历 trajectory；终止时 A_T = δ_T
- λ=0 → TD(0)（高 bias 低 var）；λ=1 → MC（低 bias 高 var）；常用 λ=0.95
**易错**:
- 正向算（错；要反向）
- done 处没截断 bootstrap
- V_target = A + V 还是 A 单独算（应是 V_target = A + V_old）
**来源**: [zhihu RL 手撕PPO](https://zhuanlan.zhihu.com/p/2012930266168046909) · [zhihu GAE](https://zhuanlan.zhihu.com/p/10343932079) · [博客园 GAE](https://www.cnblogs.com/cavalier-chen/p/18988014) · [CSDN GAE](https://blog.csdn.net/shizheng_Li/article/details/144436495)

#### R6 · 手撕 REINFORCE / policy gradient 推导 🔥×7 · L2
**考察点**: ∇J(θ) = E[∇log π(a|s) · R(τ)]；为何要 baseline 降方差
**关键实现要点**:
- loss = -mean(log_prob(a) * return)（return 通常是 G_t 或 advantage）
- baseline 减去状态价值 V(s) 不引入 bias
- 一般 MC rollout 完整 episode 后才更新
**易错**: 没标 stop_gradient on return / advantage；忘记负号；γ-discounted return 算错（应从后往前）
**来源**: [CSDN policy gradient 推导](https://blog.csdn.net/november_chopin/article/details/108032626) · [zhihu PG 推导](https://zhuanlan.zhihu.com/p/1962527453709857590) · [zhihu RL 基础四](https://zhuanlan.zhihu.com/p/31278940)

#### R7 · 手撕 DDPG / TD3 critic update 🔥×5 · L2
**考察点**: DDPG 是连续动作 Q-learning；TD3 三大改进（双 Q / 延迟 / target smoothing）
**关键实现要点**:
- DDPG target: `y = r + γ * Q_target(s', μ_target(s'))`
- TD3: `y = r + γ * min(Q1_t, Q2_t)(s', μ_t(s') + ε_clipped)`
- Actor 用 `-mean(Q1(s, μ(s)))` 上升
**易错**: 没 detach target；TD3 actor 用 Q1 还是 min(Q1,Q2)（用 Q1 即可）；exploration noise 跟 target smoothing noise 是两件事
**来源**: [博客园 调参技巧](https://www.cnblogs.com/ting1/p/16984892.html) · [CSDN TD3 vs SAC](https://blog.csdn.net/wq6qeg88/article/details/144288873) · [zhihu SAC vs PPO vs TD3](https://www.zhihu.com/question/6699179413)

#### R8 · 手撕 SAC（含 reparameterization + entropy）🔥×6 · L3
**考察点**: max entropy RL 目标；为何 SAC 需要 reparameterize
**关键实现要点**:
- π(a|s) 是 Gaussian，actor 输出 μ, log_σ；采样: a = tanh(μ + σ·ε), ε~N(0,1)
- log_prob 要补 tanh 的雅可比修正：`log_prob -= log(1 - tanh(u)^2 + 1e-6)`
- Q target: `r + γ(min(Q1_t, Q2_t)(s', a') - α·log π(a'|s'))`
- α 可学习: `loss_α = -log α * (log π(a|s) + target_entropy).detach()`
**易错**: 忘 tanh 修正；target_entropy 设错（一般 -|A|）；α 不自动调
**来源**: [CSDN TD3 vs SAC](https://blog.csdn.net/wq6qeg88/article/details/144288873)

#### R9 · 手撕 DPO loss 🔥×8 · L2
**考察点**: 由 RLHF 推导而来；只需 ref + policy 两个模型；不需要 reward model
**关键实现要点**:
- L = -log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))
- y_w = chosen, y_l = rejected
- β 控制 KL 强度
- 实现时累加 token-level log_prob 即可
**易错**: ref model 没 freeze；β 跟温度搞混；log_prob 没做 padding mask
**来源**: [zhihu DPO 公式](https://zhuanlan.zhihu.com/p/1904847799243220274) · [chaofa DPO](https://yuanchaofa.com/post/hands-on-dpo-direct-preference-optimization) · [CSDN DPO Loss](https://blog.csdn.net/weixin_36378508/article/details/152092226) · [zhihu RLHF-DPO](https://zhuanlan.zhihu.com/p/692991235)

#### R10 · 手撕 GRPO loss（group relative policy opt）🔥×5 · L3
**考察点**: DeepSeek-R1 的算法；group baseline 替代 critic；为何省内存
**关键实现要点**:
- 同一 prompt 采 G 个 response → reward r_1..r_G
- baseline = mean(r)，advantage A_i = (r_i - mean)/std
- 跟 PPO clipped loss 一样的形式，但 advantage 用 group 内 z-score
- 没有 value head
**易错**: 把 GRPO 当 DPO（DPO 是离线偏好对，GRPO 是在线 RL）；group size 选错（一般 G=4..8）
**来源**: [CSDN DeepSeek-R1 GRPO](https://deepseek.csdn.net/67c3fefc6670175f992e306b.html) · [博客园 GRPO 训练](https://www.cnblogs.com/zhiyong-ITNote/p/18702470) · [zhihu GRPO 解析](https://zhuanlan.zhihu.com/p/27454498505) · [hwcoder RLHF篇](https://hwcoder.top/Manual-Coding-6)

#### R11 · 手撕 importance sampling 比率 🔥×3 · L2
**考察点**: E_p[f] = E_q[f·p/q]；为何 off-policy 必须用
**关键实现要点**:
- IS ratio = p(x)/q(x)
- RL 中 ratio = π_θ(a|s) / π_θ_old(a|s)
- 多步 IS 累乘 → variance 爆炸（PPO clip 解决这个）
**易错**: 把 ratio 当 reward；忘 detach old policy
**来源**: [CSDN importance sampling](https://blog.csdn.net/hehedadaq/article/details/112232179) · [zhihu importance sampling](https://zhuanlan.zhihu.com/p/36816898) · [zhihu RL-IS](https://zhuanlan.zhihu.com/p/669378380)

---

### 1.3 模仿学习 / VLA（频次 ≥3，共 8 题）

#### IL1 · 手撕 Behavior Cloning（连续 vs 离散动作）🔥×7 · L1
**考察点**: 连续动作用 MSE / Gaussian NLL；离散动作用 CE；BC 的根本限制（distribution shift）
**关键实现要点**:
- 连续: `loss = MSE(π(s), a*)` 或 `loss = -log N(a*; μ(s), σ(s))`
- 离散: `loss = CE(logits=π(s), target=a*)`
- 监督学习 pipeline，无需 reward 信号
**易错**: 把 BC 当 RL；状态分布不匹配（OOD）问题没意识；用 L2 但忘加 σ 学习
**为何考**: VLA 训练的最朴素版本，π0/OpenVLA 在动作 head 上本质就是 BC
**来源**: [minari BC tutorial](https://minari.farama.org/v0.4.2/tutorials/using_datasets/behavioral_cloning/) · [CSDN 行为克隆](https://blog.csdn.net/weixin_45116099/article/details/135663133) · [腾讯云 BC](https://cloud.tencent.cn/developer/article/2128944)

#### IL2 · 手撕 Diffusion Policy 训练 step（DDPM 噪声预测）🔥×8 · L3
**考察点**: q(x_t|x_0) 的 closed form；ε-prediction 目标；条件 = visual obs
**关键实现要点**:
- β schedule（cosine 比 linear 好），ᾱ_t = ∏(1-β_i)
- forward: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`，ε~N(0,I)
- 模型: `ε_pred = model(x_t, t, condition=visual_feat)`
- loss = MSE(ε_pred, ε)
- 推理: DDIM/DDPM K 步迭代去噪 → action sequence
**易错**:
- 忘了 visual obs 作为 condition cross-attention 进 UNet
- 推理时 K 选错（训练 1000 步，推理 DDIM 10-50 步）
- 没 EMA → 生成质量差
- 把 action 当图像维度（应是低维向量 [pred_horizon, action_dim]）
**为何考**: π0 / Diffusion Policy / RDT 都是这个范式
**来源**: [CSDN DDPM 深度指南](https://www.cnblogs.com/zhangdoudou/p/18537276) · [CSDN Diffusion Policy](https://blog.csdn.net/zhaoliang38/article/details/139135283) · [CSDN DDPM 系列1](https://blog.csdn.net/g11d111/article/details/131326934) · [coderSarts DDPM PyTorch](https://www.codersarts.com/post/how-to-build-a-diffusion-model-from-scratch-in-pytorch-ddpm-ddim-classifier-free-guidance) · [TDS DDPM](https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/)

#### IL3 · 手撕 ACT action chunking loss 🔥×5 · L2
**考察点**: chunk 长度 k；CVAE 风格的 latent z；temporal ensemble 推理
**关键实现要点**:
- 编码: `(qpos, a_1..a_k) → z` 用 BERT 风格 encoder, [CLS] 输出 μ/log_σ
- 解码: `cross_attn(query=fixed, k/v=resnet_feat ⊕ qpos ⊕ z) → â_1..â_k`
- loss = L1(â, a) + β·KL(N(μ,σ²) ‖ N(0,I))
- 推理时 z=0；temporal ensemble 用指数加权融合多步预测
**易错**: chunk 长度太短退化为 BC；β KL 系数选错；推理时还采样 z
**来源**: [zhihu ACT 解读](https://zhuanlan.zhihu.com/p/676520960) · [HF lerobot ACT](https://hugging-face.cn/docs/lerobot/act) · [vizuara ACT](https://vizuara.substack.com/p/how-does-act-action-chunking-with) · [GitHub Shaka-Labs ACT](https://github.com/Shaka-Labs/ACT)

#### IL4 · 手撕 OpenVLA 风格 action token CE loss 🔥×4 · L2
**考察点**: 把连续动作离散化为 256 bin → 用 LM head 的 CE 训
**关键实现要点**:
- 每维 action 离散化到 256 bin（uniform 或 quantile）
- action token = vocab 中预留 256 个特殊 token
- training: 标准 next-token CE loss
- 推理: argmax 取 token → de-tokenize 回连续值
**易错**: bin 数目太少 → 精度不够；bin 边界与 dataset 分布不匹配；忘 mask 掉非 action token 的 loss
**来源**: [zhihu OpenVLA](https://zhuanlan.zhihu.com/p/1925499475973116151) · [zhihu VLA 求职路线](https://zhuanlan.zhihu.com/p/1955704833358160798)

#### IL5 · 手撕 CLIP contrastive loss (InfoNCE) 🔥×9 · L2
**考察点**: 对称交叉熵；温度 τ 的作用；为何 image→text + text→image 各算一遍
**关键实现要点**:
- 投影并归一化: `img_emb = F.normalize(W_i @ f)`, `txt_emb = F.normalize(W_t @ g)`
- logits = `img_emb @ txt_emb.T / τ`（τ ~ exp(t), t 可学）
- labels = `torch.arange(N)`
- `loss = (CE(logits, labels) + CE(logits.T, labels)) / 2`
**易错**: 忘归一化；τ 没 clamp 导致梯度爆炸；batch size 太小 → 负样本不够
**为何考**: VLN/Embodied VLM 的视觉-文本对齐底层都是这个
**来源**: [CSDN CLIP 训练](https://blog.csdn.net/shizheng_Li/article/details/144460416) · [zhihu CLIP loss 源码](https://zhuanlan.zhihu.com/p/624173920) · [博客园 CLIP loss](https://www.cnblogs.com/chester-cs/p/17478159.html) · [CSDN CE 实现](https://blog.csdn.net/caroline_wendy/article/details/125088243)

#### IL6 · 手撕 DAgger（数据聚合）训练流程 🔥×3 · L2
**考察点**: 解决 BC 的 distribution shift；每轮拿当前 π 跑 rollout，专家标注新数据回灌
**关键实现要点**:
- 初始: BC 在专家数据 D_0 上训
- 迭代: π_i rollout 拿新 obs → 专家 label → D_{i+1} = D_i ∪ new
- 注意 β-mix: 早期用专家动作多，后期用 π
**易错**: 没 β-mix 直接全用 π → 早期还是 OOD；专家标注成本爆炸
**来源**: [CSDN 模仿学习路线](https://blog.csdn.net/Yong_Qi2015/article/details/148621978) · [CSDN BC](https://blog.csdn.net/qq_40206371/article/details/125061986)

#### IL7 · 手撕 ResNet residual block 🔥×6 · L1
**考察点**: shortcut 跳过；BN 在 conv 之后 ReLU 之前；下采样时 shortcut 用 1x1 conv 升维
**关键实现要点**:
- `out = Conv3x3 → BN → ReLU → Conv3x3 → BN`
- `out += shortcut(x)`（如果 channel/stride 不同，shortcut 用 1x1 conv）
- `out = ReLU(out)`
**易错**: ReLU 位置（应在加之后）；BN 在 ReLU 后（应在前）；downsample 没匹配 shape
**来源**: [CSDN 手动实现 ResNet](https://blog.csdn.net/bu_fo/article/details/109203360) · [d2l ResNet](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html) · [zhihu ResNet 代码](https://zhuanlan.zhihu.com/p/169460083)

#### IL8 · 手撕 PointNet（permutation invariance + T-Net）🔥×3 · L2
**考察点**: max pooling 解决无序；T-Net 学旋转/对齐矩阵
**关键实现要点**:
- 共享 MLP 处理每个点 → [N, d]
- max pool over points → [d] 全局特征
- T-Net 输出 3×3 (或 64×64) 旋转矩阵，应用到输入点
**易错**: 用 mean pool（错；max 才是对称函数）；T-Net 矩阵没正则化（应加 ||I - AA^T||²）
**来源**: [CSDN PointNet 笔记](https://blog.csdn.net/hongbin_xu/article/details/84638109) · [CSDN T-Net](https://blog.csdn.net/weixin_45641915/article/details/132340216)

---

### 1.4 视觉/感知与几何（频次 ≥3，共 10 题）

#### V1 · 手撕 IoU（两个矩形框）🔥×11 · L1
**考察点**: 交集 / 并集；为何要 max(0,·) 防止负值；axis-aligned 与 rotated 的区别
**关键实现要点**:
- inter_w = max(0, min(x2a, x2b) - max(x1a, x1b))
- inter_h 同理
- inter = inter_w * inter_h
- iou = inter / (area_a + area_b - inter + 1e-6)
**易错**: 负值没 clip → IoU 可能 >1；面积重叠为 0 时除零
**来源**: [CSDN IOU+NMS](https://blog.csdn.net/Jiangnan_Cai/article/details/132662191) · [CSDN 手撕题二](https://blog.csdn.net/qq_39444290/article/details/137288088) · [博客园 手撕题二](https://www.cnblogs.com/chuqianyu/p/18062881) · [zhihu 秋招手撕](https://zhuanlan.zhihu.com/p/666849216) · [牛客 CV 手撕(一)](https://www.nowcoder.com/discuss/756981253181542400) · [牛客 CV 手撕(三)](https://www.nowcoder.com/discuss/768600194819559424)

#### V2 · 手撕 NMS（非极大值抑制）🔥×11 · L2
**考察点**: 排序 + 贪心；soft-NMS 区别（不删，降分）
**关键实现要点**:
- 按 conf 降序
- 取第一个，与剩余算 IoU
- IoU > thresh 的删；剩下的继续
- 复杂度 O(N²)
**易错**: 没按类别分开做；忘了 score thresh 预过滤；用 IoU 1.0 当阈值
**来源**: [CSDN IOU+NMS](https://blog.csdn.net/Jiangnan_Cai/article/details/132662191) · [CSDN NMS](https://blog.csdn.net/weixin_43662553/article/details/125285305) · [zhihu NMS](https://zhuanlan.zhihu.com/p/512686599) · [CSDN 手撕 IOU NMS BN](https://blog.csdn.net/weixin_44398263/article/details/123407466)

#### V3 · 手撕 Conv2d 输出维度 / 感受野计算 🔥×8 · L1
**考察点**: out_h = ⌊(H + 2P - K) / S⌋ + 1；感受野递推
**关键实现要点**:
- 输出: `out = (in + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
- 感受野: `RF_l+1 = RF_l + (K_l+1 - 1) * ∏(stride_i)`
**易错**: 忘 dilation；池化层也算 stride；感受野是叠加不是相乘
**来源**: [d2l 填充步幅](https://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html) · [zhihu 卷积面试](https://zhuanlan.zhihu.com/p/477558365) · [zhihu CNN 卷积细节](https://zhuanlan.zhihu.com/p/85296579)

#### V4 · 手撕 2D Conv (numpy 实现) 🔥×4 · L2
**考察点**: 滑窗 + 卷积核翻转（CNN 中其实是 cross-correlation，不翻转）
**关键实现要点**:
- 双重循环遍历输出位置
- 每个位置取 K×K patch 与 kernel 做 elementwise product 再 sum
- 加 padding：先 np.pad
**易错**: 忘了 batch / channel 维度；in_c × out_c 没 broadcast；padding 模式（'valid' / 'same'）
**来源**: [楚千羽手撕(一)](https://www.cnblogs.com/chuqianyu/p/18048501) · [CSDN 手撕 IOU NMS BN](https://blog.csdn.net/weixin_44398263/article/details/123407466) · [牛客 CV 手撕(一)](https://www.nowcoder.com/discuss/756981253181542400)

#### V5 · 手撕 max pooling / avg pooling（numpy）🔥×4 · L1
**考察点**: stride 与 kernel 关系；为何 pooling 没有可学参数；max pool 反向梯度只回最大点
**关键实现要点**:
- 滑窗 + np.max / np.mean
- backward：max 把梯度全给最大位置（argmax 处），avg 平均分配
**易错**: 把 pooling 看作 conv（pooling 没有 kernel 参数）
**来源**: [CSDN 手撕 IOU NMS BN](https://blog.csdn.net/weixin_44398263/article/details/123407466)

#### V6 · 手撕 BFS / DFS 在 grid 上找最短路径 🔥×9 · L2
**考察点**: 经典 leetcode pattern；BFS 必给最短，DFS 不一定；4 邻居 vs 8 邻居
**关键实现要点**:
- BFS: `deque`, visited set, 一层一层 expand
- DFS 递归或栈，回溯时撤销 visited
- 网格题模板: (x,y) + dx[4], dy[4]
**易错**: visited 没及时标记导致重复入队 → TLE；DFS 求最短（错，需 BFS）
**来源**: [leetcode 图论题单](https://leetcode.cn/discuss/post/3581143/fen-xiang-gun-ti-dan-tu-lun-suan-fa-dfsb-qyux/) · [zhihu BFS 总结](https://zhuanlan.zhihu.com/p/62884729) · [牛客 BFS/DFS 合集](https://blog.nowcoder.net/n/dd4bbbfc1c634db7a97134fe0d110e35)

#### V7 · 手撕 A* / Dijkstra 路径规划 🔥×7 · L2
**考察点**: Dijkstra 没启发；A* 用 f=g+h，h 是 admissible 启发；为何 A* 比 BFS 快
**关键实现要点**:
- 用 heapq (priority queue)
- 节点: (f, g, position)
- close set 保证不重复展开
- A* heuristic 一般用 Manhattan / Euclidean / Chebyshev
**易错**: heuristic 不 admissible → 不最优；忘 close set 导致重复展开
**为何考**: VLN 中导航模块、机器人规划必考
**来源**: [CSDN A*/Dijkstra](https://blog.csdn.net/weixin_46039719/article/details/128585574) · [博客园 A* 算法](https://www.cnblogs.com/QiQi-Robotics/p/14931545.html) · [博客园 Dijkstra](https://www.cnblogs.com/QiQi-Robotics/p/14927660.html)

#### V8 · 手撕 K-Means 聚类 🔥×4 · L1
**考察点**: E-step (assignment) + M-step (update centers)；k 选择；初始化
**关键实现要点**:
- 随机初始化 k 个 center
- 每点分配到最近 center
- center 更新为该 cluster 均值
- 直到 center 变化 < eps
**易错**: 初始化用 K-Means++ 否则收敛慢；空簇没处理
**来源**: [zhihu 秋招手撕](https://zhuanlan.zhihu.com/p/666849216) · [楚千羽手撕(一)](https://www.cnblogs.com/chuqianyu/p/18048501)

#### V9 · 手撕 PCA（主成分分析）🔥×3 · L2
**考察点**: 协方差矩阵特征分解 / SVD；前 k 个特征向量
**关键实现要点**:
- 数据中心化
- cov = X.T @ X / (n-1)
- 特征分解或 SVD(X)
- 取前 k 个特征向量作为投影矩阵
**易错**: 没中心化；用了 max eigenvalue 而非 top-k
**来源**: [zhihu 秋招手撕](https://zhuanlan.zhihu.com/p/666849216)

#### V10 · 手撕 NeRF 前向（位置编码 + density + RGB）🔥×3 · L3
**考察点**: 位置编码 sin/cos × 2^k；MLP 输出 σ + RGB；体渲染累积
**关键实现要点**:
- pos_enc(x) = [sin(2^0·x), cos(2^0·x), ..., sin(2^L·x), cos(2^L·x)]
- MLP1: pos_enc(x) → 256 维 + density σ
- MLP2: 256 + pos_enc(d) → RGB
- 体渲染: `C(r) = Σ T_i (1-exp(-σ_i δ_i)) c_i`
**易错**: 位置编码维度数量错（L=10 一般）；δ_i 是采样点间距，不是位置；T_i 累乘忘了
**来源**: 体渲染基础在多数 NeRF 教程都有，但具身岗考频较低（confidence: low）

---

### 1.5 数值稳定与工程（频次 ≥3，共 8 题）

#### N1 · 手撕 numerically stable softmax 🔥×8 · L1
**考察点**: 减去 max 防止 exp 溢出
**关键实现要点**:
- `x_max = x.max(axis=-1, keepdims=True)`
- `exp_x = exp(x - x_max)`
- `softmax = exp_x / exp_x.sum(axis=-1, keepdims=True)`
**易错**: 没 keepdims 导致 broadcast 错；用 float32 还可能溢出（直接 -inf）
**来源**: [zwn LogSumExp](https://www.zwn2001.space/posts/Graduate-Works/DL/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-LogSumExp%E6%8A%80%E5%B7%A7/index.html) · [CSDN softmax/LSE](https://blog.csdn.net/qq_27590277/article/details/125568062) · [zhihu 手撕 Softmax CE](https://zhuanlan.zhihu.com/p/9231669809)

#### N2 · 手撕 LogSumExp / log_softmax 🔥×5 · L2
**考察点**: log(Σexp(x)) = max + log(Σexp(x-max))；log_softmax 更稳
**关键实现要点**:
- `lse(x) = x.max() + log(exp(x - x.max()).sum())`
- `log_softmax(x) = x - lse(x)`（一步而非两步分开）
**易错**: 直接 log(softmax(x))（数值不稳）
**来源**: [zwn LogSumExp](https://www.zwn2001.space/posts/Graduate-Works/DL/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-LogSumExp%E6%8A%80%E5%B7%A7/index.html)

#### N3 · 手撕 cross-entropy（含数值稳定）🔥×7 · L1
**考察点**: 与 NLL Loss 关系；为何用 log_softmax 而非 softmax + log
**关键实现要点**:
- `CE(logits, target) = -log_softmax(logits)[target]`
- 多分类用 `F.cross_entropy(logits, target)` 内部已稳定
- one-hot 版本: `-Σ y_i · log(p_i)`
**易错**: 把 softmax + log 分开调用导致数值不稳；忘 ignore_index（padding 时常用）
**来源**: [zhihu Softmax/CE 手撕](https://zhuanlan.zhihu.com/p/9231669809) · [CSDN CE 实现](https://blog.csdn.net/caroline_wendy/article/details/125088243) · [博客园 CE 详解](https://www.cnblogs.com/clnchanpin/p/19457261)

#### N4 · 手撕 KL divergence（forward / reverse）🔥×5 · L2
**考察点**: KL(p||q) 与 KL(q||p) 不对称；高斯之间 KL 闭式
**关键实现要点**:
- 离散: `KL = Σ p(x) log(p(x)/q(x))`
- 高斯 vs 标准: `0.5 * (μ² + σ² - 1 - 2·log σ)` 沿特征维 sum
- forward (mode-covering) vs reverse (mode-seeking)
**易错**: log 比例符号；输入接 log_softmax 还是 softmax (PyTorch F.kl_div 要 log-prob)
**来源**: [zhihu VAE KL](https://zhuanlan.zhihu.com/p/345095899) · [CSDN 多变量高斯 KL](https://blog.csdn.net/wangpeng138375/article/details/78060753) · [博客园 KL 推导](https://www.cnblogs.com/qizhou/p/13804283.html) · [hsinjhao KL](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/)

#### N5 · 手撕 mini-batch 训练循环 🔥×9 · L1
**考察点**: zero_grad / backward / step 顺序；为何要 zero_grad
**关键实现要点**:
```
for x, y in loader:
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```
- `zero_grad` 必须在 backward 前
- 推理时 `model.eval() + torch.no_grad()`
**易错**: 顺序错；忘了 eval mode；多任务时 zero_grad 漏了某个 optimizer
**来源**: [博客园 zero_grad](https://www.cnblogs.com/h694879357/p/15855281.html) · [CSDN backward 顺序](https://blog.csdn.net/PanYHHH/article/details/107361827) · [CSDN 训练循环详解](https://blog.csdn.net/vvilkim/article/details/148480202)

#### N6 · 手撕 warmup + cosine schedule 🔥×4 · L1
**考察点**: warmup 防止初始 lr 太大；cosine 末期渐降
**关键实现要点**:
- warmup_steps 前线性增 0→lr_max
- 之后 `lr_t = 0.5 * lr_max * (1 + cos(π · (t-warmup)/(T-warmup)))`
**易错**: 边界不连续（warmup_end 时 lr 突跳）；T 算错（总 step vs 总 epoch）
**来源**: [CSDN warmup+cosine](https://blog.csdn.net/sinat_41667032/article/details/116454406) · [博客园 warmup 调研](https://www.cnblogs.com/Stareven233/p/17870826.html) · [d2l lr scheduler](https://zh.d2l.ai/chapter_optimization/lr-scheduler.html)

#### N7 · 手撕 Adam 优化器更新 🔥×5 · L2
**考察点**: m / v 一阶/二阶 momentum；bias correction；为何 AdamW 比 Adam 好
**关键实现要点**:
- m_t = β1·m_{t-1} + (1-β1)·g_t
- v_t = β2·v_{t-1} + (1-β2)·g_t²
- bias correct: m̂ = m_t/(1-β1^t), v̂ = v_t/(1-β2^t)
- update: `θ -= lr · m̂ / (√v̂ + ε)`
- AdamW: weight decay 直接乘 θ，不进 momentum
**易错**: 忘 bias correction；weight decay 加到 grad 里（错；这是 L2 正则不是 AdamW）
**来源**: [zhihu Adam 优化器](https://zhuanlan.zhihu.com/p/1928484130594747517) · [CSDN Adam 详解](https://blog.csdn.net/weixin_39228381/article/details/108548413)

#### N8 · 手撕 dropout（训练 vs 推理）🔥×3 · L1
**考察点**: 推理时不 drop 且要 scale；inverted dropout 直接训练时 /(1-p)
**关键实现要点**:
- train: `mask ~ Bernoulli(1-p)`, `out = x * mask / (1-p)`
- eval: `out = x`
- 等价 scale 加在训练侧（PyTorch 标准实现）
**易错**: 推理时还 drop；忘 scale 导致 train/test 期望不匹配
**来源**: [hwcoder 神经网络篇](https://hwcoder.top/Manual-Coding-2)

---

### 1.6 概率与统计基础（频次 ≥3，共 5 题）

#### P1 · 手撕高斯 KL 闭式 🔥×6 · L2
**考察点**: KL(N(μ1, σ1²) ‖ N(μ2, σ2²))
**关键公式**:
- 一维: `log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 0.5`
- 多维: `0.5 * (log(|Σ2|/|Σ1|) + tr(Σ2⁻¹Σ1) + (μ2-μ1)^T Σ2⁻¹ (μ2-μ1) - d)`
- VAE 用的特殊情况 (vs N(0,I)): `0.5 * Σ(μ² + σ² - 1 - log σ²)`
**易错**: log 用 σ 还是 σ²；多元的 trace 项漏；维度 d 漏减
**来源**: [博客园 多维高斯 KL 推导](https://www.cnblogs.com/qizhou/p/13804283.html) · [CSDN 多变量 KL](https://blog.csdn.net/wangpeng138375/article/details/78060753)

#### P2 · 手撕 VAE ELBO + 重参数化 🔥×7 · L2
**考察点**: ELBO = E[log p(x|z)] - KL(q(z|x) ‖ p(z))；reparameterize 让梯度可回传
**关键实现要点**:
- encoder 输出 μ, log_σ²
- reparameterize: `z = μ + σ·ε`, ε~N(0,I)
- recon loss + KL loss
- 通常 `loss = recon + β · KL`（β-VAE）
**易错**:
- 直接采样 N(μ,σ) → 梯度断
- log_σ 没 exp 就当 σ
- KL 加错符号
- recon 用 MSE 还是 BCE 取决于数据
**来源**: [CSDN VAE 详解](https://blog.csdn.net/m0_56942491/article/details/136265500) · [zhihu VAE ELBO](https://zhuanlan.zhihu.com/p/138123592) · [zhihu 手撕 VAE](https://zhuanlan.zhihu.com/p/719968411) · [CSDN reparameterization](https://blog.csdn.net/shizheng_Li/article/details/145996540)

#### P3 · 手撕 reparameterization trick（通用版）🔥×4 · L2
**考察点**: 哪些分布支持（Gaussian 一阶可微）；哪些不支持（Bernoulli 需要 Gumbel-Softmax）
**关键实现要点**:
- Gaussian: `z = μ + σ·ε`
- Gumbel-Softmax for discrete: `z = softmax((logits + Gumbel(0,1)) / τ)`
**易错**: 不知道 Bernoulli 不能直接 reparam
**来源**: [CSDN reparameterization](https://blog.csdn.net/shizheng_Li/article/details/145996540) · [CSDN VAE reparameterization](https://blog.csdn.net/weixin_51176105/article/details/134741436)

#### P4 · 手撕 sigmoid + log-sigmoid 数值稳定 🔥×3 · L1
**考察点**: BCE 实现；log(1+exp(x)) 数值稳定
**关键实现要点**:
- `log σ(x) = -softplus(-x)`
- `log(1-σ(x)) = -x - softplus(-x)`
- `BCE(logit, y) = max(logit, 0) - logit·y + log(1+exp(-|logit|))`
**易错**: 直接 log(sigmoid)（x 很负时数值不稳）
**来源**: [zwn LogSumExp](https://www.zwn2001.space/posts/Graduate-Works/DL/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-LogSumExp%E6%8A%80%E5%B7%A7/index.html)

#### P5 · 手撕 Bernoulli/Categorical sampling + log_prob 🔥×3 · L1
**考察点**: 多项式采样；如何算 log_prob（用 logits 比 probs 稳）
**关键实现要点**:
- 用 `torch.distributions.Categorical(logits=logits)`
- log_prob = `F.log_softmax(logits, -1).gather(-1, a)`
**易错**: 用 probs 而非 logits 导致 0 概率时 log(0)；torch.multinomial 不返回 log_prob
**来源**: [hwcoder 神经网络篇](https://hwcoder.top/Manual-Coding-2)

---

### 1.7 低频补充 / 仅供参考（频次 1-2，9 题，备选）

- **手撕 dueling DQN**（Q = V + A - mean(A) 形式）freq=2 · L2 · [zhihu D3QN](https://zhuanlan.zhihu.com/p/490163865)
- **手撕 prioritized experience replay**（按 TD error 排）freq=2 · L3
- **手撕 SwiGLU** activation (LLaMA FFN 用)：`x·SiLU(Wx)` freq=2 · L1
- **手撕 GELU** (近似版 vs 精确版) freq=2 · L1
- **手撕 Gumbel-Softmax**（离散 reparam）freq=1 · L2
- **手撕 contrastive predictive coding (CPC)** freq=1 · L3
- **手撕 ALiBi** 位置编码（线性 bias）freq=1 · L2
- **手撕 Hungarian matching**（DETR loss）freq=1 · L3
- **手撕 BC-Z / language-conditioned BC** freq=1 · L2

---

## 2. LeetCode 调研

### 2.1 按公司分组（信心评估）

| 公司 | 题型偏好 | 信心 | 报告原题（部分） |
|---|---|---|---|
| **字节 Seed Robotics** | 链表 > 字符串/哈希 > 树/图 > 栈/队列 > DP > CV | 高（CodeTop 公开数据） | LC3 无重复字符最长子串(×87) / LC15 三数之和(×53) / LC206 反转链表 / LC25 K个一组反转 / LC121 买卖股票 |
| **智元（Agibot）** | 反转链表、二叉树遍历、二分查找；偏简单中等 | 中（少量牛客面经） | LC206 反转链表 / 二叉树层序 |
| **银河通用** | 项目拷打为主，手撕中等 | 低（绝对面经数少） | - |
| **星海图** | 算法岗考 VLA/RL 项目细节 + 中等 leetcode | 低 | - |
| **小米机器人 / 小米大模型** | 链表/树/字符串；八股 + 手撕反转链表/删除倒数第 n 个节点 | 中 | LC19 删除链表倒数第 N 个 / LC206 反转链表 / LC3 最长子串 |
| **蔚来 智能装备 / 自动驾驶** | 力扣原题；A* 路径规划走流程问；Trie | 中（自动驾驶面经丰富） | A* / Dijkstra 流程 / LC212 单词搜索 II |
| **小鹏 / 理想 自动驾驶** | DP / 滑动窗口 / 二分；A* / KD-Tree | 中 | LC53 最大子序和 / LC300 最长递增子序列 |
| **美团无人车** | 双指针 / DP / 二分 | 中 | LC42 接雨水 / LC239 滑动窗口最大值 |
| **WeRide / 文远** | 数据结构基础 + 计算几何 + DP | 中 | 中等-困难偏多 |
| **Tesla（AI/Optimus）** | 中等 leetcode 加 OOD design (LRU/parking lot) | 中 | LC146 LRU / LC41 第一个缺失正数 / LC300 LIS / 设计 parking lot |
| **Boston Dynamics / AI Institute** | C++ heavy；线程安全队列；2 道 LC Hard | 中（Glassdoor 部分面经） | LC23 合并 K 个有序链表 / LC42 接雨水 / thread-safe queue |
| **GM / Figure / 1X** | 偏向项目+设计，coding 不多 | 低 | - |

> **Note**: 牛客面经汇总指出 `算法岗考察频次: 链表 > 字符串/哈希 > 二叉树与图 > 栈/队列 > 查找/排序/搜索 > 动态规划 > CV > 其他`。

### 2.2 高频原题统一列表（频次 ≥3，30 题）

> 频次基于跨公司面经合并统计（CodeTop / 牛客 / 知乎 / 一亩三分地交叉验证）。

| 编号 | 题号 | 题名 | 难度 | 频次 | 类型 |
|---|---|---|---|---|---|
| L1 | LC3 | 无重复字符的最长子串 | 中 | 🔥×30 | 滑动窗口 |
| L2 | LC15 | 三数之和 | 中 | 🔥×20 | 双指针 |
| L3 | LC206 | 反转链表 | 易 | 🔥×25 | 链表 |
| L4 | LC25 | K 个一组翻转链表 | 难 | 🔥×12 | 链表 |
| L5 | LC146 | LRU 缓存机制 | 中 | 🔥×18 | 设计/哈希+双链表 |
| L6 | LC215 | 数组中第 K 个最大元素 | 中 | 🔥×15 | 堆/快速选择 |
| L7 | LC53 | 最大子序和 | 易 | 🔥×14 | DP |
| L8 | LC121 | 买卖股票的最佳时机 | 易 | 🔥×12 | DP |
| L9 | LC42 | 接雨水 | 难 | 🔥×11 | 双指针/单调栈 |
| L10 | LC1 | 两数之和 | 易 | 🔥×10 | 哈希 |
| L11 | LC141 | 环形链表 | 易 | 🔥×9 | 快慢指针 |
| L12 | LC102 | 二叉树层序遍历 | 中 | 🔥×11 | BFS |
| L13 | LC236 | 二叉树最近公共祖先 | 中 | 🔥×8 | 递归 |
| L14 | LC200 | 岛屿数量 | 中 | 🔥×10 | DFS/BFS（grid） |
| L15 | LC994 | 腐烂的橘子 | 中 | 🔥×6 | 多源 BFS |
| L16 | LC239 | 滑动窗口最大值 | 难 | 🔥×8 | 单调队列 |
| L17 | LC32 | 最长有效括号 | 难 | 🔥×6 | 栈 / DP |
| L18 | LC20 | 有效的括号 | 易 | 🔥×8 | 栈 |
| L19 | LC155 | 最小栈 | 易 | 🔥×7 | 栈 |
| L20 | LC300 | 最长递增子序列 | 中 | 🔥×7 | DP |
| L21 | LC72 | 编辑距离 | 难 | 🔥×6 | DP |
| L22 | LC5 | 最长回文子串 | 中 | 🔥×6 | DP/中心扩展 |
| L23 | LC22 | 括号生成 | 中 | 🔥×5 | 回溯 |
| L24 | LC46 | 全排列 | 中 | 🔥×6 | 回溯 |
| L25 | LC56 | 合并区间 | 中 | 🔥×7 | 区间排序 |
| L26 | LC560 | 和为 K 的子数组 | 中 | 🔥×8 | 前缀和+哈希 |
| L27 | LC169 | 多数元素 | 易 | 🔥×4 | Boyer-Moore |
| L28 | LC23 | 合并 K 个升序链表 | 难 | 🔥×7 | 堆/分治 |
| L29 | LC19 | 删除链表倒数第 N 个节点 | 中 | 🔥×6 | 双指针 |
| L30 | LC1143 | 最长公共子序列 | 中 | 🔥×5 | DP |

### 2.3 题型分布（基于上面 30 题）

| 类型 | 占比 | 说明 |
|---|---|---|
| 链表 | 17% | 反转/合并/双指针 |
| DP | 23% | 一维 DP / 区间 / 串DP |
| 双指针 / 滑动窗口 | 17% | 滑窗经典 |
| 树（BFS/DFS） | 13% | 层序/LCA |
| 图算法（grid BFS/DFS） | 10% | 导航相关公司偏好 |
| 栈 / 单调栈 / 单调队列 | 13% | |
| 设计题 | 7% | LRU / 最小栈 |
| 回溯 | 7% | |

**结论**: 链表 + DP + 滑窗 三件套占近 60%，必须熟。grid BFS/DFS 对 VLN/导航岗尤其有用。

---

## 3. 系统设计题

### 3.1 ML 系统设计（5 题）

#### S1 · 设计一个 VLA 训练 pipeline（数据 → 训练 → 部署）🔥×5 · L3
**考察点**:
- 数据格式（RLDS / TFDS / LeRobot HF dataset 等）
- 多机器人 cross-embodiment 数据混合（不同 dof / action space）
- 视觉 backbone 选型（SigLIP / DINOv2 / CLIP）
- 训练: DDP + ZeRO / FSDP；学习率 schedule
- checkpoint 管理 / EMA
**易错**: 没考虑 cross-embodiment；忘提 mixed precision；data loader bottleneck
**来源**: [zhihu OpenVLA](https://zhuanlan.zhihu.com/p/1925499475973116151) · [zhihu π0](https://blog.csdn.net/v_JULY_v/article/details/143472442) · [zhihu VLA 高效化](https://zhuanlan.zhihu.com/p/1964132820160053654)

#### S2 · 设计 VLA 推理服务（多机器人并发调用一个 endpoint）🔥×4 · L3
**考察点**:
- batching strategy（动态 batch / continuous batch）
- KV cache 管理（PagedAttention 思想）
- 帧率约束（机器人控制 20-50Hz vs VLA 1-5Hz）→ action chunking
- 多模型路由 (LLaVA / OpenVLA / π0)
**易错**: 没考虑 latency budget；KV cache 在多请求间没共享；没说 streaming token output
**来源**: [zhihu VLA 实时控制](https://zhuanlan.zhihu.com/p/2022992211273348705) · [zhihu VLA 综述](https://zhuanlan.zhihu.com/p/1907961280112877856) · [zhihu VLA 高效化](https://zhuanlan.zhihu.com/p/1964132820160053654)

#### S3 · 设计 RLHF / preference 数据采集系统 🔥×3 · L2
**考察点**:
- pairwise comparison UI
- 标注员一致性（kappa 系数）
- 数据存储 schema（prompt, chosen, rejected, annotator, timestamp）
- reward model training 与数据采集解耦
**易错**: 没说 annotator bias；缺一致性检验；reward model 过拟合
**来源**: [datawhalechina RLHF](https://github.com/datawhalechina/base-llm/blob/main/docs/chapter12/01_RLHF.md) · [HF blog RLHF 中文](https://huggingface.co/blog/zh/rlhf) · [鹤啸九天 RLHF](https://wqw547243068.github.io/rlhf)

#### S4 · 设计 multi-robot fleet data collection 🔥×3 · L3
**考察点**:
- 多机器人异构数据汇总 (RLDS standard)
- delta vs absolute action 标准化
- 时间戳对齐与传感器同步
- 隐私 / 数据脱敏（人脸等）
**易错**: 没考虑不同 robot 的 control frequency；忘了 calibration
**来源**: 多来自 LeRobot / OpenX-Embodiment 文档（confidence: medium）

#### S5 · 设计 LLM offline batch inference for trajectory labeling 🔥×3 · L2
**考察点**:
- vLLM / SGLang 批处理
- prefix cache 复用（相同 system prompt 跨样本共享）
- 失败重试 / 数据漂移监控
**易错**: 没说 cost estimation；没说 quality sampling
**来源**: [vLLM 文档 + zhihu VLA 综述](https://zhuanlan.zhihu.com/p/1907961280112877856)

### 3.2 机器人系统设计（5 题）

#### S6 · 设计 perception-planning-control 完整栈 🔥×4 · L3
**考察点**:
- 模块: perception (LiDAR/camera fusion) → state estimation → planning (global+local) → control (PID/MPC)
- ROS 2 node 划分；topic 频率（perception 10-30Hz, control 100Hz+）
- 安全层（emergency stop / collision check）
**易错**: 没分高低频；忘了 safety layer；没说 simulation-real gap
**来源**: [TechSynapse ROS2/MoveIt](https://www.cnblogs.com/TS86/p/18888159) · [zhihu ROS 具身入门](https://zhuanlan.zhihu.com/p/20329244481) · [emergentmind ROS framework](https://www.emergentmind.com/topics/ros-based-control-framework)

#### S7 · 设计 multi-sensor fusion 系统（camera + LiDAR + IMU）🔥×3 · L3
**考察点**:
- 时间同步（PTP / hardware trigger）
- 标定（extrinsic / intrinsic）
- EKF / UKF / factor graph
- 故障检测与降级（sensor dropout）
**易错**: 不知道 hardware trigger 比软同步精度高；没提 covariance 怎么估计
**来源**: [emergentmind ROS-based control](https://www.emergentmind.com/topics/ros-based-control-framework) · [深蓝学院 ROS 课程](https://www.shenlanxueyuan.com/archive/course/168/lesson/893)

#### S8 · 设计 VLN agent 系统（自然语言指令→导航）🔥×3 · L3
**考察点**:
- 语义地图构建（CLIP/Grounding DINO 标注 voxel）
- 指令解析（subgoal 拆分）
- waypoint policy + low-level controller
- 失败 recovery / 重新规划
**易错**: 没说 closed-loop；没说 hallucination 处理
**用户研究方向直接相关**

#### S9 · 设计 SLAM 状态估计模块 🔥×3 · L3
**考察点**:
- frontend (feature extraction / matching) + backend (BA / pose graph)
- loop closure 检测（DBoW / NetVLAD）
- IMU 预积分
**易错**: 不知道 frontend/backend 分工；忘 loop closure
**来源**: 来自自动驾驶 SLAM 面经（confidence: medium-high）

#### S10 · 设计 ROS 2 节点架构（双臂操作 demo）🔥×3 · L2
**考察点**:
- 节点拆分（perception / planning / arm_controller / gripper）
- topic vs service vs action（不同通信模式）
- QoS profile（reliable vs best_effort）
- launch file 编排
**易错**: 把所有功能塞一个节点；不会用 action 做长时任务
**来源**: [TechSynapse ROS2/MoveIt](https://www.cnblogs.com/TS86/p/18888159)

---

## 4. 给用户的插入建议（关键交付物）

### Option A: 散布插入现有 7 卷

| 现有卷 | 插入位置 | 数量 | 主题 |
|---|---|---|---|
| vol-1 basics | 新增 §I "手撕代码基础" | 10 题 | T1-T2, T5-T8, V1-V3, N5 |
| vol-2 rl_algo | 新增 §G "手撕 RL 核心" | 10 题 | R1-R5, R7-R9, R11, P5 |
| vol-3 vla_il | 新增 §F "手撕 IL/VLA" | 8 题 | IL1-IL5, IL6, IL7, T11 |
| vol-4 world_sim | 不加（世界模型方向手撕题少） | 0 | - |
| vol-5 engineering | 新增 §F "数值稳定与工程" | 8 题 | N1-N4, N6-N8, P2 |
| vol-6 legged_control | 不加（运控偏控制理论） | 0 | - |
| vol-7 perception_nav | 新增 §E "手撕感知/导航" | 7 题 | V6-V9, V10, IL8, T10 |

- **优点**: 题目主题更紧贴现有卷主题，用户复习现有卷时顺手过手撕；不破坏现有册结构
- **缺点**: LeetCode + 系统设计两类没地方放（不属于任何现有主题卷）；总插入题数 43 题，每卷新增 ~7 题，HTML 重新审计 + 渲染成本高
- **推荐场景**: 用户对"手撕代码 = 主题知识的实操检验"理解度高，倾向"按主题学习"

### Option B: 新开 vol-8 "手撕专题"

| §节 | 题数 | 主题 |
|---|---|---|
| §A Transformer 手撕 | 14 题 | T1-T14 |
| §B RL 手撕 | 11 题 | R1-R11 |
| §C IL/VLA 手撕 | 8 题 | IL1-IL8 |
| §D 视觉/感知/几何 手撕 | 10 题 | V1-V10 |
| §E 数值稳定与工程 | 8 题 | N1-N8 |
| §F 概率统计基础 | 5 题 | P1-P5 |
| §G LeetCode 高频精选 | 30 题 | L1-L30 |
| §H 系统设计 | 10 题 | S1-S10 |

**总计**: 96 题，约 2000-2500 行 markdown

- **优点**: 完整自包含，专门刷"手撕"时一卷打通；与现有 7 卷"概念+答案"形成清晰二元结构
- **缺点**: 单卷过长（接近 vol-3 的 2 倍），用户上线复习一次性看不完；index.html 卡片要重排
- **推荐场景**: 用户偏好"分类备战"——主题理论(vol 1-7) + 手撕(vol-8)清晰分流

### Option C: 混合（**作者推荐**）

1. **手撕代码题（A-A6 共 56 题）→ 插现有卷的"项目拷打"或"实战"段**
   - 像 Option A 那样按主题分流到 vol-1/2/3/5/7
   - **关键调整**: 每个手撕题答案体只写"考察点 + 关键实现要点 + 易错"（**仍 ≤350 字**），**不写完整代码**——保持"面试题库"风格不变；用户想看代码自己点链接
   - vol-1 +10、vol-2 +10、vol-3 +8、vol-5 +8、vol-7 +7 = +43 题
2. **LeetCode 高频 30 题 + 系统设计 10 题 → 新开 vol-8 "工程通识"**
   - LeetCode 用一句话题意 + 频次 + 关键思路（不贴完整代码）
   - 系统设计用现有题库一样的"答+易错"格式
   - vol-8 约 40 题，比现有任一卷都精简

**为什么推 Option C**:
- **风格一致**: 题库用户偏好的"≤350 字概念+答案"格式不被破坏。手撕代码塞进现有卷不需要新增公式推导/代码块（保持 CLAUDE.md §0 的项目定位）
- **LeetCode 单独成卷符合直觉**: leetcode 题型本来就是"通用八股"，与具体技术主题正交；机器人岗用户准备 leetcode 跟 cv/nlp 岗用户没本质区别
- **系统设计放在 vol-8 与 leetcode 同册合理**: 都是"通用工程能力"考察，不限于具身领域；与 vol-5 engineering（更偏机器人工程，如部署/数据/sim2real）天然区分
- **工作量适中**: vol-8 约 40 题（含 leetcode 30+设计 10），比 vol-3 (53 题/1159 行) 短，跨模型审查 5-6 轮可控；插入现有卷只增 ~7 题/卷，不需要全卷重审

**实施建议时间表**:
- Day 1: vol-1 / vol-2 / vol-3 三卷各加 8-10 题（仍走 Phase 0-7 流程）
- Day 2: vol-5 / vol-7 两卷各加 7-8 题
- Day 3: 新开 vol-8 完整跑一遍
- Day 4: 主册 index.html 更新（vol-8 入口卡片 + Top N 增加新题）

---

## 5. 来源清单（按平台）

### 5.1 牛客网（面经 / 八股）
- [大模型算法工程师八股仓库](https://www.nowcoder.com/feed/main/detail/7c19d5b025334e88a8a6b79f4debefea)
- [字节多模态大模型面经](https://www.nowcoder.com/feed/main/detail/ce451a35177e487f916fe34374ad9df8)
- [深度学习面经-Attention/Transformer](https://www.nowcoder.com/discuss/513538091714404352)
- [算法岗常考题(八)Transformer](https://www.nowcoder.com/discuss/473903838680875008)
- [CV感知算法面试手撕题(一)](https://www.nowcoder.com/discuss/756981253181542400)
- [CV感知算法面试手撕题(三)](https://www.nowcoder.com/discuss/768600194819559424)
- [Transformer高频考点(一)](https://www.nowcoder.com/discuss/774055884853997568)
- [自动驾驶规划算法面经汇总](https://www.nowcoder.com/discuss/518560994684084224)
- [自动驾驶规划与决策面试总结](https://www.nowcoder.com/discuss/353158840235532288)
- [小米具身智能算法岗实习一面](https://www.nowcoder.com/feed/main/detail/e675e68f0ac142f48f696e268f9b583d)
- [小米北京自动驾驶系统开发实习](https://www.nowcoder.com/discuss/435863716467400704)
- [小米大模型算法实习二面](https://www.nowcoder.com/feed/main/detail/6e4fa662d94d4023b8ca88bd30e4e4a7)
- [蔚来 智能装备开发实习 面经](https://www.nowcoder.com/feed/main/detail/906e9d7003bb444a8e7b2dea4f345c13)
- [WeRide 2026 校招实习面经指南](https://www.nowcoder.com/feed/main/detail/d893d6b752ec4249beda5c99c7469209)
- [拼多多 2025 提前批面经](https://www.nowcoder.com/discuss/782301095736504320)
- [谈谈 VLA 自动驾驶框架](https://www.nowcoder.com/discuss/790206571551735808)
- [面啥挂啥的算法/机器学习面经](https://www.nowcoder.com/discuss/36815)
- [Leetcode 高频 DFS/BFS](https://blog.nowcoder.net/n/dd4bbbfc1c634db7a97134fe0d110e35)
- [125 道高频题](https://blog.nowcoder.net/n/d0636b59436c48a8afc51bd6a4bc4b86)
- [牛客 TOP101](https://www.nowcoder.com/exam/oj)

### 5.2 知乎（深度技术博客）
- [手撕 Transformer 面试](https://zhuanlan.zhihu.com/p/16064503712)
- [手撕 multi-head attention](https://zhuanlan.zhihu.com/p/13166045093)
- [Multi-Head Attention 代码实现](https://zhuanlan.zhihu.com/p/657619260)
- [Attention 面经干货](https://zhuanlan.zhihu.com/p/379033238)
- [Transformer 八股看完手撕面试官](https://zhuanlan.zhihu.com/p/1899071091349127409)
- [LLM 手撕代码合集(2)](https://zhuanlan.zhihu.com/p/2022092403721479678)
- [RoPE 详解与代码](https://zhuanlan.zhihu.com/p/684666015)
- [RoPE 深度解析](https://zhuanlan.zhihu.com/p/645263524)
- [Flash Attention 全解析](https://zhuanlan.zhihu.com/p/1953761827025584899)
- [Flash Attention 简单解读](https://zhuanlan.zhihu.com/p/655448380)
- [Flash Attention 含代码](https://zhuanlan.zhihu.com/p/676655352)
- [多模态面试 Flash Attention](https://zhuanlan.zhihu.com/p/15572641555)
- [KV cache 推理阶段](https://zhuanlan.zhihu.com/p/1965823665364014863)
- [看图学 KV Cache](https://zhuanlan.zhihu.com/p/662498827)
- [华为 KV cache 面试](https://zhuanlan.zhihu.com/p/18748221598)
- [LayerNorm vs BN vs RMSNorm](https://zhuanlan.zhihu.com/p/694909672)
- [BN/LN/IN/GN 实现](https://zhuanlan.zhihu.com/p/630720228)
- [PPO 强化学习面试手撕](https://zhuanlan.zhihu.com/p/2012930266168046909)
- [DPO step by step](https://zhuanlan.zhihu.com/p/692991235)
- [DPO 原理 手撕公式](https://zhuanlan.zhihu.com/p/1904847799243220274)
- [GAE 优势函数](https://zhuanlan.zhihu.com/p/10343932079)
- [Importance Sampling](https://zhuanlan.zhihu.com/p/36816898)
- [RL 重要性采样](https://zhuanlan.zhihu.com/p/669378380)
- [Policy Gradient 推导](https://zhuanlan.zhihu.com/p/1962527453709857590)
- [RL 基础四 PG](https://zhuanlan.zhihu.com/p/31278940)
- [Diffusion Model 详解](https://zhuanlan.zhihu.com/p/638442430)
- [手撕 VAE 一篇就够](https://zhuanlan.zhihu.com/p/719968411)
- [VAE ELBO 推导](https://zhuanlan.zhihu.com/p/138123592)
- [一文解释 VAE+ELBO](https://zhuanlan.zhihu.com/p/575984592)
- [VAE 中的 KL 散度](https://zhuanlan.zhihu.com/p/345095899)
- [手撕 Softmax Cross-entropy](https://zhuanlan.zhihu.com/p/9231669809)
- [秋招手撕代码合集（非 leetcode）](https://zhuanlan.zhihu.com/p/666849216)
- [各公司秋招手撕 / 笔试](https://zhuanlan.zhihu.com/p/666648780)
- [校招算法岗面经总结](https://zhuanlan.zhihu.com/p/620290527)
- [字节算法面试高频汇总](https://zhuanlan.zhihu.com/p/336117700)
- [字节高频算法题库汇总](https://zhuanlan.zhihu.com/p/365332969)
- [面试高频分类刷题总结](https://zhuanlan.zhihu.com/p/349940945)
- [BFS 总结](https://zhuanlan.zhihu.com/p/62884729)
- [ACT 解读](https://zhuanlan.zhihu.com/p/676520960)
- [ACT 论文笔记](https://zhuanlan.zhihu.com/p/1981673900472557572)
- [OpenVLA 解读](https://zhuanlan.zhihu.com/p/1925499475973116151)
- [VLA 求职路线](https://zhuanlan.zhihu.com/p/1955704833358160798)
- [VLA 模型综述](https://zhuanlan.zhihu.com/p/1907961280112877856)
- [VLA 推理加速](https://zhuanlan.zhihu.com/p/2022992211273348705)
- [VLA 高效化路径](https://zhuanlan.zhihu.com/p/1964132820160053654)
- [GRPO 核心技术](https://zhuanlan.zhihu.com/p/25985130568)
- [GRPO 解析](https://zhuanlan.zhihu.com/p/27454498505)
- [GRPO 详解](https://zhuanlan.zhihu.com/p/21046265072)
- [ROS 具身入门](https://zhuanlan.zhihu.com/p/20329244481)
- [TD3 / SAC / DDPG 对比](https://www.zhihu.com/question/6699179413)
- [稚晖君公司工资和招人标准](https://zhuanlan.zhihu.com/p/1971861765001314603)
- [具身招聘帖](https://zhuanlan.zhihu.com/p/1915789829809108777)
- [秋招总结自动驾驶机器人](https://zhuanlan.zhihu.com/p/667687639)
- [自动驾驶感知秋招面经](https://zhuanlan.zhihu.com/p/674917858)
- [自动驾驶感知面经20公司](https://zhuanlan.zhihu.com/p/656952371)
- [自动驾驶规控算法面经](https://zhuanlan.zhihu.com/p/501589139)
- [蔚来汽车算法一面](https://zhuanlan.zhihu.com/p/381315220)
- [2024 自动驾驶秋招面经](https://zhuanlan.zhihu.com/p/647137052)
- [大厂面经记录一](https://zhuanlan.zhihu.com/p/4583793442)
- [BPE WordPiece Unigram](https://zhuanlan.zhihu.com/p/620508648)

### 5.3 CSDN
- [手撕 self-attention 学习路线](https://blog.csdn.net/qq_44949041/article/details/128087174)
- [Transformer self-attention PyTorch](https://blog.csdn.net/aidanmo/article/details/121445183)
- [Multi-head NumPy + PyTorch](https://blog.csdn.net/weixin_43114209/article/details/150490531)
- [RoPE 代码详解](https://blog.csdn.net/BIT_666/article/details/133696553)
- [Rotary Position Embedding](https://blog.csdn.net/weixin_43646592/article/details/130924280)
- [神经网络必面 IOU NMS BN](https://blog.csdn.net/weixin_44398263/article/details/123407466)
- [手撕题 (二) AI 深度学习](https://blog.csdn.net/qq_39444290/article/details/137288088)
- [手撕 IOU NMS](https://blog.csdn.net/Jiangnan_Cai/article/details/132662191)
- [PPO loss 推导四个模型](https://blog.csdn.net/weixin_36378508/article/details/152085552)
- [PPO 系列 5 实现](https://blog.csdn.net/smartcat2010/article/details/144467884)
- [DPO Loss 解析](https://blog.csdn.net/weixin_36378508/article/details/152092226)
- [GAE 优势函数](https://blog.csdn.net/shizheng_Li/article/details/144436495)
- [Policy Gradient 详细推导](https://blog.csdn.net/november_chopin/article/details/108032626)
- [Q-learning 详解](https://blog.csdn.net/qq_30615903/article/details/80739243)
- [Double/Dueling DQN](https://blog.csdn.net/sinat_52032317/article/details/134095353)
- [DDPM 系列1](https://blog.csdn.net/g11d111/article/details/131326934)
- [Diffusion Policy 解读](https://blog.csdn.net/zhaoliang38/article/details/139135283)
- [VAE 超详解](https://blog.csdn.net/m0_56942491/article/details/136265500)
- [VAE 重参数化](https://blog.csdn.net/weixin_51176105/article/details/134741436)
- [reparameterization trick](https://blog.csdn.net/shizheng_Li/article/details/145996540)
- [PointNet 论文笔记](https://blog.csdn.net/hongbin_xu/article/details/84638109)
- [T-Net 旋转网络](https://blog.csdn.net/weixin_45641915/article/details/132340216)
- [ResNet 手动实现](https://blog.csdn.net/bu_fo/article/details/109203360)
- [ResNet 原理与代码](https://blog.csdn.net/m0_74055982/article/details/137927190)
- [ViT patch embedding 详解](https://blog.csdn.net/fulva/article/details/121045938)
- [DeepSeek-R1 GRPO 详解](https://deepseek.csdn.net/67c3fefc6670175f992e306b.html)
- [模仿学习路线](https://blog.csdn.net/Yong_Qi2015/article/details/148621978)
- [行为克隆](https://blog.csdn.net/weixin_45116099/article/details/135663133)
- [LayerNorm/RMSNorm](https://blog.csdn.net/m0_37586991/article/details/149251473)
- [BN/LN/RMS 详解](https://blog.csdn.net/wxc971231/article/details/139925707)
- [warmup + cosine](https://blog.csdn.net/sinat_41667032/article/details/116454406)
- [小米感知算法实习一面](https://blog.csdn.net/litterfinger/article/details/142152892)
- [小米大模型算法面试](https://blog.csdn.net/2401_85378759/article/details/141925323)
- [大模型公司面经](https://blog.csdn.net/m0_63171455/article/details/140654570)
- [自动驾驶规控面经](https://blog.csdn.net/CV_Autobot/article/details/133566058)
- [Attention Mask](https://blog.csdn.net/weixin_43889416/article/details/115301178)
- [Casual Mask](https://blog.csdn.net/weixin_43833206/article/details/144566539)
- [PyTorch 训练循环详解](https://blog.csdn.net/vvilkim/article/details/148480202)
- [Q-Learning Python from scratch](https://blog.csdn.net/2501_91624122/article/details/147916483)
- [Adam 详解](https://blog.csdn.net/weixin_39228381/article/details/108548413)
- [大模型 BPE 实践](https://blog.csdn.net/2401_85325397/article/details/139472993)
- [LLM Cut Cross Entropy](https://zhuanlan.zhihu.com/p/13548439339)
- [Softmax LogSumExp](https://blog.csdn.net/qq_27590277/article/details/125568062)
- [A*/Dijkstra 路径规划](https://blog.csdn.net/weixin_46039719/article/details/128585574)

### 5.4 博客园
- [楚千羽 手撕题(一)](https://www.cnblogs.com/chuqianyu/p/18048501)
- [楚千羽 手撕题(二)](https://www.cnblogs.com/chuqianyu/p/18062881)
- [手撕字节面试算法题](https://github.com/lewiscrow/WorkHardAndFindJob/blob/master/%E5%A4%8D%E4%B9%A0/%E9%9D%A2%E8%AF%95/%E6%89%8B%E6%92%95%E5%AD%97%E8%8A%82%E8%B7%B3%E5%8A%A8%E9%9D%A2%E8%AF%95%E6%97%B6%E5%87%BA%E7%8E%B0%E8%BF%87%E7%9A%84%E7%AE%97%E6%B3%95%E9%A2%98.md)
- [DDPM 深度指南](https://www.cnblogs.com/zhangdoudou/p/18537276)
- [Diffusion DDPM 数学代码](https://www.cnblogs.com/zhangxianrong/p/18326866)
- [Q-learning Python 实现](https://www.cnblogs.com/hhh5460/p/10134018.html)
- [GAE-87 大模型](https://www.cnblogs.com/cavalier-chen/p/18988014)
- [zero_grad backward 用法](https://www.cnblogs.com/h694879357/p/15855281.html)
- [DDPG TD3 SAC 调参](https://www.cnblogs.com/ting1/p/16984892.html)
- [Transformer Mask 机制](https://www.cnblogs.com/wevolf/p/12484972.html)
- [Multi-Head Attention 核心机制](https://www.cnblogs.com/1314520xh/p/18881721)
- [A* 算法（机器人）](https://www.cnblogs.com/QiQi-Robotics/p/14931545.html)
- [Dijkstra（机器人）](https://www.cnblogs.com/QiQi-Robotics/p/14927660.html)
- [ROS2 MoveIt! 机械臂](https://www.cnblogs.com/TS86/p/18888159)

### 5.5 GitHub 仓库
- [zxuu/Self-Attention](https://github.com/zxuu/Self-Attention)
- [Shaka-Labs/ACT](https://github.com/Shaka-Labs/ACT)
- [datawhalechina/base-llm RLHF](https://github.com/datawhalechina/base-llm/blob/main/docs/chapter12/01_RLHF.md)
- [amusi/AI-Job-Notes](https://github.com/amusi/AI-Job-Notes)
- [amusi/Deep-Learning-Interview-Book 面试经验](https://github.com/amusi/Deep-Learning-Interview-Book/blob/master/docs/%E9%9D%A2%E8%AF%95%E7%BB%8F%E9%AA%8C.md)
- [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note)
- [km1994/LLMs_interview_notes](https://github.com/km1994/LLMs_interview_notes)
- [lcylmhlcy/Awesome-algorithm-interview](https://github.com/lcylmhlcy/Awesome-algorithm-interview)
- [afatcoder/LeetcodeTop](https://github.com/afatcoder/LeetcodeTop)
- [youngyangyang04/leetcode-master](https://github.com/youngyangyang04/leetcode-master)
- [resumejob/Leetcode-retag](https://github.com/resumejob/Leetcode-retag)
- [YaxeZhang/Just-Code](https://github.com/YaxeZhang/Just-Code)
- [StarCycle/Awesome-Embodied-AI-Job](https://github.com/StarCycle/Awesome-Embodied-AI-Job)
- [0voice/2026 春招汇总](https://github.com/0voice/2026-Computer-Spring-Recruitment-Job-Compilation)
- [datawhalechina/hello-agents Extra01](https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra01-%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93.md)

### 5.6 英文资源（Medium / 官方文档 / Blog）
- [MLM scaled dot-product attention](https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/)
- [TensorTactics interview Q](https://www.tensortactics.com/machine-learning-interview-questions/scaled-dot-product-attention)
- [shadecoder SDPA practical guide](https://www.shadecoder.com/topics/scaled-dot-product-attention-a-comprehensive-guide-for-2025)
- [agrimpaneru SDPA PyTorch](https://agrimpaneru.com.np/blog/self-attention-pytorch/)
- [apxml LLM book](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-10-implementing-transformer-from-scratch/implementing-scaled-dot-product-attention)
- [arthurchiao 600 行 transformer](https://arthurchiao.art/blog/transformers-from-scratch-zh/)
- [TDS Diffusion from scratch](https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946/)
- [codersarts DDPM + DDIM](https://www.codersarts.com/post/how-to-build-a-diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946)
- [HF lerobot ACT](https://hugging-face.cn/docs/lerobot/act)
- [HF blog RLHF 中文](https://huggingface.co/blog/zh/rlhf)
- [chaofa DPO](https://yuanchaofa.com/post/hands-on-dpo-direct-preference-optimization)
- [hwcoder 神经网络篇](https://hwcoder.top/Manual-Coding-2)
- [hwcoder RLHF 篇](https://hwcoder.top/Manual-Coding-6)
- [labml.ai RoPE](https://nn.labml.ai/zh/transformers/rope/index.html)
- [苏剑林 RoPE](https://kexue.fm/archives/10862)
- [zwn LogSumExp](https://www.zwn2001.space/posts/Graduate-Works/DL/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-LogSumExp%E6%8A%80%E5%B7%A7/index.html)
- [snailcoder BN/LN](https://snailcoder.github.io/2024/05/01/batchnorm-and-layernorm.html)
- [interviewquery Tesla SWE](https://www.interviewquery.com/interview-guides/tesla-software-engineer)
- [aiofferly Tesla ML](https://www.aiofferly.com/career-guide/tesla-ml-interview-questions)
- [glassdoor Boston Dynamics](https://www.glassdoor.com/Interview/Boston-Dynamics-Interview-Questions-E261553.htm)
- [glassdoor BD AI Institute](https://www.glassdoor.com/Interview/Boston-Dynamics-AI-Institute-Interview-Questions-E8742645.htm)
- [emergentmind VLN topic](https://www.emergentmind.com/topics/vision-language-navigation-vln)
- [GitHub Thinking-VLN](https://github.com/YicongHong/Thinking-VLN)
- [d2l 学习率调度](https://zh.d2l.ai/chapter_optimization/lr-scheduler.html)
- [d2l ResNet](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)
- [d2l 填充步幅](https://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)
- [hrl.boyuai 动手学 RL](https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B/)
- [datawhalechina ViT](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%8D%81%E7%AB%A0/ViT%E8%A7%A3%E8%AF%BB.html)

### 5.7 一亩三分地 / 其他
- [蔚来面凉经](https://www.1point3acres.com/bbs/thread-721539-1-1.html)
- [品玩 AI 教授 AMA](类似来源仅作交叉参考)

---

## 6. 调研信心评估

### 6.1 信心高的类别（题源 ≥5，证据充分）
- ✅ Transformer 家族手撕 (T1, T2, T4, T5)：每题至少 5-10 个交叉来源（牛客 + 知乎 + CSDN + GitHub 多平台）
- ✅ RL 核心公式手撕 (R1, R4, R5)：PPO/GAE/Q-learning 是 RL 八股的必考题
- ✅ Visual primitives (V1 IoU, V2 NMS, V3 conv 输出维度)：CV 岗经典题，多年高频
- ✅ LeetCode 高频 Top-15 (L1-L15)：CodeTop 数据库公开可查
- ✅ 数值稳定题 (N1, N3, N5)：所有 ML 岗一面常问
- ✅ DPO loss 推导 (R9)：RLHF 时代的新八股，知乎/CSDN 文章 ≥10 篇

### 6.2 信心中（题源 3-5，证据基本充分但有特定背景）
- ⚠️ Diffusion Policy / ACT / OpenVLA 专项手撕 (IL2, IL3, IL4)：具身岗权威面经支持，但绝对数量不大
- ⚠️ Flash Attention / KV cache（T9, T12）：大模型岗高频，具身岗 less common 但被问到时是 L3 大题
- ⚠️ GRPO loss (R10)：DeepSeek-R1 后新八股，发酵中，频次预计上涨
- ⚠️ ML 系统设计 (S1, S2, S6)：来源主要为综述文章而非真实面经
- ⚠️ 公司专项 LeetCode 偏好：除字节有 CodeTop 数据外，其他公司面经稀薄

### 6.3 信心低（题源 1-2，建议二次调研）
- ❌ Figure AI / 1X / 银河通用 / 星海图等"独角兽"公司的具体原题：这些公司面经绝对量小，多为定性描述
- ❌ NeRF 手撕 (V10)：具身岗频次低
- ❌ Hungarian matching / DETR 手撕：偶尔出现，非主流
- ❌ Multi-sensor fusion 系统设计 (S7)、SLAM 状态估计 (S9)：自动驾驶面经多，但具身岗（操作类）频次不高

### 6.4 建议
- **直接用**: §1.1-§1.6 + §2.2 LeetCode Top-30 + §3 系统设计 Top-5——这些是面试必备
- **二次调研**: 用户面具体公司前，针对性补查那家公司近 6 个月的牛客/一亩三分地面经；具身领域公司面经稀缺时可以拿同岗位的自动驾驶面经做"近似估计"
- **不必投入**: §1.7 低频补充，等遇到再说

---

> **本笔记仅供调研参考。所有题目均基于公开面经平台调研合并，未做跨模型审查（仅当作初稿用，正式入题库前需走 Phase 3 跨模型审查流程）。**
