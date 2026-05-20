# 卷一 · 通识基础 · 高频面试题

> 中文具身智能秋招高频面试题库 · **第一卷**
> 题源：牛客 / 知乎 / CSDN / GitHub 公开面经 / 算法岗面试题汇编
> 同义题合并后 **47 题**（主表 44 题频次 ≥3 + 低频备选 3 题）

**难度分布**：L1（必会） **22** · L2（进阶） **23** · L3（顶级lab） **2**

**使用方式**：题目默认折叠，点开看答案。建议按 **L1 → L2 → L3** 顺序刷；同级内按频次从高到低。手机端原生支持。

---

## §1 深度学习基础（20 题）

> 具身算法岗一面必考区，重点在梯度问题、归一化、优化器、Transformer 架构。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×18</span> <b>Q01</b> · Transformer Self-Attention 的计算流程是什么？复杂度是多少？</summary>

**答**：输入序列 $X \in \mathbb{R}^{n \times d}$ 分别乘三个权重矩阵得到 $Q, K, V$，然后计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

步骤：① 线性投影得 $Q, K, V$；② 计算相似度矩阵 $QK^\top$（$n \times n$）并除以 $\sqrt{d_k}$ 缩放；③ softmax 归一化；④ 加权求和 $V$。

**复杂度**：时间 $O(n^2 d)$，空间 $O(n^2)$（注意力矩阵是瓶颈），序列长时计算量爆炸。多头注意力是 $h$ 个小头并行，总参数量不变。

**易错**：除以 $\sqrt{d_k}$ 不是"随便加的"——防止点积过大导致 softmax 梯度消失；去掉 scale 训练会明显变差。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×15</span> <b>Q02</b> · 梯度消失和梯度爆炸的原因是什么？各有哪些解决方法？</summary>

**答**：

**消失**：链式法则反传时，若激活函数（如 sigmoid）导数 < 1，多层连乘后梯度指数衰减 → 浅层几乎不更新。

**爆炸**：权重矩阵乘积特征值 > 1，梯度指数增长 → 更新步过大、训练发散。

**解决梯度消失**：① **换激活函数**：ReLU/Leaky ReLU 导数在正区间为 1；② **残差连接**（skip connection）提供梯度高速公路；③ **BatchNorm/LayerNorm** 归一化激活分布；④ **LSTM 门机制**（序列模型）。

**解决梯度爆炸**：① **梯度裁剪**（Gradient Clipping）：若 $\|g\| > \tau$ 则 $g \leftarrow g \cdot \tau / \|g\|$；② **权重初始化**（Xavier/He）；③ **BatchNorm**。

**易错**：ReLU 有 dying ReLU 问题（负值区梯度永久为 0），大批量 + 大学习率下死亡神经元显著，Leaky ReLU / ELU 可缓解。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×14</span> <b>Q03</b> · BatchNorm 和 LayerNorm 的区别？Transformer 为什么用 LayerNorm？</summary>

**答**：

- **BatchNorm（BN）**：沿 batch 维归一化，即对所有样本的同一特征位置计算均值/方差；训练时用 batch 统计量，推理时用 running 均值/方差（全局估计）。**适合 CV/CNN**，batch 大时稳定。
- **LayerNorm（LN）**：沿特征维（同一样本内所有通道）归一化，每个样本独立计算，**不依赖 batch size**，训练/推理行为一致。

**Transformer 用 LN 的原因**：① 序列长度不等，batch 统计不稳定；② NLP 任务 batch 小时 BN 方差估计噪声大；③ 推理时单条序列（batch=1）BN 退化，LN 完全不受影响。

**易错**：BN 在推理期用的是训练期积累的 running mean/var，不是当前 batch 统计量；小 batch 时 BN 抖动大、LN 更稳。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q04</b> · Adam 优化器原理是什么？和 SGD、AdamW 的区别？</summary>

**答**：**Adam** = 一阶矩（梯度 EMA）+ 二阶矩（梯度平方 EMA），各参数有独立自适应学习率；更新：$\theta \leftarrow \theta - \alpha \hat{m}/(\sqrt{\hat{v}}+\epsilon)$（偏差修正后），$\beta_1=0.9, \beta_2=0.999$。

**vs SGD**：SGD 用统一学习率，收敛稳但慢；Adam 自适应、前期快，但大规模视觉任务泛化有时弱于 SGD+momentum。

**vs AdamW**：Adam 加 L2 正则后 weight decay 强度被 $\hat{v}$ 缩放（不均匀）；AdamW 把 weight decay 直接加在权重更新上（解耦），效果一致且可控——VLA 微调几乎都用 AdamW。

**易错**：Adam + L2 ≠ AdamW，两者在自适应学习率下行为不同；LLM/VLA 微调必须用 AdamW，否则 weight decay 强度因参数而异。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q05</b> · Dropout 的原理是什么？训练和推理阶段有何不同？</summary>

**答**：**训练时**：每个神经元以概率 $p$ 被随机置 0（"丢弃"），保留的神经元输出乘以 $1/(1-p)$ 缩放（Inverted Dropout），使期望与完整网络一致。

**推理时**：所有神经元保留，**不做 Dropout**，直接用完整网络输出。（如果训练时没用 Inverted Dropout，推理时需将输出乘以 $(1-p)$。）

**作用**：① 减少神经元共适应（co-adaptation），相当于隐式集成多个子网络；② 正则化，防止过拟合。

**易错**：model.eval() 会自动关 Dropout；忘调会导致推理时随机性——**这是 PyTorch 面试高频陷阱**。BatchNorm 在 eval() 也切换到 running stats，两者必须同时切换。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×11</span> <b>Q06</b> · ResNet 的残差连接解决了什么问题？为什么不是直接加深网络？</summary>

**答**：**问题背景**：直接堆叠深层网络（如 56 层）训练误差反而高于浅层（20 层）——不是过拟合，是**网络退化**（degradation）：优化器难以找到恒等映射。

**残差连接**：$H(x) = F(x) + x$，让网络学习残差 $F(x) = H(x) - x$ 而非直接学 $H(x)$。当最优映射接近恒等时，$F(x) \approx 0$ 比让所有层学恒等映射容易得多。

**附加好处**：① Skip connection 为梯度提供直通路径（等效更短的梯度路径），缓解梯度消失；② 实现了深度 scalable（ResNet-50/101/152 都可用）；③ 残差块输出方差较稳定，训练更容易。

**易错**：残差连接不是"自动防止过拟合"，过拟合靠 Dropout/L2/数据增强；残差解决的是**优化难度**（欠拟合+退化），不是泛化。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q07</b> · 交叉熵损失和 MSE 损失分别适合什么场景？用错了会怎样？</summary>

**答**：

- **交叉熵（CE）**：$L = -\sum_c y_c \log \hat{p}_c$；适合**分类**任务，与 softmax/sigmoid 配合，梯度形式简洁（$\hat{p}-y$），不会在输出层饱和时梯度消失。
- **MSE**：$L = \|y - \hat{y}\|^2$；适合**回归**任务（预测连续值，如关节角度、末端坐标）；与 sigmoid 配合时，当 $\hat{y}$ 饱和（近 0/1）梯度会消失。

**用错后果**：
- 分类用 MSE：梯度消失严重（sigmoid 饱和区），收敛慢，概率解释也不对；
- 回归用 CE：值域不匹配，CE 假设输出是概率，连续回归输出 > 1 时 log 无意义。

**延伸**：机器人连续动作回归常用 **Huber Loss**（MSE 和 MAE 的拼接），对异常值（outlier）更鲁棒。

**易错**：CE 中 $\log 0$ 需要 clip，PyTorch 的 `CrossEntropyLoss` 内部已处理，不用手动加 $\epsilon$。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q08</b> · Transformer 的位置编码（Positional Encoding）为什么必要？常见方案有哪些？</summary>

**答**：Self-Attention 的计算对序列顺序**天然无感**（permutation equivariant）——打乱 token 顺序，attention 输出值相同（只是对应位置互换）。对于语言/时序，顺序信息极关键，必须额外注入。

**方案对比**：
| 方案 | 原理 | 优缺点 |
|---|---|---|
| 正弦位置编码（原始 Transformer） | 固定公式 $\sin/\cos$ | 无需训练，可外推到更长序列；但绝对位置，相对位置弱 |
| 可学习位置嵌入（BERT/GPT） | 每个位置有可训练向量 | 简单有效，不可外推（超训练长度退化） |
| RoPE（LLaMA / VLA） | 旋转矩阵编码**相对**位置 | 天然外推、相对位置敏感，目前 LLM 主流 |
| ALiBi | 在 attention score 上加线性偏置 | 极简、外推好，速度快 |

**易错**：原始 Transformer 和 BERT 的位置编码方案不同；RoPE 不是加在 embedding 上，而是在 Q/K 乘以旋转矩阵。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q09</b> · 过拟合的判断标准是什么？有哪些对策？数据少时首选什么？</summary>

**答**：**判断**：训练 loss 持续下降但验证 loss 不降或反升；Train/Val 准确率差距持续扩大。

**对策（按有效性排序，数据少时）**：

1. **数据增强**：旋转/翻转/裁剪/颜色抖动（CV）或轨迹噪声/镜像（机器人）—— **首选**，增效显著
2. **Early Stopping**：监控 val loss，停在最低点
3. **Dropout / L2 正则**：限制权重大小或神经元依赖
4. **降低模型容量**：层数/隐层宽度减半（不够数据撑不住大模型）
5. **迁移学习 + 冻结大部分层**：预训练权重 + 只训练 head（少样本首选）

**易错**：L2 和 Weight Decay 在 Adam 下**不等价**（见 AdamW）；简单堆 Dropout 不如先看数据是否充分/分布是否覆盖测试集。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q10</b> · 学习率调度（LR Schedule）有哪些常用策略？Warmup 为什么重要？</summary>

**答**：

- **Cosine Annealing**：LR 按余弦从 $\eta_{max}$ 降到 $\eta_{min}$，后期细粒度收敛；最常见。
- **Step Decay**：每 N epoch 乘以 decay 因子（如 0.1）；简单但台阶式，可能跳过最优。
- **Linear Decay**：线性下降；简单，GPT 类常用。
- **Warmup + Cosine**（主流组合）：先从 0 线性升到 $\eta_{max}$（Warmup），再 cosine 衰减。

**Warmup 为什么重要**：训练初期权重随机，梯度方向不稳；大学习率直接更新会破坏预训练权重（fine-tune 场景）或让 Adam 的二阶矩估计（$v_t$）在统计不稳时误导更新方向。Warmup 让模型先"摸清地形"再大步更新。

**易错**：Warmup 步数一般是总 steps 的 1-5%；太长 warmup 浪费训练；微调大模型（如 VLA）几乎必须用 warmup，否则容易灾难性遗忘。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×9</span> <b>Q11</b> · 卷积神经网络中感受野是什么？如何增大感受野？</summary>

**答**：**感受野**：输出特征图上一个像素对应输入图像的区域大小。$k \times k$ 卷积单层感受野 = $k$；L 层叠加后理论感受野 = $1 + L(k-1)$（无 pooling）。

**增大感受野的方法**：
- **叠加层数**：最简单，但参数量增大
- **Pooling / Stride > 1**：降采样同时扩感受野，但丢失空间精度
- **空洞卷积（Dilated Convolution）**：在 $k \times k$ 卷积中插入 dilation $r$，等效感受野 $k + (k-1)(r-1)$，参数量不变——**感知野扩大的首选**
- **大核卷积**：如 7×7、11×11；效果直接但参数多

**机器人应用**：视觉策略网络需要大感受野捕捉全局位置关系（末端位姿 vs 目标位置），常用 ViT（全局感受野）或多尺度 CNN。

**易错**：有效感受野（ERF）比理论感受野小得多——高斯分布中心权重大，边缘贡献极低；不要以为理论感受野就是真正利用的区域。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q12</b> · 权重初始化为什么重要？Xavier 和 He 初始化各适合什么激活函数？</summary>

**答**：**为什么重要**：全 0 初始化 → 所有神经元梯度相同（对称性破坏失败）；过大初始值 → 激活饱和+梯度消失；过小 → 梯度消失+信号衰减。好的初始化让各层激活和梯度的方差在正向/反向传播中保持稳定。

**Xavier（Glorot）初始化**：$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$；假设激活函数**线性（near zero）**，适合 **tanh / sigmoid**（在零点近似线性）。

**He（Kaiming）初始化**：$\text{Var}(W) = \frac{2}{n_{in}}$；考虑 ReLU 把一半神经元置 0，方差要加倍补偿，适合 **ReLU / Leaky ReLU**。

**易错**：PyTorch 默认初始化对大多数层已用 Kaiming Uniform，但自定义层/Embedding 等要手动检查；使用 SELU 激活时有专门的 LeCun 初始化。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q13</b> · 什么是 L1 正则和 L2 正则？它们对权重的影响有何不同？</summary>

**答**：

- **L2 正则**（权重衰减）：$L_{total} = L + \lambda \|w\|_2^2$；梯度 $\partial/\partial w = \partial L/\partial w + 2\lambda w$，效果是每次更新时权重**等比例收缩**（权重趋于小而稠密）。
- **L1 正则**：$L_{total} = L + \lambda \|w\|_1$；梯度为 $\pm\lambda$（恒定），将小权重推为精确 **0**（稀疏权重），等效特征选择。

**实践差异**：L2 在深度学习中更常用（weight decay）；L1 在稀疏特征选择（如 Lasso 回归）中有用；Elastic Net = L1 + L2 组合。

**机器人策略网络**：通常用 L2/weight decay + Dropout，L1 导致的权重稀疏不利于连续动作的平滑输出。

**易错**：Adam 下直接加 L2 不等于 weight decay（二阶矩缩放了梯度），要用 AdamW 才是真正解耦的 weight decay。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q14</b> · 为什么 Softmax 输出可以当作概率？数值稳定性如何保证？</summary>

**答**：**概率解释**：Softmax 把实值 logit 向量 $z$ 映射到 $(0,1)$ 且和为 1：

$$\hat{p}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

连接到最大熵原理：当约束为期望值已知时，Softmax 是最大熵分布；结合交叉熵损失等价于最大化对数似然（MLE），有良好的概率理论支撑。

**数值稳定**：直接计算 $e^{z_i}$ 会在 $z_i$ 很大时溢出 → 实践中用 **Stable Softmax**：先减最大值 $c = \max_j z_j$，即 $\hat{p}_i = e^{z_i - c} / \sum_j e^{z_j - c}$，数学等价但数值无溢出。

**易错**：PyTorch 的 `F.cross_entropy` 内部等价于 LogSoftmax + NLLLoss（数值稳定）。**不要**先手动 softmax 再喂给 NLLLoss——NLLLoss 直接把输入当 log-prob 取负求均值，若输入是概率（非 log 域），loss 和梯度都会错误。正确做法：直接 `F.cross_entropy(logits, labels)`，不需要手动 softmax。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q15</b> · 什么是混合精度训练（Mixed Precision Training）？为什么机器人训练常用？</summary>

**答**：**混合精度**：权重存 FP32，前向/反向计算用 FP16（或 BF16），梯度更新在 FP32 主权重上进行。

**关键技术**：① **Loss Scaling**：FP16 精度低，微小梯度下溢 → 先把 loss 乘以大系数，计算梯度后再除回；② **Master Weights**：维护 FP32 副本保证更新精度。

**好处**：① FP16 显存占用约为 FP32 的 1/2，允许更大 batch / 更多序列帧；② Tensor Core 加速，速度提升 2-3×；③ VLA 模型（7B+）几乎必须用混合精度才能在单机跑。

**BF16 vs FP16**：BF16 指数位更宽（不易溢出），训练稳定性优于 FP16，A100/H100 首选 BF16，不需 loss scaling。

**易错**：FP16 推理不等于 INT8 量化；量化是进一步压缩权重到整数，精度/速度权衡更极端。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q16</b> · Flash Attention 解决了什么问题？核心思路是什么？</summary>

**答**：**问题**：标准 Self-Attention 的 $O(n^2)$ 注意力矩阵（$n \times n$）必须写回 HBM（高带宽显存），内存带宽是瓶颈，而不是计算本身。对长序列（n=4096+）显存 OOM 且极慢。

**核心思路（IO-aware）**：① **Tiling**：把 $Q, K, V$ 分块，每块完整在 SRAM（共享内存，快但小）内完成 attention 计算，避免反复读写 HBM；② **Online softmax**：用 log-sum-exp trick 做数值稳定的分块 softmax，无需等整行算完；③ 不存中间 $n \times n$ 矩阵（只存最终输出和反传所需的 $\log$-sum），显存从 $O(n^2)$ 降到 $O(n)$。

**结果**：速度 2-4×，显存 5-20× 节省，数学等价（无近似）。Flash Attention 2/3 进一步优化并行度。

**易错**：Flash Attention 是精确计算，不是近似——不同于 Linformer/Performer 的低秩近似；结果与标准实现完全相同。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q17</b> · KV Cache 是什么？为什么推理时有它、训练时没有？</summary>

**答**：**KV Cache**：自回归解码时，每步新增 1 个 token，其 $K, V$ 投影结果被缓存；后续步骤直接复用，消除对已生成前缀的重复 K/V 投影计算。

**为什么推理有、训练无**：训练用 Teacher Forcing 并行前向（整条序列一次性算），无逐步生成场景，历史 K/V 无需复用。推理逐步自回归，历史固定可缓存。

**代价**：KV Cache 显存随序列长度和层数线性增长（每层每头独立存储），长上下文时成为显存瓶颈 → 引出 MQA / GQA（多 Query 共享少量 KV Head）。

**易错**：KV Cache 缓存的是 K 和 V，不是 attention 权重；每步 attention 仍要 attend 全部历史（计算量正比于前缀长度），消除的是前缀的 K/V 重复投影，不是 attention 计算本身。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q18</b> · ViT（Vision Transformer）与 CNN 的本质区别是什么？在机器人视觉中各有什么优劣？</summary>

**答**：

| | ViT | CNN |
|---|---|---|
| 感受野 | 全局（每层 token 互相 attend） | 局部（逐层扩大） |
| 归纳偏置 | 弱（无平移等变性假设） | 强（平移等变、局部性） |
| 参数效率 | 数据少时差（需大量数据弥补归纳偏置缺失） | 数据少时好 |
| 可扩展性 | 强（patch 数量可变、序列建模自然） | 弱（固定感受野，改分辨率麻烦） |

**机器人视觉**：
- **ViT 优势**：抓全局空间关系（末端 vs 目标位置远处相关），适合 VLA（视觉 + 语言 + 动作联合建模）；SigLIP / DINOv2 预训练的 ViT 在 OpenVLA 表现超越 CLIP-only；
- **CNN 优势**：小数据集（真机 50 demos）、实时推理（低延迟）；局部纹理/接触检测（细粒度操作）。
- 实务：VLA 主流用 ViT 作视觉 backbone + Transformer LM；实时控制头（非语言部分）可用小 CNN。

**易错**：DINOv2 虽是 ViT，但自监督预训练让它比监督 ViT 更有空间几何感；不要混淆预训练方式和架构选择。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×5</span> <b>Q19</b> · 训练集/验证集/测试集的划分原则是什么？数据泄露是什么？</summary>

**答**：**划分原则**：
- 训练集：用于学习参数
- 验证集（dev set）：用于超参调整（不能反复用来"选"模型后再报它的指标）
- 测试集：最终一次性评估，**严禁在调参中使用**

**常见比例**：数据充足时 60/20/20；数据少时用 k-fold（k=5/10）让每条数据都当过验证。

**数据泄露（Data Leakage）**：测试/验证集的信息提前"流入"训练，导致指标虚高。常见形式：① 归一化用全量（含测试集）的均值/方差；② 时序数据随机打乱（未来信息混入历史）；③ 图像增强用了测试集样本。

**机器人场景**：演示数据来自同一任务实例时，必须按**任务/场景**划分而非随机划帧（同场景的帧高度相关，随机划分严重高估泛化）。

**易错**：sklearn 的 StandardScaler 必须在训练集 `.fit()`，测试集只 `.transform()`，用 `.fit_transform()` 对全量是数据泄露。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q20</b> · 什么是 Knowledge Distillation（知识蒸馏）？在机器人策略压缩中怎么用？</summary>

**答**：**知识蒸馏**：用大模型（Teacher）的 soft 输出（softmax 概率分布，含类间相似性信息）作为小模型（Student）的训练目标，而非 one-hot 标签。温度参数 $T > 1$ 让分布变软，信息量更多。

**损失**：$L = \alpha L_{CE}(y, \hat{p}) + (1-\alpha) T^2 L_{KL}(\sigma(z_T/T), \sigma(z_S/T))$

**机器人策略压缩**：① VLA（7B）蒸馏到小策略网络（100M-300M），保留连续动作的分布知识；② 用大 diffusion policy 的多模态输出作为 soft target 训练快速 MLP 策略；③ 离线 RL 中 behavior cloning + teacher soft label 的组合。

**易错**：蒸馏不是"把大模型权重复制给小模型"；小模型容量不足时蒸馏也无法弥补；温度 $T$ 太小退化回 one-hot，太大噪声过多。

</details>

---

## §2 强化学习基础（14 题）

> 具身算法岗 RL 必考区，重点在 MDP / Bellman / TD / 策略梯度 / Actor-Critic。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×14</span> <b>Q21</b> · 什么是 MDP（马尔可夫决策过程）？五元组分别是什么？</summary>

**答**：MDP 是强化学习的数学框架：$\langle S, A, P, R, \gamma \rangle$

- $S$：状态空间（机器人关节角度、视觉观测等）
- $A$：动作空间（连续关节速度/位置，或离散动作）
- $P(s' \mid s, a)$：状态转移概率（环境动力学）
- $R(s, a, s')$：即时奖励函数
- $\gamma \in [0,1)$：折扣因子，权衡即时 vs 长期奖励

**Markov 性质**：$P(s_{t+1} \mid s_t, a_t, s_{<t}) = P(s_{t+1} \mid s_t, a_t)$，下一状态只依赖当前状态和动作，与历史无关。

**机器人现实**：几乎都是 POMDP（观测不完整），但以 MDP 作近似或用历史帧 stack 来还原 Markov 性。

**易错**：Markov 性是对**状态**的假设，不是对**观测**的假设；如果状态包含足够历史信息（如 RNN 隐状态），POMDP 也可被 MDP 框架处理。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×13</span> <b>Q22</b> · Bellman 方程是什么？$V(s)$ 和 $Q(s,a)$ 的关系？</summary>

**答**：**Bellman 期望方程**（某策略 $\pi$ 下）：

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')]$$

**关系**：$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$（对动作的加权期望）；$Q^\pi(s,a) = R + \gamma \mathbb{E}_{s'}[V^\pi(s')]$（即时奖励 + 折扣后继状态值）。

**最优 Bellman**：$V^*(s) = \max_a Q^*(s,a)$；$Q^*(s,a) = R + \gamma \mathbb{E}_{s'}[\max_{a'} Q^*(s', a')]$。

**易错**：Bellman 方程是递推定义（自洽方程），不是显式解；Q-learning 正是通过 TD 更新迭代逼近 $Q^*$，无需知道 $P$。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×11</span> <b>Q23</b> · Q-learning 的更新公式是什么？DQN 相比 Q-learning 加了哪些改进？</summary>

**答**：**Q-learning 更新**（off-policy TD）：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \underbrace{\big[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\big]}_{\text{TD error}}$$

**DQN 三大改进**：
1. **Experience Replay**：将 $(s, a, r, s')$ 存入 replay buffer，随机采样打破样本相关性，提升数据效率
2. **Target Network**：用参数 $\theta^-$（定期硬拷贝或软更新）计算 $\max_{a'} Q(s', a'; \theta^-)$，稳定训练目标（否则目标也在变动 → 振荡）
3. **函数近似（神经网络）**：用 CNN 直接从像素输入端到端估计 $Q$ 值

**易错**：Q-learning 是 off-policy（目标用 max，不管行为策略）；SARSA 是 on-policy（用实际选择的 $a'$）；区别在 TD target 的动作来源。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q24</b> · Monte Carlo 和 TD 方法的区别？各自适合什么情况？</summary>

**答**：

| | Monte Carlo (MC) | TD (Temporal Difference) |
|---|---|---|
| 更新时机 | Episode 结束后（用完整回报 $G_t$） | 每步（用下一状态估计 bootstrap） |
| 方差 | 高（完整轨迹噪声叠加） | 低 |
| 偏差 | 无偏（真实样本 return） | 有偏（bootstrap 引入估计误差） |
| 适合场景 | 回合制、非 Markov 环境 | 持续交互、长 episode |

**联系**：TD($\lambda$) 通过 $\lambda$ 在 MC 和 TD 之间插值；GAE 是 TD($\lambda$) 在 advantage 估计中的应用（PPO 的核心组件）。

**易错**：MC 不需要 Markov 假设（用真实 return），但 variance 太高、数据利用率低；TD 需要 bootstrap（依赖 Markov 性），但在线学习效率高。机器人长任务（500步以上）用 TD 而非 MC。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q25</b> · 策略梯度（Policy Gradient）的核心思想是什么？REINFORCE 算法怎么工作？</summary>

**答**：**核心**：直接参数化策略 $\pi_\theta(a \mid s)$，对 $J(\theta) = \mathbb{E}_\tau[G_0]$ 做梯度上升。策略梯度定理：$\nabla J = \mathbb{E}[\sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t]$。

**REINFORCE**：① 采样完整轨迹；② 计算 $G_t = \sum_{k \geq t}\gamma^{k-t}r_k$；③ 用 $\nabla \log \pi_\theta \cdot G_t$ 更新参数（on-policy）。

**直觉**：$G_t$ 大 → 增大 $a_t$ 的概率；$G_t$ 小 → 减小概率。等同于"奖励加权的最大似然"。

**易错**：REINFORCE 方差极高（$G_t$ 是完整轨迹的累积奖励，噪声大）→ 引入 baseline $b(s_t) = V(s_t)$，用 $A_t = G_t - V(s_t)$ 替代 $G_t$。Baseline 不改变期望梯度（$\mathbb{E}[\nabla \log \pi \cdot b] = 0$），但大幅降方差。这就是 Actor-Critic 的起点。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×9</span> <b>Q26</b> · Actor-Critic 是什么？为什么它结合了 PG 和值函数估计？</summary>

**答**：**Actor-Critic** = Actor（策略网络 $\pi_\theta$）+ Critic（值函数估计器 $V_\phi$ 或 $Q_\phi$）。

**Actor**：用策略梯度更新，目标是选更好的动作；
**Critic**：用 TD 误差训练，目标是准确估计 $V(s)$ 或 $Q(s,a)$；
**Critic 作 baseline**：$A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$（TD advantage）直接作 Actor 的加权项，替代 REINFORCE 中的高方差 $G_t$。

**优势**：① 比纯 MC（REINFORCE）方差低（用 $A_t$）；② 比纯值方法（DQN）可以处理连续动作；③ 在线学习（不需要等 episode 结束）。

**延伸路径**：A2C（同步）→ A3C（异步）→ PPO（clip 截断比例）→ SAC（max-entropy AC）。

**易错**：Actor 和 Critic 可以共享主干（节省参数），也可以完全分开；机器人长程任务通常分开，防止两个目标梯度互相干扰。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q27</b> · On-policy 和 Off-policy 的区别？各自典型算法是什么？</summary>

**答**：

- **On-policy**：行为策略 = 目标策略（用来采样的策略就是正在优化的策略）。数据必须来自当前策略，旧数据无效。典型：**PPO, A3C, TRPO, SARSA**。
- **Off-policy**：行为策略 ≠ 目标策略（可以用其他策略（含过去旧策略）采样的数据来训练）。数据可复用（replay buffer）。典型：**Q-learning, DQN, SAC, TD3, DDPG**。

**权衡**：
- On-policy 样本效率低（用完即扔），但稳定、收敛保证好；
- Off-policy 样本效率高（replay buffer 重复用），但需处理重要性采样 / 分布偏移。

**机器人选择**：真机采样贵 → 首选 off-policy（SAC）；仿真大规模可重置 → on-policy（PPO）也可接受；RLHF 微调 LLM/VLA 语言部分常用 PPO；VLA 的动作策略本身更多用 BC/LoRA/flow-matching 等监督学习方式。

**易错**：PPO 看似可以用旧数据（clip 范围内），但理论上仍是 on-policy（clip 是近似约束，偏离太多就不行）；**PPO 不是 off-policy**。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q28</b> · 折扣因子 $\gamma$ 的作用是什么？设置过大/过小会怎样？</summary>

**答**：$\gamma \in [0,1)$ 对未来奖励的权重指数衰减：$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$。

**直觉**：① 数学上保证无限 horizon 的回报收敛（$\gamma < 1$）；② 语义上表示"对未来不确定性的折价"，近期奖励更可靠。

**$\gamma$ 太大（如 → 1）**：对遥远未来奖励权重高，bootstrap 误差远程传播，训练不稳定；长期规划强但 credit assignment 难（某奖励追溯很多步前的动作）。

**$\gamma$ 太小（如 → 0）**：只看即时奖励，变成贪心策略（myopic）；长程任务中错过关键延迟奖励。

**机器人取值**：通常 $\gamma \in [0.95, 0.99]$；稀疏奖励长程任务（200步以上）用 0.99；短程密集奖励（10步内）可用 0.95。

**易错**：$\gamma = 1$ 仅在有限 horizon + episode 一定结束的场景才可用；无限 horizon 时 $\gamma = 1$ 导致 return 不收敛。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q29</b> · 稀疏奖励（Sparse Reward）有什么问题？有哪些缓解方法？</summary>

**答**：**问题**：大多数时候奖励为 0，随机策略极难偶然达到目标 → 梯度几乎全是 0 → 训练无信号，无法学习。

**缓解方法**：
1. **Reward Shaping**：人工设计中间奖励（接近目标、减少距离等），但需领域知识且可能引入错误最优化
2. **HER（Hindsight Experience Replay）**：把失败轨迹的实际到达状态当作"新目标"，重新标注为成功，生成大量正样本 — **off-policy 机器人任务最常用**
3. **课程学习（Curriculum）**：从易到难逐步增加任务难度，确保早期能获得正奖励
4. **探索增强**：RND（随机网络蒸馏）/ Intrinsic Curiosity Module，用"新奇性"作内在奖励鼓励探索未访问状态
5. **模仿学习预热**：先用 BC（专家演示）学会基础行为再 RL 精调

**易错**：Reward Shaping 若与真实目标不对齐，会导致策略学会"刷虚假奖励"而非完成任务（reward hacking）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q30</b> · 什么是 Advantage 函数？它为什么比直接用 $Q(s,a)$ 更好？</summary>

**答**：**Advantage 函数**：$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$，表示在状态 $s$ 执行动作 $a$ 相对于"平均水平"的超额价值。

**为什么更好**：
- 纯用 $Q(s,a)$ 作 PG 的权重：量纲大（$Q$ 的绝对值可能很大），方差高；
- $A(s,a)$ 零中心化（某些动作正、某些负），梯度信号更清晰——好于平均的动作被强化，差于平均的被抑制；
- 方差显著降低（baseline = $V(s)$ 的效果）；
- PPO 的 clip 目标就是 $\hat{A}_t$，没有 advantage 就没有 PPO。

**GAE（广义优势估计）**：$\hat{A}_t^{\text{GAE}} = \sum_{l \geq 0} (\gamma\lambda)^l \delta_{t+l}$（TD 残差加权求和），$\lambda$ 在低方差 TD 和低偏差 MC 之间插值，实务 $\lambda \approx 0.95$。

**易错**：理论上 $\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0$（对任意状态 $s$ 成立），但实践中 Critic 的 $V$ 是估计值，批量计算的 $\hat{A}_t$ 不一定严格零均值；对 mini-batch 内 advantage 做归一化（减均值除标准差）是常见 trick，可稳定训练但不等同于真实 advantage 值。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q31</b> · PPO 的 Clip 机制是什么？为什么比 TRPO 更实用？</summary>

**答**：$L^{CLIP} = \mathbb{E}_t[\min(r_t \hat{A}_t,\ \text{clip}(r_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_t)]$，其中 $r_t = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$，$\epsilon \approx 0.1$-$0.2$。

**直觉**：$\hat{A}_t > 0$（好动作）时增大 $r_t$，但截到 $1+\epsilon$；$\hat{A}_t < 0$ 时减小 $r_t$，但截到 $1-\epsilon$。防止更新幅度过大，新旧策略保持接近。

**关键对比**：
- **TRPO**：KL 散度硬约束 + 共轭梯度二阶优化，理论严格但计算复杂；
- **PPO**：clip 是软约束，一阶梯度即可，实现简单，是 RLHF 偏好对齐的工程主流；VLA 动作策略本身更多用 BC/flow-matching 监督学习。

**易错**：PPO 不保证真正的 trust region（多次 mini-epoch 后 $r_t$ 可能超出 clip 范围）；clip 是启发式，不是严格约束。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q32</b> · SAC（Soft Actor-Critic）的最大熵框架是什么？为什么它在机器人真机训练中更受青睐？</summary>

**答**：**最大熵 RL**：目标从最大化期望 return 改为同时最大化期望 return 和策略熵：

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t)))\right]$$

$\alpha$ 为温度参数（自动调整），$\mathcal{H}$ 为策略熵。

**好处**：① **探索内化**：高熵策略自然探索多种动作，不易陷入局部最优；② **鲁棒性**：同时优化多种良策略，对环境扰动（真机噪声）更鲁棒；③ **off-policy + replay buffer**：样本效率高，真机数据贵时可多次复用。

**自动温度**：$\alpha$ 自动调节让熵约束在目标熵附近，无需手调——这是 SAC 工程上比 TD3 友好的关键。

**易错**：SAC 的"soft"是指软 Q-learning（熵正则化），不是"软"更新（虽然 SAC 也用软更新 target network）；两个概念不同。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×5</span> <b>Q33</b> · Offline RL 与 Online RL 的核心区别？为什么 Offline RL 有分布偏移问题？</summary>

**答**：

- **Online RL**：边训练边与环境交互，数据来自当前策略；可以探索、纠错。
- **Offline RL**：完全从固定数据集（历史演示/日志）中学习，**不与环境交互**；数据集可能来自多种行为策略。

**分布偏移（Distribution Shift）**：学到的策略可能访问数据集中从未出现的 $(s,a)$ 对 → Q 函数对这些 OOD（out-of-distribution）动作的估计极不准确（往往过高，因为神经网络外推不可靠）→ 策略被错误高估的动作引导走错。

**对策**：
- **CQL（Conservative Q-Learning）**：在 Q 值优化中额外惩罚 OOD 动作的 Q 值，强制保守估计；
- **IQL（Implicit Q-Learning）**：避免直接对 OOD 动作查 Q，只在数据集内做 expectile 回归；
- **BC 约束**：限制策略不偏离行为策略太远（与 KL 正则类似）。

**机器人应用**：真机无法探索时（危险/昂贵）必须 offline RL；离线预训练 + 少量在线 fine-tune 是常见组合。

**易错**：Offline RL 不是"有数据就行"——数据覆盖度和质量极关键；覆盖差+高 OOD 比率时 CQL 等方法也会失效。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×4</span> <b>Q34</b> · 什么是 Reward Hacking？举一个机器人场景的例子。</summary>

**答**：**Reward Hacking**：策略找到一种刷奖励但不完成真实任务目标的方式——最大化了代理奖励（proxy reward）但不是真实意图。

**机器人例子**：
- 任务：把物体放到目标位置；奖励函数：$r = -\|p_{ee} - p_{goal}\|$（末端接近目标）→ 策略学会用末端**压住**目标物体（距离为 0，奖励最大），而非"把物体放到目标位置"。
- 更极端：擦桌子任务，奖励用"摄像头亮度变化"衡量清洁程度 → 策略学会遮挡摄像头（最高变化）。

**根本原因**：奖励函数是真实目标的不完美代理；优化器（RL agent）总能找到"钻空子"路径。

**对策**：① Reward 设计时多场景验证（用 test 场景检查有无异常行为）；② RLHF（人类反馈）反复审查；③ Constrained RL（加硬约束）；④ 多奖励信号互相制衡。

**易错**：Reward hacking 不是"bug"，是优化成功——只是目标函数写错了。调试时关键要看策略实际行为，而不只看奖励曲线。

</details>

---

## §3 机器人学基础（10 题）

> 具身岗专项，考察坐标系、运动学、控制基础；工程岗必掌握，研究岗了解即可。

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q35</b> · 旋转矩阵、欧拉角、四元数三种姿态表示的区别和联系？</summary>

**答**：

| 表示 | 参数数 | 优点 | 缺点 |
|---|---|---|---|
| 旋转矩阵 $R \in SO(3)$ | 9（实际 3 自由度） | 无奇异、组合简单（矩阵乘法） | 冗余参数，存储/传输大 |
| 欧拉角（roll/pitch/yaw） | 3 | 直观（人类可理解） | **万向节死锁**（Gimbal Lock），依赖轴顺序 |
| 四元数 $q = [w,x,y,z]$，$\|q\|=1$ | 4（实际 3 自由度） | 无奇异、插值平滑（SLERP）、计算高效 | 不直观，有双倍映射（$q$ 和 $-q$ 表示同一转） |

**联系**：三者等价，可互相转换（旋转矩阵 ↔ 四元数 ↔ 轴角 ↔ 欧拉角）。

**机器人实践**：运动规划/插值用四元数；人机界面 / 调试显示用欧拉角；姿态矩阵运算用旋转矩阵；神经网络输出姿态通常用 6D 旋转表示（$R$ 的前两列）或四元数。

**易错**：欧拉角的旋转顺序（ZYX vs XYZ vs ZYZ）不同结果完全不同——拿到欧拉角必须确认约定；Gimbal Lock 发生在 pitch = ±90° 时 roll 和 yaw 轴重合。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q36</b> · 齐次变换矩阵（Homogeneous Transformation Matrix）是什么？如何用它表示坐标系变换？</summary>

**答**：**齐次变换矩阵** $T \in SE(3)$，将旋转 $R$ 和平移 $p$ 合并为 4×4 矩阵：

$$T = \begin{bmatrix} R_{3\times3} & p_{3\times1} \\ 0_{1\times3} & 1 \end{bmatrix}$$

**坐标变换**：若点在 frame B 中坐标为 $\tilde{q}_B = [x, y, z, 1]^\top$，在 frame A 中：$\tilde{q}_A = {}^A T_B \cdot \tilde{q}_B$。

**链式组合**：多个变换可直接矩阵连乘：${}^0 T_n = {}^0 T_1 \cdot {}^1 T_2 \cdots {}^{n-1} T_n$（正向运动学本质）。

**逆变换**：$T^{-1} = \begin{bmatrix} R^\top & -R^\top p \\ 0 & 1 \end{bmatrix}$，旋转部分取转置（正交矩阵性质）。

**易错**：旋转矩阵的逆是其转置（$R^{-1} = R^\top$），但整个齐次矩阵的逆**不是**直接转置 $T^\top$（平移部分要另外处理）。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×9</span> <b>Q37</b> · D-H 参数法（Denavit-Hartenberg）是什么？四个参数各代表什么？</summary>

**答**：D-H 参数法是用**最少 4 个参数**描述相邻关节坐标系变换的标准化方法：

| 参数 | 含义 | 变化/固定 |
|---|---|---|
| $\theta_i$（关节角） | 绕 $Z_{i-1}$ 轴的旋转角 | **旋转关节变量**（固定关节为常数）|
| $d_i$（连杆偏距） | 沿 $Z_{i-1}$ 轴的平移距离 | **移动关节变量**（旋转关节为常数）|
| $a_i$（连杆长度） | 沿 $X_i$ 轴的距离（$Z_{i-1}$ 到 $Z_i$ 的公垂线长） | 固定（几何参数）|
| $\alpha_i$（连杆扭转角） | 绕 $X_i$ 轴从 $Z_{i-1}$ 转到 $Z_i$ 的角度 | 固定（几何参数）|

**正运动学**：给定所有关节变量 $\theta_i$（或 $d_i$），连乘 $T_i$ 得末端位姿：${}^0 T_n = \prod_{i=1}^n {}^{i-1}T_i$。

**两种约定**：标准 D-H（Craig） vs 改进 D-H（MDH / Spong）参数定义顺序略有不同，拿到参数表必须确认约定。

**易错**：标准 D-H 每帧 $X_i$ 沿公垂线方向，并不是"随便放"；如果机器人有平行或相交关节轴，D-H 坐标系有歧义，需额外约定。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×9</span> <b>Q38</b> · 逆运动学（IK）的解析解和数值解各有什么优劣？哪种更常用于 VLA 策略执行？</summary>

**答**：

**解析解（Analytical IK）**：
- 对特殊构型（如满足 Pieper 准则：3 轴交于一点的 6-DOF 机器人）可求闭式解；
- 优点：速度极快（毫秒级）、确定性强；
- 缺点：仅适用于特定构型，通用性差，解可能有多解需选优。

**数值解（Numerical IK）**：
- 基于雅可比矩阵的迭代法（梯度下降 / Newton-Raphson / Damped Least Squares）；
- 优点：通用（任意构型）、可加约束（关节限位、奇异回避）；
- 缺点：迭代耗时（ms 到 tens of ms），可能陷入局部最优，初值敏感。

**VLA 策略**：不同 VLA 的动作空间不同——OpenVLA 官方动作空间是**末端执行器 7 维增量**（$\Delta x, \Delta y, \Delta z, \Delta$roll, $\Delta$pitch, $\Delta$yaw + gripper），底层执行层需要 IK/OSC（操作空间控制）将其转换为关节指令。部分 VLA 直接输出关节角度，不需要 IK。**结论**：VLA 动作空间因模型而异，不能一概而论；但相比传统离线规划，VLA 的 IK 调用是实时嵌在控制循环中的。

**易错**：7-DOF 冗余机械臂有无穷多 IK 解（冗余自由度），需加零空间运动（null space motion）约束来选择"最优"解（如避开奇异形位）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×11</span> <b>Q39</b> · 雅可比矩阵（Jacobian）是什么？奇异性是怎么产生的？</summary>

**答**：$J(\theta) \in \mathbb{R}^{6 \times n}$ 把关节速度映射到末端速度：$\dot{x} = J\dot{\theta}$。上 3 行线速度 $J_v$，下 3 行角速度 $J_\omega$。旋转关节 $i$：$J_v^{(i)} = z_{i-1} \times (p_n - p_{i-1})$，$J_\omega^{(i)} = z_{i-1}$。

**奇异性**：$J$ 行秩降低（不满秩）时某些末端运动方向不可达（或需无穷大关节速度）。典型：手腕三轴共线、连杆完全伸直。此时 Moore-Penrose 伪逆病态（满行秩情况下 $J^\dagger = J^\top(JJ^\top)^{-1}$）→ 关节速度爆炸。

**缓解**：① 阻尼最小二乘（DLS）：$J^\dagger_{dls} = J^\top(JJ^\top + \lambda I)^{-1}$，$\lambda$ 引入阻尼，用精度换稳定；② 路径规划提前绕开奇异形位；③ 控制层限速。

**易错**：6-DOF 方阵 $J$ 直接取逆仅在满秩时有效；7-DOF 冗余臂 $J \in \mathbb{R}^{6 \times 7}$ 必须用伪逆，存在无穷多解，需加零空间约束来选优。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q40</b> · PID 控制的三个参数各起什么作用？机器人关节 PID 调参时的典型问题？</summary>

**答**：**PID**：$u(t) = K_P e(t) + K_I \int e \,dt + K_D \dot{e}(t)$，$e = q_{desired} - q_{actual}$（关节角误差）。

- **$K_P$（比例）**：误差大则控制量大；太大 → 震荡；太小 → 稳态误差大。
- **$K_I$（积分）**：消除稳态误差（gravity compensation 不完美时的偏置）；太大 → 积分饱和（windup），超调 + 慢恢复。
- **$K_D$（微分）**：阻尼，预测趋势提前减速；太大 → 对噪声极敏感（放大高频噪声）；关节位置传感器有噪声时慎用大 $K_D$。

**机器人典型问题**：
- 负载变化（搬不同重物）→ 有效惯量变化 → 固定 PID 参数可能失稳，需自适应控制或惯量辨识。
- 重力补偿不完整 → 靠 $K_I$ 修正，易积分饱和。
- 柔性关节（弹性）→ 纯 PID 可能激发弹性振动，需加低通滤波或更高级控制（如弹性关节模型控制）。

**易错**：PID 是线性控制器，机器人动力学非线性（惯量、离心力、科里奥利力、重力随形位变化）；PID 在工作点附近可用，大范围运动需非线性补偿或现代控制。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q41</b> · 关节空间控制和任务空间（笛卡尔）控制的区别？各适合什么场景？</summary>

**答**：

- **关节空间控制**：控制目标是关节角度/速度 $\theta$，PD 控制在关节空间运行，无需 IK。**优点**：简单稳定、无奇异问题；**缺点**：末端路径不直（关节线性插值 ≠ 笛卡尔直线）。
- **任务空间（Cartesian）控制**：控制目标是末端位姿 $x$，需通过 Jacobian 或 IK 转换到关节空间。**优点**：末端路径可控（直线/圆弧）、易与视觉感知对接；**缺点**：奇异点附近不稳定，需 IK 计算开销。

**适合场景**：
- Pick-and-place、点到点运动 → 关节空间够用
- 精密装配（要求末端走直线）、力控 → 任务空间
- VLA 策略输出 → 因模型而异（OpenVLA 为末端增量、需底层 IK；部分模型直接输出关节角）

**易错**："笛卡尔控制"不等于"不需要 IK"——只是 IK 在每个控制循环内完成（数值迭代），且控制频率要够高（>100 Hz）才能近实时。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q42</b> · 阻抗控制（Impedance Control）和力控制的区别？具身机器人什么时候用力控？</summary>

**答**：

- **纯位置控制**：仅跟踪位置目标，接触时力无法限制（硬接触易损伤）。
- **力控制（Force Control）**：直接控制接触力，需要力/力矩传感器或力矩估计；精度高但刚性差（力变化时位置漂移）。
- **阻抗控制**：在期望位置轨迹上叠加弹簧-阻尼模型 $F = K(x_d - x) + B(\dot{x}_d - \dot{x})$，**将力偏差转化为位置偏差允许**；不需要显式力控制器，只要能控制关节力矩或末端位置。

**具身机器人使用场景**：
- 装配（轴孔配合/插拔连接器）：位置不确定时力控防止过载 → **阻抗/力控**
- 拧螺丝/研磨打磨：需要恒定接触力 → **力控**
- 普通 pick-and-place（抓轻物）：位置控制即可
- 人机协作（安全要求）：阻抗控制让机器人"柔软"，碰到人自动退让

**易错**：阻抗控制中 $K$（刚度）高 ≈ 位置控制；$K$ 低 ≈ 力控；调 $K/B$ 是在"硬度"和"力控精度"间权衡，不是随意设。

</details>

<details class="qa">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×6</span> <b>Q43</b> · 什么是工作空间（Workspace）？奇异形位和工作空间边界有什么关系？</summary>

**答**：**工作空间**：机器人末端执行器能到达的所有**位置**集合（注：完整意义的可达性还需考虑姿态约束）。分：
- **可达工作空间**（Reachable）：至少以某一姿态能到达的位置集合；
- **灵巧工作空间**（Dexterous）：以任意姿态都能到达的位置集合（往往小得多）。

**与奇异形位的关系**：位置工作空间**边界**通常恰好是奇异形位（完全伸展或极度折叠）——此时 Jacobian 降秩，末端某方向运动需无穷大关节速度。内奇异点（如腕部奇异）在工作空间内部也可能存在。

**机器人设计取舍**：7-DOF 冗余臂扩大灵巧工作空间 + 提供零空间运动绕奇异，但增加控制复杂度。

**易错**：灵巧工作空间既受位置限制又受姿态约束，两者分开分析；可达工作空间描述的是位置可达性，不自动保证任意姿态下的可达性。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q44</b> · 机器人控制频率（Control Frequency）的要求是什么？VLA 策略频率低有什么影响？</summary>

**答**：**基本要求**：控制频率需远高于系统最高频动态（奈奎斯特采样定理），一般：
- 关节 PD 控制：**1 kHz**（高刚度电机）或 **500 Hz**（标准）
- 力控 / 阻抗控制：**100-500 Hz**（力传感器带宽）
- 视觉反馈（闭环）：**30-100 Hz**（摄像头帧率限制）

**VLA 策略频率低的影响**：OpenVLA 在 Franka-Tabletop 约 **5 Hz**（串行 AR 解码 7 token），部分部署平台可达 15 Hz，大部分 VLA 在 5-25 Hz。
- **安全风险**：高动态任务（抓运动物体、避碰）控制环太慢，实际执行与规划脱节；
- **Action Chunking**（关键解法）：一次预测 K 步动作（如 K=10/20/50）并缓存，底层控制器以高频插值执行，策略只需低频推理；
- **π0**：用独立 Action Expert（Flow Matching，H=50 chunk）以约 50 Hz 频率输出动作，VLM 低频提供上下文理解；**GR00T-N1** 类似思路（Flow Matching action chunk），具体控制频率因机器人平台而异。

**易错**：控制频率和推理频率是两层：VLA 推理 10 Hz，但底层 PD 控制仍跑 1 kHz；两层解耦是工程关键。

</details>

---

## 低频备选（未入主表，频次 1-2）

> 以下题目在面经中出现频次低（1-2 次），或来源单一。用户可根据目标岗位选入。

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N1</b> · 什么是 Group Normalization（GN）？它和 BN/LN 有什么区别？【低频备选未入主表】</summary>

**答**：**GN** 在同一样本的特征通道上分组归一化（同一样本内每 $G$ 个通道一组），不依赖 batch size（与 LN 类似），但归一化粒度比 LN 细（LN 是全部通道一起归一化）。

**对比**：BN 依赖 batch 维度（batch 小时不稳）；LN 在整个特征维度（适合 NLP/Transformer）；GN 在图像特征通道分组（适合 CV 小 batch，如目标检测/视频分析）；Instance Normalization 对每个样本每个通道独立（适合风格迁移）。

**机器人应用**：VLA 的 ViT backbone（BN/LN 都有用）；视频/点云处理小 batch 场景 GN 较好。

**易错**：GN 与 batch size 无关，batch=1 完全可用；这是它在小 batch 目标检测（Mask RCNN 等）取代 BN 的原因。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×2</span> <b>N2</b> · 什么是 Curriculum Learning（课程学习）？在机器人任务训练中怎么应用？【低频备选未入主表】</summary>

**答**：**课程学习**：模仿人类教育的"先易后难"——让模型从简单样本/任务开始训练，逐步增加难度，最终收敛到完整任务分布。可以手动设计难度梯度，也可自动（Auto Curriculum，用成功率或 episodic return 驱动）。

**机器人应用**：① 从短任务（2 步抓取）到长任务（10 步组装）；② 从目标在抓手附近到随机位置；③ 从干净背景到复杂遮挡；④ 技能链（先学各子任务再组合）。

**效果**：显著加速稀疏奖励任务的收敛（对比纯随机初始化），减少随机探索代价。

**易错**：课程难度增长需要平衡——增长太慢浪费训练时间；增长太快相当于没有课程（跳回难题，刚学会的容易遗忘）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×1</span> <b>N3</b> · 什么是 Spectral Normalization（谱归一化）？在机器人策略训练中有什么用？【低频备选未入主表】</summary>

**答**：**谱归一化**：将每个权重矩阵 $W$ 除以其最大奇异值（谱范数）$\sigma_{max}(W)$，使 Lipschitz 常数 $\leq 1$：$\hat{W} = W / \sigma_{max}(W)$。

**作用**：① 训练稳定（限制梯度爆炸）；② GAN 判别器使用谱归一化防止模式崩塌（SNGAN、SAGAN、BigGAN 等广泛采用）；③ Q 函数在 RL 中用谱归一化可限制 Q 值过估的幅度（EDAC 等方法）。

**机器人**：Offline RL 中 CQL / SAC 的 Q 网络加谱归一化可提升训练稳定性；策略网络本身较少使用。

**易错**：谱归一化≠权重归一化（WeightNorm）——后者只归一化模长，不控制奇异值分布；两者作用不同。

</details>
