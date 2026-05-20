# CLAUDE.md — Embodied-AI-in-Offer 项目工作指令

> 当 Claude Code 在 `/home/scut/embodied_offer` 目录启动时自动加载此文件。
> 用户输入"做 XXX 系列"或"帮我做 XXX 速查表"时，按本文件的工作流执行，**无需重新调研构建思路**。

## 0. 项目定位

中文具身智能秋招速查手册集合，仿照 [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer) 范式：

- 每篇是一份 600–1000 行的长文 cheat sheet（公式 + from-scratch 代码 + 25 道 L1/L2/L3 面试题）
- 输出双产物：Markdown 源 + 单文件 HTML（响应式，手机/平板/PC 通用）
- 部署：**GitHub Pages 纯静态托管**，无后端
- 所有产物必须经过跨模型审查（Claude 写 / GPT-5.5 xhigh 审查）才能发布

## 1. 用户偏好（HARD RULES — 不可违反）

| 规则 | 来源 | 触发场景 |
|---|---|---|
| **写之前先列大纲让用户审过** | 2026-05-20 反馈 "Hi Robot 不是我想要的内容你也写进去了" | 任何新主题开工前，先列覆盖的子主题清单 + 各自的核心点，明确询问"哪些要 / 哪些不要 / 哪些要补"，等用户回复再开写 |
| **完成后必须 push 到 GitHub** | 2026-05-20 反馈 "我不是为了在本地阅读的" | 渲染 + 审计完成后，主动 commit + push；如果还没远程仓库，要询问仓库名+可见性 |
| **手机端必须可用** | 2026-05-20 反馈 "手机点击也能看，排版不会乱" | 用 `academic` 模板（已自带 responsive CSS）；不要用 `dashboard` 模板 |
| **首选主题准确性优先于覆盖广度** | 同上 | 宁可少写一个版本/方法，也不要塞用户没要的内容 |

## 2. 标准工作流（用户说"做 XXX"时的固定动作序列）

```
Phase 0 — 解析与对齐
  ① 解析 "XXX" 是哪个主题（RL / VLA / 世界模型 / 仿真等）
  ② 在 docs/tutorials/README.md 路线图里找对应位置
  ③ 用 Agent 子代理快速调研该主题的最新进展（用 WebFetch + WebSearch）
  ④ ★ 列大纲：12-14 节标题 + 每节核心 1-2 句 → 提交用户审过
  ⑤ ★ 等用户确认增删后再进 Phase 1

Phase 1 — 深度调研
  调 general-purpose Agent，写一份结构化报告：
    - 版本/方法时间线（一张表）
    - 每个版本/方法的资料卡（含 arXiv ID, 作者, 关键数字）
    - 数学公式 + 横向对比
    - 25 道面试题草稿（L1×10 + L2×10 + L3×5）

Phase 2 — 起草 Markdown
  写到 docs/tutorials/<slug>_tutorial.md
  风格严格参照 /home/scut/aris_repo/docs/tutorials/attention_tutorial.md：
    - ## §N Title（§N 后有空格，不可粘连）
    - 表格内 math 用 \lvert ... \rvert 不用 |...|
    - callout intro line 与列表分两行（不要 "> 💡 **xxx** — 1. ... 2. ..."）
    - callout 前缀仅 💡 ⚠️ ✅ ❌（其它没 class）
    - $...$ 行内 / $$...$$ 行间 / $$\boxed{...}$$ 关键框
    - 中文为主，英文术语保留原文
    - 代码块用 ```python，且要 statically correct
  长度目标 1000 行（max effort）/ 600 行（balanced）

Phase 3 — 跨模型审查（必跑，不可跳）
  调用 mcp__codex__codex：
    model: "gpt-5.5"
    config: {model_reasoning_effort: "xhigh"}
    sandbox: "read-only"
    fresh thread 每轮（绝不复用 threadId）
  审查 10 项：
    1. formula_correctness（独立重推每个 $$ 公式）
    2. code_correctness（每个 python 块能否跑、shape/device 一致）
    3. interview_answer_correctness（25 题答案核对）
    4. historical_citations（arXiv ID / 年份 / 作者）
    5. table_pipe_escape（表格内是否有 |x| 未转义）
    6. callout_list_collision（callout + list 同行碰撞）
    7. heading_consistency（§N 空格）
    8. section_completeness（§0..§N + §A 齐全）
    9. length_target（±20% 内）
    10. personal_info_leak（无 SJTU / JHC / Server5 / /Users/ 等）
  循环修 FAIL → fresh thread 再审，直到 PASS。
  典型收敛 3-5 轮；超过 6 轮无收敛 → 停下来报用户。
  审计日志写到 docs/tutorials/<slug>_tutorial.review.json

Phase 4 — 渲染 HTML（必用 academic 模板）
  cd /home/scut/embodied_offer
  python3 tools/render_html.py docs/tutorials/<slug>_tutorial.md \
    --template academic \
    --out docs/tutorials/<slug>_tutorial.html \
    --title "<Topic> 面试 Cheat Sheet" \
    --subtitle "<scope summary>" \
    --eyebrow "Embodied AI Interview Prep · <Topic>" \
    --author "WinstonJQ" \
    --lang zh-CN

Phase 5 — 更新索引
  ① docs/index.html 把新主题从 TODO 换成 DONE 卡片
  ② README.md 路线图勾选

Phase 6 — Push 到 GitHub
  git add docs/tutorials/<slug>_tutorial.{md,html,review.json} \
          docs/index.html README.md
  git commit -m "docs: add <Topic> interview cheat sheet"
  git push origin master
  （远程仓库已存在；首次 push 见 §5）
  push 完后给用户最终报告：发布 URL + 审查轮数 + fix 数

Phase 7 — 报告
  ✅ <Topic> cheat sheet 完成
    HTML:   https://winstonjq.github.io/embodied_offer/tutorials/<slug>_tutorial.html
    Markdown: docs/tutorials/<slug>_tutorial.md (<lines> 行)
    审查:   PASS after N rounds, M fixes
    Commit: <commit hash>
```

## 3. 主题清单与 slug 命名

slug = topic 的 kebab/snake-case，**不含日期/版本号噪声**。

| 主题（用户说法） | slug | 状态 |
|---|---|---|
| π 系列 / pi 系列 | `pi_series` | ✅ done |
| VLA 综述 / RT 系列 / OpenVLA | `vla_survey` | TODO |
| Diffusion Policy / 动作扩散 | `diffusion_policy` | TODO |
| RL 基础 / 强化学习基础 | `rl_foundations` | TODO |
| PPO / TRPO / GAE | `ppo_trpo` | TODO |
| SAC / TD3 / DDPG | `sac_td3` | TODO |
| Offline RL / 离线 RL | `offline_rl` | TODO |
| 模仿学习 / BC / GAIL | `imitation_learning` | TODO |
| 机器人学速查 | `robot_kinematics` | TODO |
| MPC / 轨迹优化 | `mpc_trajopt` | TODO |
| Sim-to-Real | `sim2real` | TODO |
| 世界模型基础 / Dreamer | `world_models` | TODO |
| 大世界模型 / Genie / GAIA / V-JEPA | `large_world_models` | TODO |
| 表征学习 / R3M / VC-1 / Voltron | `robot_representation` | TODO |
| 触觉 / 多模态感知 | `tactile_multimodal` | TODO |
| MBRL / Planning | `mbrl_planning` | TODO |
| Manipulation 基准 | `manipulation_benchmarks` | TODO |
| 导航 / VLN | `vln_navigation` | TODO |

如果用户说的主题不在这张表里，先列大纲 + 提议新 slug 让用户确认，再补充进表。

## 4. 关键文件路径

```
/home/scut/embodied_offer/                  ← 本仓库 (cwd)
├── CLAUDE.md                                ← 本文件
├── README.md                                ← 仓库主页
├── docs/
│   ├── index.html                           ← GitHub Pages 入口
│   └── tutorials/
│       ├── <slug>_tutorial.md               ← Markdown 源
│       ├── <slug>_tutorial.html             ← 渲染 HTML
│       └── <slug>_tutorial.review.json      ← 审计日志
└── tools/
    ├── render_html.py                       ← 渲染脚本（零依赖）
    └── templates/
        ├── academic.html                    ← ★ 必用模板（responsive）
        └── dashboard.html                   ← 不用

/home/scut/aris_repo/docs/tutorials/         ← 风格参照（attention_tutorial.md）
/home/scut/.claude/skills/                   ← 备用 skill 副本
```

## 5. GitHub 远程仓库

- GitHub 用户：`WinstonJQ`
- 仓库名：`embodied_offer`（与本地目录同名）
- 远程 URL：`git@github.com:WinstonJQ/embodied_offer.git` 或 `https://github.com/WinstonJQ/embodied_offer.git`
- 默认分支：`master`
- 可见性：**public**（GitHub Pages 免费版要求 public）
- GitHub Pages 配置：Source = `master / docs`
- 发布 URL 模板：`https://winstonjq.github.io/embodied_offer/tutorials/<slug>_tutorial.html`

**首次 push 流程**（仓库已在 GitHub 网页端创建空仓库后）：

```bash
git remote add origin https://github.com/WinstonJQ/embodied_offer.git
git push -u origin master
```

**后续 push**：

```bash
git add docs/tutorials/<slug>_tutorial.{md,html,review.json} \
        docs/index.html README.md
git commit -m "docs: add <Topic> interview cheat sheet"
git push origin master
```

## 6. 跨模型审查 — codex MCP 调用模板

```python
mcp__codex__codex(
    model="gpt-5.5",
    config={"model_reasoning_effort": "xhigh"},
    sandbox="read-only",
    cwd="/home/scut/embodied_offer",
    prompt="""FRESH THREAD — review of <FILE>.

# Files (READ-ONLY)
- Draft MD: /home/scut/embodied_offer/docs/tutorials/<slug>_tutorial.md
- Style ref: /home/scut/aris_repo/docs/tutorials/attention_tutorial.md (style only)

# 10 checks (see CLAUDE.md §2 Phase 3)

Return JSON:
{
  "verdict": "PASS | WARN | FAIL",
  "checks": {check_name: status + note + file:line},
  "blocking_issues": [],
  "warnings": [],
  "summary": "..."
}
"""
)
```

**每轮必开 fresh thread** — 绝不调用 `mcp__codex__codex-reply`。

## 7. 已知陷阱（lessons from 过往修正）

| 陷阱 | 触发条件 | 避免方法 |
|---|---|---|
| Flow matching path 方差 | 描述 $A^\tau = \tau A + (1-\tau)\varepsilon$ 时 | 必须用 $(1-\tau)^2 I$ 不是 $(1-\tau) I$ |
| Advantage 公式过度简化 | RECAP / DT 章节 | 用 n-step 形式 $A_t = \mathbb{E}[\sum r] + V(s_{t+N}) - V(s_t)$ |
| 控制频率单位 | "50 Hz 要求 X ms" | 是 **20 ms / control step**，不是 20 ms / chunk |
| Stage 数据组成对不上 | 多阶段训练描述 | 列每个 stage 的具体数据集集合（不要写 "MM/ME/CE/HL/WD/VI 一锅炖"，要分阶段） |
| arXiv license 默认 | "论文 license CC-BY-4.0" | arXiv 默认是 "non-exclusive distribution"，不是 CC-BY |
| 复杂度表混搭 | flow inference 行 | π0/π0.5 = 10 步；π0.6/π0.7 = 5 步（KI recipe）；π0-FAST 是 AR 不在 flow 行 |
| Code 预期输出 | range(N) 循环的 print | range(2000) 不打印 step 2000，最后是 1500 |

## 8. 不要做的事

- ❌ 不要塞用户没明确同意的子主题（例如这次的 Hi Robot）
- ❌ 不要用 dashboard 模板（用 academic）
- ❌ 不要跳过跨模型审查直接渲染
- ❌ 不要在 codex 审查时复用 threadId
- ❌ 不要忘记 push GitHub 就报告完成
- ❌ 不要在 Markdown 里写 SJTU / JHC / Server5 / /Users/ 等个人信息
- ❌ 不要 force push 到 master（先 review 再 push）
