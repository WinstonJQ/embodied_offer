# CLAUDE.md — 具身智能高频面试题库 项目工作指令

> 当 Claude Code 在本仓库（`embodied-interview-qa`）目录启动时自动加载此文件。
> 用户输入"做卷 X / 做 XXX 卷 / 继续下一卷"时，按本文件的工作流执行。

## 0. 项目定位

**这是一个纯面试题库项目**，不是学习笔记 / cheat sheet。

- 内容：题目 + 200 字内精简答案 + 频次/难度标签
- 不含：公式推导、代码实现、知识讲解（如需深入学习另找资源）
- 形式：Markdown 源 + 单文件折叠 HTML（默认收起，点开看答案）
- 部署：GitHub Pages 静态托管，无后端
- 跨模型审查（Claude 写 / GPT-5.5 xhigh 审查）后发布

旧的 cheat sheet 模式（pi_series / rl_foundations / ppo_sac）已 **彻底废弃**，相关文件已 git rm。不要互链、不要复用其内容。

## 1. 用户偏好（HARD RULES — 不可违反）

| 规则 | 触发场景 |
|---|---|
| **写之前先列题目清单让用户审过** | 每卷开工前提交频次+难度+题目清单，等用户审过删/加/换 |
| **完成后必须 push 到 GitHub** | 渲染审计完成后，主动 commit + push |
| **手机端必须可用** | 用 academic 模板（已自带响应式 CSS） |
| **不按公司分类** | 题目不打公司标签；公司只出现在 Phase 1 调研报告的来源备注里 |
| **题量由频次决定** | 不强求 50 题/卷；某些卷可能 30 题或 80 题 |
| **同义题合并，频次 ≥3 才进卷** | 防止单条噪声面经污染 |
| **不写公式推导/代码块/知识讲解** | 答案是"标准回答模板"风格，不是"教程" |
| **答案精简（≤200 字）便于记忆** | 超过的拆"答 + 易错 + 延伸"3 段 |

## 2. 标准工作流

每一卷独立跑一遍：

```
Phase 1  → 调研：多关键词跨平台爬取 + 同义题合并 + 频次统计
            产出：/tmp/<slug>_questions_research.md（题目清单 + 频次 + 难度）
Phase 0  → 提交题目清单给用户审过删/加/换（必停下等用户回复）
Phase 2  → 起草 docs/interviews/XX_<slug>.md（<details> 折叠块格式）
Phase 3  → codex gpt-5.5 xhigh 跨模型审查（10 项检查）
            循环修 FAIL → fresh thread 再审，典型收敛 5-6 轮（卷三实测，见 §11.2）
            审计日志：docs/interviews/XX_<slug>.review.json
Phase 4  → 渲染 academic 模板 HTML
Phase 5  → 更新 docs/index.html（主册入口卡片状态 + Top N）
Phase 6  → git add + commit + push origin master
Phase 7  → 报告发布 URL + 题数 + 审查轮数
```

## 3. 分册

| 卷 | slug | 主题 | 状态 |
|---|---|---|---|
| 一 | basics | 通识基础（DL / RL 基础 / 机器人学） | TODO |
| 二 | rl_algo | RL 算法（PPO / SAC / Offline RL） | TODO |
| 三 | vla_il | VLA / 模仿学习（OpenVLA / π / Diffusion Policy / BC） | **进行中** |
| 四 | world_sim | 世界模型 / Sim2Real | TODO |
| 五 | engineering | 工程落地 / 系统设计 / 开放题 | TODO |

题量灵活：调研出多少进卷由频次决定，不强求 50。

## 4. 题目格式

每题用 HTML5 `<details>` 标签实现折叠：

```markdown
<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×12</span> <b>Q01</b> · OpenVLA 和 RT-2 在架构上的主要区别？</summary>

**答**：200 字内精简回答（含必要对比/数字，不写公式推导）。

**易错**：一句话点关键陷阱。

</details>
```

渲染后：
- 默认折叠，只看到一行（难度色标 + 频次徽章 + 题号 + 题目）
- 点击展开看答案
- 手机端原生支持（HTML5 `<details>`，零 JS）

**字段约定**：
- `<span class="lv lv-l1">L1</span>` 绿色 / `lv-l2` 黄色 / `lv-l3` 红色
- `<span class="freq">🔥×N</span>` N 为同义题合并后的出现次数
- 题号 Q01...QN：卷内顺序，按频次或难度组织（频次高优先）
- 答案目标：≤200 字；超过的拆 3 段（答 / 易错 / 延伸）

## 5. 题源调研规则

**关键词**（每个都必须用 WebSearch 跑过）：

中文：
- 「VLA 实习 面经」「具身实习 面经」「具身智能 算法岗 面经」「具身 offer」
- 「OpenVLA 面试」「π0 面试」「diffusion policy 面试」「模仿学习 面试题」
- 「机器人 算法岗 面经」「robot policy 面试」
- 公司名作搜索关键词（仅搜索用，不打分类标签）：字节 Seed Robotics / 智元 / 银河通用 / 星海图 / π / Figure / 1X / 宇树 / 蔚来 / 小米 CyberOne / 优必选

英文（少量）：「embodied AI internship interview」「VLA interview questions」

**调研平台**：
- 牛客网 / 知乎 / 小红书 / 一亩三分地
- GitHub awesome-* 仓库
- 公众号"具身智能之心""自动驾驶之心"网页版归档
- Reddit r/robotics, r/MachineLearning（少量补充）

## 6. 频次合并规则

- 相似度 ≥ 70% 视为同一题 → 合并计数
- 同一帖子里重复出现的同义题只算一次该帖
- 不同帖子各算一次
- 合并到最规范表述（简洁、含问号的中文版本）
- 频次 ≥3 进主表
- 频次 1-2 放"低频备选"档（用户可选入卷，要标注）
- 2 年以上的过时题（如只考 RT-1）剔除

## 7. 跨模型审查（必跑，10 项）

```python
mcp__codex__codex(
    model="gpt-5.5",
    config={"model_reasoning_effort": "xhigh"},
    sandbox="read-only",
    cwd="<repo-root>",  # 用当前 cwd，不要写绝对路径
    prompt="""FRESH THREAD — review of docs/interviews/<XX_slug>.md (面试题库).

# 10 checks
1. answer_correctness：每题答案技术正确，公式/概念/数字无误
2. frequency_validity：🔥×N 与 Phase 1 调研报告频次一致
3. difficulty_appropriate：L1/L2/L3 标定合理（L1 不是太难，L3 不是太浅）
4. timeliness：题目仍是当前高频（>2 年过时题剔除）
5. answer_conciseness：单题答案体 **≤350 字 PASS / 350-500 WARN / >500 FAIL**（见 §11.1）
6. details_block_well_formed：每个 <details><summary> 结构闭合
7. duplicate_detection：无未合并的同义题（70% 相似视为重复）
8. citation_check：题目引用的论文/数字正确
9. personal_info_leak：无 SJTU / SCUT / JHC / Server5 / /home/scut / /Users/ 等
10. length_target：卷长度合理（不刻意凑题，但 <30 或 >100 警告）

Return JSON:
{
  "verdict": "PASS | WARN | FAIL",
  "checks": {check_name: {"status": ..., "note": ..., "evidence": "file:line"}},
  "blocking_issues": [],
  "warnings": [],
  "summary": "..."
}
"""
)
```

每轮必开 **fresh thread**（绝不调用 `mcp__codex__codex-reply`）。

## 8. 关键文件路径

```
<repo-root>/                                ← 本仓库 (cwd)
├── CLAUDE.md                                ← 本文件
├── README.md                                ← 仓库主页
├── docs/
│   ├── index.html                           ← GitHub Pages 入口 / 主册
│   └── interviews/
│       ├── 01_basics.{md,html}              ← TODO
│       ├── 02_rl_algo.{md,html}             ← TODO
│       ├── 03_vla_il.{md,html}              ← 进行中
│       ├── 04_world_sim.{md,html}           ← TODO
│       └── 05_engineering.{md,html}         ← TODO
└── tools/
    ├── render_html.py                       ← 渲染脚本（原生支持 <details>）
    └── templates/
        └── academic.html                    ← ★ 必用模板（含难度色标 CSS）
```

## 9. GitHub 远程仓库

- GitHub 用户：`WinstonJQ`
- 仓库：`embodied-interview-qa`
- 远程：`https://github.com/WinstonJQ/embodied-interview-qa.git`
- 默认分支：`master`
- 可见性：public
- Pages 配置：Source = `master / docs`
- 发布 URL：
  - 主册：`https://winstonjq.github.io/embodied-interview-qa/`
  - 卷 X：`https://winstonjq.github.io/embodied-interview-qa/interviews/XX_<slug>.html`

**每卷完成后的 push**：

```bash
git add docs/interviews/XX_<slug>.{md,html,review.json} docs/index.html README.md
git commit -m "docs(interviews): add vol-X <slug> (N questions)"
git push origin master
```

## 10. 不要做的事

- ❌ 不要把 cheat sheet 风格的公式推导/代码块塞进答案
- ❌ 不要给题目打公司分类标签（如 `#字节Seed`）
- ❌ 不要强求每卷 50 题（按真实频次走）
- ❌ 不要跳过跨模型审查直接渲染
- ❌ 不要在 codex 审查时复用 threadId
- ❌ 不要忘记 push GitHub 就报告完成
- ❌ 不要写超过 400 字的答案（拆段或精简）
- ❌ 不要保留 SJTU / SCUT / JHC / Server5 / /home/scut / /Users/ 等个人信息（仓库已开源）
- ❌ 不要 force push 到 master
- ❌ 不要复活旧 pi_series / rl_foundations / ppo_sac 内容

## 11. 经验沉淀（来自卷三 VLA/IL 实战）

> 新卷开工前先看一遍。这部分**只随实战更新**，不要凭印象改。

### 11.1 起草阶段（Phase 2）

- 答案体目标 **≤350 字**（不是 200）——实务里 200 字目标常超到 400-500，留 100-150 字 buffer 给后期压缩
- 超过 350 字的题拆成 **答 / 关键对比 / 易错** 3 段；超过 5 段反而冗长
- **每题必有"易错"一句**——这是面试题库区分于教程的核心
- 避免过度精确小数字（如"10-20% 提升"）除非有论文表格依据；笼统说"显著提升"反而更稳

### 11.2 跨模型审查（Phase 3）

- **第 1 轮 prompt 就明确字数阈值**（`≤350 PASS / 350-500 WARN / >500 FAIL`），否则 codex 自己定标准、越往后越苛刻，到 Round 3-5 才发现长度问题
- 每轮 prompt 都要明确告知 "**本轮已应用 N 个 fix**"——避免 codex 重复发现已修问题，浪费轮次
- **fresh thread 不可降级**（绝不调 `codex-reply`）；`sandbox=read-only` 防 codex 误修文件
- 输出 JSON schema 必须在 prompt 明示——否则 codex 散文输出难解析
- **典型收敛 5-6 轮**（实测卷三 6 轮，不是 CLAUDE.md §2 原写的 3-5 轮）；超过 7 轮无收敛 → 停下来报用户
- 审计日志 `.review.json` 必含：每轮 verdict + 主要 fix、`key_facts_verified` 清单、`sources_consulted`（arXiv 链接）——便于下卷参考已验证过的事实

### 11.3 Edit / Bash 操作（避免破坏文件）

- 批量 `replace_all=true` 后再单独 Edit 同一段易失败（上下文已变）——**顺序：先单独 Edit → 最后批量 replace_all**；批量改后必须 `Read` 确认上下文
- `<details>...</details>` 块替换易留多余闭合——前后用 `grep -c "<details" / "</details>"` 验证 N/N 平衡
- 用 HTML 注释 `<!-- TODO §X 待补 -->` 做分批 Write 的 marker，**不要用裸字符串占位符**（会被后续 Edit 误匹配 / 造成内容重复）
- **Anthropic auto-classifier 偶尔限流**——不是 rate limit，是后端 classifier 服务可用性问题，几分钟恢复，不用紧张

### 11.4 题源调研（Phase 1）

- **小红书反爬严，且具身面经绝对数量也不大**——主流用户群（年轻女性 / 生活分享）与具身求职人群（理工科男性）重叠度低；卷三 30 分钟 8 种反爬策略最终只挖到 4 题。**不要假设"小红书是富矿"**
- **跨平台转载（如品玩 AI 教授 AMA）是隐藏好题源**——顶级 lab 教授（许华哲 / 周博宇 / 高飞 / 梁俊卫等）在小红书答疑被科技媒体整理转载，可公开引用，等同高权威性题源
- **一亩三分地全部 login wall**——除非用户能授权登录态，否则**不可作主源**；少量信息靠 SERP 摘要
- **真正的"具身 VLA 算法岗"面经数量仍偏少**（2025-2026 行业窗口期）——题目集中度高（OpenVLA vs RT-2 / DP / ACT 等几题占大半）；**频次 2 + 来源权威**可破例入选主表
- 同义题合并 70% 相似 + 频次 ≥3 入主表 + 频次 1-2 入低频备选——这套规则实务有效

### 11.5 Phase 0 / 用户审过

- 题目清单给用户审时，**按节展开 + 频次降序**，每题一行（不要把 200 字答案也展示）；用户决策成本要低
- 必须提供 ≤4 个明确决策点（如 "全收 / 严守阈值 / 跳过"），避免开放式问"你觉得呢"
- N1/N2/... 这种"破例入选"题要**单独标记**并给出破例理由（来源权威 + 频次说明）
