# 具身智能高频面试题库

> 中文具身智能（Embodied AI）秋招高频面试题库 · 题目来自公开面经，频次合并后入卷，每题答案经独立 AI 二次审查  
> *A Chinese-language interview question bank for Embodied-AI engineering roles (VLA / IL / RL / World Models / Engineering).*

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-2ea44f?logo=github)](https://winstonjq.github.io/embodied-interview-qa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/WinstonJQ/embodied-interview-qa/pulls)
[![Stars](https://img.shields.io/github/stars/WinstonJQ/embodied-interview-qa?style=social)](https://github.com/WinstonJQ/embodied-interview-qa/stargazers)

---

## 📖 在线阅读

**👉 <https://winstonjq.github.io/embodied-interview-qa/>**

打开网页直接看；默认全部折叠，点击题目展开答案；手机端原生支持（地铁公交都能刷）。

### 🖥️ 桌面端预览

<table>
  <tr>
    <td width="33%" align="center" valign="top">
      <img src="images/computer_page_overview_01.png" alt="桌面端主册上半"><br>
      <sub><em>主册 · 上半部（hero + 五卷目录）</em></sub>
    </td>
    <td width="33%" align="center" valign="top">
      <img src="images/computer_page_overview_02.png" alt="桌面端主册下半"><br>
      <sub><em>主册 · 下半部（标签速查 + 使用方式）</em></sub>
    </td>
    <td width="33%" align="center" valign="top">
      <img src="images/computer_page_overview_03.png" alt="桌面端卷子页"><br>
      <sub><em>卷子页 · 左侧 TOC + 「← 返回主册」</em></sub>
    </td>
  </tr>
</table>

### 📱 手机端预览 · 默认折叠 = 自测 + 复盘

看到题目先在脑子里答一遍，答不上来再点开对答案——折叠交互天然就是一种**自测玩法**。

<table>
  <tr>
    <td width="33%" align="center" valign="top">
      <img src="images/mobile_page_overview_01.jpeg" alt="手机端主册"><br>
      <sub><em>手机端主册 · 浏览器直接打开 · 无需 app</em></sub>
    </td>
    <td width="33%" align="center" valign="top">
      <img src="images/mobile_page_overview_02.jpeg" alt="题目折叠态"><br>
      <sub><em>默认折叠 · 只看题目 · 自己先想答案</em></sub>
    </td>
    <td width="33%" align="center" valign="top">
      <img src="images/mobile_page_overview_03.jpeg" alt="题目展开态"><br>
      <sub><em>点开看答案 · 含表格、易错点</em></sub>
    </td>
  </tr>
</table>

---

## 这是什么

2024-2026 是具身智能（Embodied AI）行业的窗口期——VLA、人形、四足、灵巧操作算法岗的招聘量井喷，但**面试题源散乱在牛客、知乎、小红书、一亩三分地等多个平台**，没有一个专门针对这一方向、可在手机上随时翻的题库。

本项目把公开面经里**真实被反复问到的题目**整理成七卷，每题给出 **≤350 字的精简答案 + "易错"一句**——面试前 30 分钟过一遍补盲区，不是看长篇教程的时机。

---

## 七卷目录

| 卷 | 主题 | 题数 | 一句话 |
|---|---|---:|---|
| [一](https://winstonjq.github.io/embodied-interview-qa/interviews/01_basics.html) | **通识基础** | 44 | DL 基本盘 + RL 入门 + 机器人学，一面常从这里开题 |
| [二](https://winstonjq.github.io/embodied-interview-qa/interviews/02_rl_algo.html) | **RL 算法** | 40 | PPO / SAC / TD3 / 离线 RL / RLHF 的设计动机与取舍 |
| [三](https://winstonjq.github.io/embodied-interview-qa/interviews/03_vla_il.html) | **VLA / 模仿学习** | 58 | OpenVLA / RT-2 / π 系列 / Diffusion Policy，投 VLA 岗几乎必问 |
| [四](https://winstonjq.github.io/embodied-interview-qa/interviews/04_world_sim.html) | **世界模型 / Sim2Real** | 31 | Dreamer / V-JEPA / 域随机化 / Isaac Lab，人形和四足公司重点 |
| [五](https://winstonjq.github.io/embodied-interview-qa/interviews/05_engineering.html) | **工程落地** | 39 | VLA 怎么塞进 Jetson、FSDP/ZeRO、teleop 数据飞轮、开放题与系统设计 |
| [六](https://winstonjq.github.io/embodied-interview-qa/interviews/06_legged_control.html) | **腿足机器人控制 / 遥操作** | 48 | 浮动基座、MPC / WBC、RL-locomotion、AVP teleop、HumanPlus、actuator network sim2real |
| [七](https://winstonjq.github.io/embodied-interview-qa/interviews/07_perception_nav.html) | **3D 感知 / SLAM / VLN / ObjectNav / Embodied VLM** | 60 | 点云 / 深度估计 / NeRF / 3DGS / ORB-SLAM3 / VINS / Nav2 / HAMT / DUET / NaVid / Uni-NaVid / VLFM / OpenFMNav / Grounding DINO / SAM 2 / VLMaps，投 VLN / 导航 / VLM 算法岗专用 |

**主表共 317 题**，另含 ~19 题低频备选（每卷末单列）。

---

## 特点

- 🔥 **频次驱动**：题目来自牛客、知乎、小红书、一亩三分地、GitHub 公开面经，同义题合并后**频次 ≥3 才入主表**，杜绝单条噪声面经污染
- ✏️ **短答案**：每题 ≤350 字精简答案 + "易错"一句，面试前快速过；不写公式推导和代码块（要深入学习另找资源）
- 📱 **折叠交互**：默认全部折起只看题目，点击展开答案；基于 HTML5 `<details>` 原生折叠，零 JS，手机端无需任何 app
- 🔍 **跨模型审查**：每题答案经**另一组独立 AI**二次审查（执行者 ≠ 审查者），减少单一模型的事实错误
- 🏷️ **难度标签**：L1 必会 · L2 进阶 · L3 顶级 lab，方便按层次规划复习
- 🏢 **不分公司**：题目按技术主题组织，不打公司标签——同一道题可能在多家面试出现

---

## 使用建议

1. **首次浏览**：直接点开[在线题库](https://winstonjq.github.io/embodied-interview-qa/)，挑你最关心的方向（VLA / RL / 工程落地……）
2. **复习顺序**：按 L1 → L2 → L3，同等级内按频次（🔥×N）从高到低
3. **手机端**：地铁公交也能刷，HTML5 折叠原生支持，无需任何 app
4. **快速过场**：面试前 30 分钟，只看每题的"易错"一句，刷过的题快速复盘

---

## ⭐ 觉得有用？

**点个 Star 是对维护者最大的鼓励** —— 也能让算法群里其他在找具身岗的同学更容易发现这个题库。

如果你在面试中真的被考到了里面的题，欢迎来 [Issue](https://github.com/WinstonJQ/embodied-interview-qa/issues) 留言或私信——这是后续频次更新的一手依据。

---

## 贡献

发现答案错误、想补新题、想质疑某条结论——都欢迎 [开 Issue](https://github.com/WinstonJQ/embodied-interview-qa/issues) 或 [提 PR](https://github.com/WinstonJQ/embodied-interview-qa/pulls)。

**贡献新题的简单格式**（在对应卷的 Markdown 末尾追加即可）：

````markdown
<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×N</span> <b>Q??</b> · 你的题目？</summary>

**答**：精简答案（≤350 字）。

**易错**：一句话点关键陷阱。

</details>
````

或者直接在 issue 里描述题目 + 来源（哪个平台 / 帖子链接），维护者负责合并到对应卷。

**贡献题源要求**：必须来自**公开面经**（牛客 / 知乎 / 小红书 / 一亩三分地 / GitHub awesome-* 等），不接受个人编造的"我觉得会考"题——这条规则是题库可信度的基石。

---

## 项目幕后 · Vibe Coding 实战记录

本项目本身就是一次完整的 **vibe coding** 实践案例——7 卷题库的题源调研、答案起草、跨模型审查、HTML 渲染、Git 发布**全部在维护者睡眠期间由 AI agent 自动完成**。维护者醒来只做最终验收和文案微调。

**调度模式**：

- 主控 Claude Code（Opus 4.7 · 1M context）串行派遣多个 subagent，每个 agent 独立完成 1 卷的完整 7 阶段工作流（调研 → 起草 → 跨模型审查 → 渲染 → 发布）；当前已完成 7 卷题库
- 每个 agent 在 fresh context 里工作，互不污染——避免了"长上下文导致质量衰退"和"需要手动 /clear"两个痛点
- 跨模型审查由独立的 Codex（GPT-5.5 xhigh）通过 MCP 协议完成，每轮 fresh-thread，多轮收敛
- 执行者（Claude）≠ 审查者（GPT-5.5），是题库可信度的核心不变量

**展示的工程实践**：

- **Multi-agent 编排**：subagent prompt 自包含、status code 协议（DONE / WARN / FAIL / BLOCKED）、用户审批回退路径
- **跨模型协作 via MCP**：Codex MCP server 集成、`sandbox=read-only` + `approval-policy=never` 双权限层、隐私脱敏 banlist 硬编码在审查 prompt
- **内容质量 SLO**：频次合并阈值（≥3 入主表）、字数 PASS/WARN/FAIL 阈值、强制收尾的容错策略（>7 轮无收敛即推当前最佳版本，避免单卷阻塞全局）
- **经验沉淀回流**：每完成一卷的实战教训写回 [`CLAUDE.md`](CLAUDE.md) §11，下卷自动继承——形成可复用、可演化的 prompt 流程

**致谢**：本项目的"睡眠期自动化研究"流水线灵感和 HTML 渲染脚本来自 [**ARIS · Auto-claude-code-research-in-sleep**](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) —— 一个把 Claude Code 用作"研究协作 AI"的开源项目，感谢其提供的核心思路。

如果你对"AI 写 + AI 审 + 自动发布"这种 pipeline 感兴趣，欢迎 fork 改造成其他方向题库（金融 / 后端 / 系统设计 / 算法 …），MIT 自由。

---

## License

[MIT](LICENSE) — 自由使用、修改、再分发。

如果在你的项目里引用了本题库的内容，**给本仓库点个 ⭐ Star 是最好的引用方式**。
