# Embodied-AI Interview QA Bank

> 中文具身智能（Embodied AI）秋招高频面试题库

聚焦 **VLA / 模仿学习 / RL / 世界模型 / 工程落地** 五大方向。每题包含：

- 题目（折叠式，默认收起）
- ≤200 字精简答案（便于记忆）
- 难度标签（L1 必会 / L2 进阶 / L3 顶级 lab）
- 频次徽章（🔥×N — 同义题合并后的出现次数）

题目来自牛客 / 知乎 / 小红书 / 一亩三分地 / GitHub 公开面经，**频次 ≥3 才入卷**。

## 五卷结构

| 卷 | 主题 | 状态 |
|---|---|---|
| 卷一 通识基础 | DL / RL 基础 / 机器人学 | TODO |
| 卷二 RL 算法 | PPO / SAC / TD3 / Offline RL | TODO |
| [**卷三 VLA / 模仿学习**](docs/interviews/03_vla_il.html) | OpenVLA / π0-π0.7 / Diffusion Policy / Flow Matching / BC / DAgger / ACT / FAST / OpenVLA-OFT / RECAP | **DONE · 58 题** |
| 卷四 世界模型 / Sim2Real | Dreamer / Genie / DR / RMA | TODO |
| 卷五 工程落地 | 数据 / 部署 / 系统设计 / 开放题 | TODO |

题量灵活，由真实调研频次决定（卷三预估 50-80 题，卷四可能 30-50 题）。

## 部署

纯 **GitHub Pages 静态托管**，无后端依赖。

访问主册：<https://winstonjq.github.io/embodied_offer/>

HTML 单文件自包含：MathJax + highlight.js 从 CDN 加载（也支持 `--offline`）。

## 工作流

每卷独立跑：

```
Phase 1 调研  → 跨平台爬取 + 同义题合并 + 频次统计
Phase 0 审核  → 题目清单交用户审过删/加/换
Phase 2 起草  → Markdown（含 <details> 折叠块）
Phase 3 审查  → Codex GPT-5.5 xhigh fresh-thread 多轮（10 项检查）
Phase 4 渲染  → academic 模板 HTML
Phase 5 索引  → 更新 docs/index.html
Phase 6 推送  → push GitHub
```

**关键不变量**：执行者（Claude） ≠ 审查者（GPT-5.5）；每轮 fresh thread；隐私脱敏 banlist 硬编码在审查 prompt。

详见 [`CLAUDE.md`](CLAUDE.md)。

## License

MIT — 自由使用、修改、再分发。
