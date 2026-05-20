# Embodied-AI-in-Offer

> 中文具身智能秋招速查手册集合 · 仿照 [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer) 范式重做。

聚焦 **强化学习 / 机器人 / VLA / π 系列 / 世界模型** 五大方向，每份手册都包括：

- §0 TL;DR Cheat Sheet
- §1–§N 公式推导 + 直觉解释
- §N+1 From-scratch PyTorch 代码（要能跑）
- §N+2 25 道 L1/L2/L3 高频面试题（含可折叠答案）
- §A 一手资料引用

## 工作流

```
/interview-cheatsheet <topic>
  ├─ Step 1: 规划 12-14 节结构
  ├─ Step 2: 起草 docs/tutorials/<slug>_tutorial.md
  ├─ Step 3: 跨模型审查 — codex gpt-5.5 xhigh （fresh thread / 轮）
  │          10 项检查（公式 / 代码 / 答案 / 引用 / 表格 / callout / heading / 完整性 / 长度 / 隐私）
  ├─ Step 4: 修正 FAIL 项 → 新 thread 再审，循环至 PASS
  ├─ Step 5: 渲染 HTML — python3 tools/render_html.py
  └─ Step 6: 写 .review.json 审计日志
```

**关键不变量**：执行者（Claude） ≠ 审查者（GPT-5.5）；每轮审查开 fresh thread；隐私脱敏 banlist 硬编码在审查 prompt 里。

## 部署

纯 **GitHub Pages 静态托管**，无后端依赖：

```
Settings → Pages → Source: main / docs
```

访问地址：`https://winstonjq.github.io/embodied_offer/tutorials/<slug>_tutorial.html`

HTML 单文件自包含：MathJax + highlight.js 通过 CDN 加载（也可用 `--offline` 全本地化）。

## 已完成

| 主题 | Markdown | HTML | 审查轨迹 |
|---|---|---|---|
| **π 系列 (π0 → π0.7)** 面试 Cheat Sheet | [`pi_series_tutorial.md`](docs/tutorials/pi_series_tutorial.md) | [`pi_series_tutorial.html`](docs/tutorials/pi_series_tutorial.html) | [`.review.json`](docs/tutorials/pi_series_tutorial.review.json) — 5 节点路线 Lean+ 大纲（KI 作为 π0.5 子节），含 OpenVLA/RDT-1B/RT-2 横向比较 |
| **强化学习基础 (V1 of 3)** 面试 Cheat Sheet | [`rl_foundations_tutorial.md`](docs/tutorials/rl_foundations_tutorial.md) | [`rl_foundations_tutorial.html`](docs/tutorials/rl_foundations_tutorial.html) | [`.review.json`](docs/tutorials/rl_foundations_tutorial.review.json) — MDP → DQN → PG → A2C；含独立 §4 on/off-policy + importance sampling；4 轮跨模型审查 PASS，27 处修正 |
| **策略优化与连续控制 (V2 of 3)** 面试 Cheat Sheet | [`ppo_sac_tutorial.md`](docs/tutorials/ppo_sac_tutorial.md) | [`ppo_sac_tutorial.html`](docs/tutorials/ppo_sac_tutorial.html) | [`.review.json`](docs/tutorials/ppo_sac_tutorial.review.json) — TRPO/PPO/GAE/DDPG/TD3/SAC + PPO in RLHF；写作风格调整为先直觉后公式 + VAE 类比；3 轮跨模型审查 PASS，11 处修正 |

## 路线图（18 个主题）

**Tier 1 — RL 与决策基础（3 册系列）**
- [x] **V1 强化学习基础**（MDP / Bellman / DP / MC-TD / IS / Q-learning / DQN / PG / A2C）✅
- [x] **V2 策略优化与连续控制**（TRPO / PPO / GAE / DDPG / TD3 / SAC + PPO-RLHF）✅
- [ ] V3 离线 RL + 模仿 + LLM Post-Training（BC / Offline RL / DPO / RLHF / RLVR / GRPO）

**Tier 2 — 机器人学与控制**
- [ ] 机器人学速查（运动学 / 动力学 / 雅可比）
- [ ] MPC & 轨迹优化（iLQR / MPPI / CEM）
- [ ] Sim-to-Real（Domain Randomization / SysID / RMA）

**Tier 3 — VLA & 机器人大模型**
- [x] **π 系列**（π0 → π0.7）✅
- [ ] VLA 综述（RT-1 / RT-2 / RT-X / OpenVLA）
- [ ] Diffusion Policy / 动作扩散
- [ ] 机器人表征学习（R3M / VC-1 / Voltron / MVP）
- [ ] 触觉 & 多模态感知

**Tier 4 — 世界模型与生成式规划**
- [ ] 世界模型基础（Dreamer V1-V3 / TD-MPC2）
- [ ] 大世界模型（Genie / GAIA-1 / Sora-as-WM / V-JEPA）
- [ ] MBRL & Planning（PlaNet / MuZero）

**Tier 5 — 任务域 & 仿真**
- [ ] Manipulation 基准（RLBench / CALVIN / LIBERO / ManiSkill）
- [ ] 导航 & VLN（R2R / VLN-CE / Habitat）

## 致谢

构建思路 & 渲染脚本均来源于 [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（MIT License）。

## License

MIT — 自由使用、修改、再分发。
