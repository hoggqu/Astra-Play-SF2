# 固定学习率与余弦衰减对照

每台只跑一组：本机 A（MPS），远程 B（CUDA），完成后停止。
两边从同一完整 PPO checkpoint 出发，保留 Adam 动量、共享 128×128 网络、
原训练场景池、自适应采样与 Sagat 抽样倍率 2；不修改动作、奖励、观测或 Lua。

| 参数 | A | B |
|---|---|---|
| 优化器 | Adam，原 checkpoint 参数 | 相同 |
| 学习率 | 固定 0.0001 | 余弦 0.0001 → 0.00003 |
| 总训练决策 | 819200 | 819200 |
| 训练轮次 | 2×409600 | 相同 |
| workers / rollout / mini-batch / epochs | 12 / 16384 / 256 / 4 | 相同 |
| 每轮中间评估 / 最终固定模型评估 | 3 / 40 次自然币 | 相同 |

入口 `python -m experiments.rl.schedule_comparison --settings SETTINGS.json --output NEW_RUN`。
settings 字段沿用 sampling_comparison（arm、model、model_sha256、dataset、
dataset_sha256、device、config），不接受多组或交换组。最终选择完整预算结束的
最后 checkpoint，不根据中途成绩择优。中间评估和短测不混入最终 40 币统计。

## 调度和续训

新 managed 身份 v11，结算仍为 v12。学习率按全局已完成决策计算：
`end + (start-end) * (1 + cos(pi * min(completed/total, 1))) / 2`。
每次更新使用包含当前 rollout 的进度；第一次更新接近起始值，最后一次到达终值。
成功更新后才递增进度，并与模型、Adam 状态一起保存。每轮重启训练进程不会
重新开始衰减；超过预算后保持终值，没有 warm restart。

通用 autotrain 参数：`--learning-rate 0.0001 --lr-schedule cosine --lr-end 0.00003`。
默认总预算为按完整 rollout 向上取整的每轮步数乘以 rounds；也可显式指定
`--lr-schedule-steps 819200`，仅按 hours 运行时必须给这个决策预算。
省略调度和学习率参数续训会继承 checkpoint 的曲线及进度。
显式指定完全相同的曲线参数也会接着该进度；指定不同曲线会开始新的衰减阶段。
`--lr-schedule constant --learning-rate VALUE` 明确改为固定值；只覆盖标量
learning-rate 且不指定调度也会改为固定值。固定模式不指定值则使用保存的当前值。

调度更新 SB3 的 lr_schedule 本身，因此不会被 PPO.train() 内部的学习率设置覆盖。
日志记录各次更新的实际学习率、计划与累计进度；comparison 校验跨轮进度和曲线。

主要比较最终 40 币通关率及各对手失败分布。每组单个模型、连续投币序列与不同
GPU 后端不能给出严格因果证明；结果用于筛选。正常训练无需 AI 参与，定时脚本
只在异常或完成时通知，结束后做一次结果分析。
