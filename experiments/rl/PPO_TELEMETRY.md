# PPO 更新遥测

新的 `batch_train` 在每次实际 `model.train()` 返回后，从 SB3 内存 logger
读取当前更新指标，写入 `result.json` 的 `iterations[].ppo`；同一 iteration
也会打印到日志。仅 benchmark 的迭代没有 PPO 更新，因此不填造这些指标。
此改动只影响之后构建的运行快照，不修改正在执行的冻结包。

保留 SB3 字段 `entropy_loss`、`approx_kl`、`clip_fraction`、`value_loss`、
`explained_variance`、`policy_gradient_loss`、`learning_rate`、`n_updates`。
`policy_entropy = -entropy_loss` 是训练中 minibatch 熵的均值，不能当作
最终 checkpoint 在固定观察集上的新测量。SB3 的 `approx_kl`、损失等沿用其
更新内聚合定义；例如 2.7.1 的 `approx_kl` 对最后执行的 epoch 求均值。

所有数值转换为有限 JSON 标量。缺失、非有限或非实数标量写为 `null`，
原因在 `unavailable` 中明确标为 `missing`、`nonfinite` 或
`not_a_real_scalar`。`explained_variance` 在目标方差为零时可能非有限，
不能擅自补成 0。原有 `max_logprob_error` 衡量采样时 Lua 与 Torch 的
old-logprob 一致性，**不是** PPO 更新前后的 KL。

## 历史数据不能补造

旧结果只保留步数、耗时、一致性误差和部分 chain 计数，没有上述更新指标。
SB3 ZIP 保存参数、优化器和算法配置，但排除 logger 与 rollout buffer；
因此不能从旧 ZIP 恢复原始逐更新的熵、KL、clip fraction、value loss 或
explained variance，不能给历史 iteration 回填伪指标。

已有 checkpoint 可以在同一份保留的训练观察上重新测量动作熵、动作概率、
两个 checkpoint 的分布差异；这属于另一个离线诊断，必须说明观察来源及
模型身份，不能冒充原来的 on-policy 更新统计。旧小局日志仍可用于按对手、
轮次分析胜负、回报与决策数。缺失的原始 old-logprob、value、GAE 目标与
更新内策略版本，使原始 PPO 损失通常不可重建。

这些遥测用于解释学习变化，不能代替同一冻结模型的完整自然投币验证，
也不能仅凭少数无通关尝试断言策略已收敛或观察存在性能硬上限。
