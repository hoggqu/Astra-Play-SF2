# 128×128 共享策略候选

当前用户入口见[自主训练](AUTOTRAIN.md)：managed包已包含结算v8、按时长停止和CPU/CUDA选择。下文较早的独立派生包保留作实验历史。

本候选把[结构化画面感知](SCREEN_PERCEPTION.md)的策略网络和价值网络都设为两层 128 单元的 Linear/Tanh MLP。观测仍为 4,516 项，动作仍为原 85 种、每次 12 原生帧；观察器、奖励、折扣和 GAE 保持原样。64×64 父版本和模型保留，不把两种容量的成绩混合。

初始 128 模型从随机参数创建，不是宽度迁移；后续训练可继承兼容的完整模型及优化器状态。模型显式记录 `mlp_tanh_pi128x128_vf128x128_v1`，加载、实时策略导出和文件导出同时检查实际 actor、critic 及输出层尺寸。Lua 推理还检查身份和导出层宽，不能把旧 64 模型改个标签后直接使用。

当前新训练推荐使用 `perception128_timing_builder` 构建的独立派生包。它包含[结算 v6](SETTLEMENT_V6.md)、[计时归零后的 KO 结算 v7](SETTLEMENT_V7.md)，以及[批次暂停恢复的输入时序修复](INPUT_TIMING.md)。原始 128、单独 v6 和单独 v7 包均保留作历史对照；它们未包含这次输入时序修复，不建议作为新训练的执行源。

最终时序派生包已通过 85 动作、1,332 原生帧门禁，以及带第 38 次决策边界的 540 帧对照；后者全部 45 条、每条 4,516 维观测及状态、输入端口和事件与连续原生执行一致。这验证执行一致性，不代表该模型已通关。

## 构建和运行

安装项目与 RL 可选依赖后，以下构建只需要公共源码；生成包包含完整父配方、执行源码和 SHA256，可以搬迁。

```sh
python -m experiments.rl.perception128_timing_builder --output .local/rl-code/perception128-timing-001
python .local/rl-code/perception128-timing-001/launch.py initialize \
  --output .local/rl-runs/perception128-initial-001 --seed 856
python .local/rl-code/perception128-timing-001/launch.py batch_train \
  --dataset .local/rl-data/full85-openings-001/manifest.json \
  --init-model .local/rl-runs/perception128-initial-001/ppo-initial.zip \
  --workers 8 --steps 409600 --checkpoint-every 4096 --seed 856 \
  --output .local/rl-runs/perception128-train-001
python .local/rl-code/perception128-timing-001/launch.py native_continuous \
  --model .local/rl-runs/perception128-train-001/ppo-batch.zip \
  --difficulty 3 --attempts 20 --speed fast \
  --output .local/rl-runs/perception128-play-001
```

无人值守循环训练及自然投币评估可使用[版本化训练流程](VERSIONED_CAMPAIGN.md)，将其 `--code` 指向上述新包、`--package` 设为 `astra_sf2_rl_perception128_timing`。

数据集需要预先采集并通过原有校验，以上示例不下载存档。训练支持 1–16 workers，默认 8；步数必须为 `workers × 256` 的整数倍。继续训练时将 `--init-model` 指向上一完整训练 ZIP，保留其完整 Adam 状态，不重新创建优化器。正式游玩仍使用一个冻结模型和自然投币，未经完整原生审计的记录不能称为通关。

## 推理效率与离线验收

新版本仅对首个线性层跳过数值为零的输入，非零项严格保持原索引顺序；后续层仍为普通稠密计算，不改变观测或策略。一次 192 条真实记录观测的本机对照中，4,516 项输入平均 255.26 项非零：稠密 Lua 用时 1.547 秒，跳零计算 0.203 秒，约 7.63 倍。两种 Lua 的全部 logits 精确相等，与 Torch 最大差为 `5.86e-9`，全部 argmax 一致。这是特定样本上的推理耗时对照，不代表完整训练加速倍数。

针对性离线测试覆盖：128 网络 Lua/Torch 推理、稠密/跳零一致性、错误 actor 或 critic 宽度拒绝、实时与文件导出身份、源篡改拒绝、公共干净目录构建后删除原源码并搬迁运行。仿真原生门禁与实际训练结果另行保留；离线通过不等于已学会游戏。
