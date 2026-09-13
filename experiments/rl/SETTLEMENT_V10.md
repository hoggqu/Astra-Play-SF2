# 零血量延迟 KO 的结算（v10）

一次 Ken 对 Guile 的训练在第二小局结束时报告
`new round arrived before previous result was resolved`。当时计时已归零，
Ken 剩 4 点血，Guile 原始血量和显示血量都是 0。随后 Guile 原始血量变为 -1，
下一帧 Ken 获得原生胜场，比分从 0:1 变成 1:1。v9 的延迟 KO 检查要求
前一帧双方血量都大于 0，遗漏了这个边界，最终在第三小局开场时报错。

v10 保留此前规则，只补充满足以下证据的路径：

- 前后是相邻原生帧，计时和 TIME 血量锁定值均为 0；角色未变，之前没有胜场增加。
- 败方原始血量从 0 变成 KO 标记 -1；前一帧显示血量为 0，后一帧为 0 或 -1。
- 胜方仍有正血量，且与前一帧一致。
- 之后必须出现唯一正确的原生胜场增加，并通过完整的 360 帧成熟检查。

零血量本身不会被判为 KO。缺失相邻帧、双 KO、胜场冲突、TIME 锁定或
过早进入下一局均不能绕过原有检查。实现只读游戏状态，不修改 RAM。
动作、模型观测、奖励公式、采样和 PPO 超参数不变。

构建身份更新为 `astra.rl-screen-perception128-settlement.v10`，自主训练包为
`astra.rl-managed-perception128.v5`。旧运行的冻结源码、失败记录和模型保持原样。

## 验证与续训

`test_settlement_v10` 覆盖双方胜负、同帧胜场增加、成熟等待和上述反例，
并在新适配器上重跑 v9 及更早版本的结算合同。
`settlement_v10_native_gate` 从原始第一小局开局和完整动作记录重放 Guile 失败，
比较批量与连续执行的逐帧状态、实际输入端口和模型观测，检查第三小局历史重置
以及第一个 12 帧动作。该工具专门用于此留存案例，属于训练诊断，不是正式通关。
原始私有失败数据、存档和 ROM 不包含在 Git 中。

```sh
python -m unittest experiments.rl.test_settlement_v10 experiments.rl.test_perception128_draw experiments.rl.test_managed
python -m experiments.rl.managed_builder --output NEW_CODE
python -m experiments.rl.settlement_v10_native_gate --source NEW_CODE/astra_sf2_rl_managed --failure WORKER/training/rl-batch-unresolved-settlement.json --dataset DATASET/manifest.json --output NEW_PROBE
```

原始失败文件旁须保留该 worker 的 `manifest.json`（位于 `WORKER/`）。工具核验
训练存档哈希，并且只在诊断起点加载一次存档，其后自然进入下一小局。

续训使用当前仓库入口，省略 `--code` 以构建 v10，再传入
`--resume OLD_RUN --output NEW_RUN`。恢复已发布的完整更新 checkpoint 和优化器，
不恢复中断的模拟器局面；显式填写原来的 workers、rollout 和 mini-batch。
旧运行仍记为 invalid，新预算从续训开始计算。已经冻结在其他目录的旧入口需要
重新部署，Git 更新不会改写其源码。

## 本次验证记录（2026-09-14）

- 强化学习测试 530 项通过；主项目测试 76 项，75 项通过、1 项跳过。
- 从只含 Git 暂存文件的干净目录成功构建新训练包，标准 wheel 构建、安装与 CLI 检查通过。
- macOS 与 Linux MAME 0.288 均重放通过：6,864 原生帧、466 次决策，
  批量与连续执行的全部状态、实际输入端口和观测一致。
- 第一小局仍为负；第二小局在第 6,391 帧确认 Ken 胜，比分 1:1，
  随后自然进入第三小局，历史重置及 12 帧输入检查通过。

该结果验证了留存异常的修复，不是新权重通关率或所有稀有结算情况的保证。
