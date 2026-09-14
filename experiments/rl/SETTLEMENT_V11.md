# 普通 KO 后的最后一次计时递减（v11）

Ken 对 Vega 的一次训练中，第 1306 帧 Vega 原始生命变为 -1，计时为 60；
第 1307 帧计时减为 59，同时 Ken 获得原生胜场。Vega 血条随后自然耗尽，
双方落地，Ken 进入胜利姿态，游戏最终进入比分 1:0 的第二小局。
v10 的普通 KO 路径要求计时从第一帧 KO 起不再变化，因而取消结算候选，
在下一局报出 `new round arrived before previous result was resolved`。
父进程的 EOFError 是 worker 退出后的连带错误。

v11 保留 v10 及此前结算规则，只为普通 KO 增加严格的计时边界：

- 已记录非 TIME 的单方 KO，并保留实际相邻的终止帧状态。
- 只允许 KO 后紧接的一帧将计时减 1，且减后仍大于 0；之后必须保持该值。
- 前后 TIME 锁定值均为 0，败方仍为 KO，胜方生命、角色和显示生命保持有效一致。
- 原生胜场必须符合唯一胜者，败方血条只允许继续耗尽。
- 仍须等待原生胜场至少 360 帧、双方落地及有效胜利姿态，才确认结果。

缺失相邻证据、延迟递减、再次变化、增加、减多格、减至零以及生命或胜场冲突
均不接受。原始 `stop.timer=60` 与成熟状态 `settled.timer=59` 分别保留，
计时边界证据记入 `ko_award.timer_tail`，不修改 RAM 或原始记录。

新结算身份为 `astra.rl-screen-perception128-settlement.v11`，自主训练包为
`astra.rl-managed-perception128.v6`。模型结构、观测、动作、奖励和 PPO 参数不变。
历史包及运行中的冻结源码不被改写。

## 检查与恢复

```sh
python -m unittest experiments.rl.test_settlement_v11 experiments.rl.test_perception128_draw experiments.rl.test_managed
python -m experiments.rl.managed_builder --output NEW_CODE
python -m experiments.rl.settlement_v11_native_gate --source NEW_CODE/astra_sf2_rl_managed --failure WORKER/training/rl-batch-unresolved-settlement.json --dataset DATASET/manifest.json --output NEW_PROBE
```

上述原生探针专门重放这次 Vega 案例，需要留存的 worker manifest、失败动作记录和
训练数据；它们不纳入 Git。探针比较批量与连续执行的全部状态、实际输入端口及观测，
并检查自然第二小局的历史重置和首个 12 帧输入，不作为正式通关证据。
单元测试覆盖双方胜负、整场终局、严格成熟等待、时序反例及全部旧结算合同。

使用当前仓库入口并省略 `--code`，即可构建 v11。通过 `--resume OLD_RUN`
恢复其已发布的完整 checkpoint 和优化器，输出到新目录；显式保留原来的
workers、rollout 与 mini-batch。异常旧运行继续记为 invalid。

## 本机验证记录（2026-09-14）

535 项强化学习测试全部通过；76 项主项目测试中 75 项通过、1 项跳过。
macOS / MAME 0.288 的 Vega 原生重放通过：2,195 帧、125 次决策的
批量与连续执行状态、实际端口和观测完全一致。原留存终止记录的 362 帧
逐帧匹配。第 1,667 帧确认 Ken 胜、比分 1:0，随后自然进入第二小局；
该局历史重置与第一个 12 帧输入检查通过。

这证明留存异常已修复，不代表覆盖所有游戏结算边界。跨平台实机结果须另行验证。

另外完成一次 12 workers、16384/256 的短续训：从中断运行的完整 checkpoint 恢复，
Adam 更新计数从 8628 到 8632，保存 16384 次决策后的模型。随后一次自然投币
验证正常结束（击败 3 人后负于 Ryu，原生审计通过），未出现结算异常。
