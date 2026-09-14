# 零血量胜者的 TIME 后投技 KO（v12）

一次 Ken 对 Ryu 的训练在游戏已经进入下一名对手时，报出
`Character changed without reset`。角色变化检查正确；实际故障是上一场没有结算。

保留记录显示：第二小局第 5050 帧计时归零，Ken 原始及显示生命都是 0，
Ryu 为 9；已发出的投技继续执行。第 5071 帧 Ryu 生命从 9 变为 -1，
第 5072 帧 Ken 原生胜场从 1 变为 2。Ken 的生命仍为 0，TIME 生命锁定值
均为 0，Ryu 血条自然耗尽。旧 v11 的 TIME 后 KO 分支要求胜者生命大于 0，
因此漏认胜利。游戏自然切换角色后，感知器才报告异常；父进程 EOFError 是连带错误。

v12 只增加“零血量胜者”的严格结算路径：

- 相邻原生帧中，败者必须从正生命变为 -1；胜者原始及显示生命前后都为 0。
- 双方角色、旧比分一致，计时均为 0，TIME 锁定值均为 0；无先前 TIME 授分。
- 后续必须保持生命和计时证据，败者血条只能耗尽，原生胜场只授予唯一胜者。
- 授分满 360 帧、双方落地且胜者进入胜利姿态后，才确认胜负。

双方都是 0、缺失相邻帧、分数冲突、TIME 已锁定、生命回升或过早进入下一小局
均不被这一新路径接受。保留角色变化检查，也不将普通异常转成有效胜利。
识别原因记录为 `rl-native-zero-winner-time-ko-v12`。

构建身份为 `astra.rl-screen-perception128-settlement.v12`，自主训练包为
`astra.rl-managed-perception128.v9`。模型、Adam、观测、动作、奖励公式及出招节奏不变；
旧源码包和失败记录原样保留，新运行使用新目录。修复结算会让此前丢失的合法
终局样本进入训练，因此不同结算版本的长训结果须注明版本，不能假定逐步一致。

```sh
python -m unittest experiments.rl.test_settlement_v12 experiments.rl.test_perception128_draw experiments.rl.test_managed
python -m experiments.rl.managed_builder --output NEW_CODE
python -m experiments.rl.settlement_v12_native_gate --source NEW_CODE/astra_sf2_rl_managed --failure WORKER/training/rl-batch-unresolved-settlement.json --dataset DATASET/manifest.json --output NEW_PROBE
```

原生探针需要留存的失败动作序列、worker manifest 和相容训练开局。这些历史数据
不入 Git。它重放整场比赛，比较批量与连续执行的每帧状态、实际输入端口和模型观测，
检查第 5432 帧的成熟 2:0 胜利、两次终局 transition 及结算后的训练重置。
它是训练诊断，不是无存档通关成绩。

## 本机检查记录（2026-09-14）

545 项 RL 测试通过；主项目 76 项测试中 75 项通过、1 项平台限定跳过。
wheel 构建和隔离目录安装通过。MAME 0.288 重放 5432 帧、358 次决策，
批量与连续执行的全部状态、实际输入端口及观测一致，原失败记录中的 3431 帧
逐帧匹配；第 5432 帧正确确认 Ken 2:0，正常发布两次终局 transition。

12 workers、MPS、16384/256 的短续训完成 16384 次决策，
完整恢复旧 checkpoint 和 Adam，并保存更新后的完整模型；采样约 61.8 秒，
PPO 更新约 4.8 秒。这是修复验证，不是完整 A/B 或通关率结论。

短续训之后的一次 Normal 自然投币自动验证通过，击败 11 名对手，原生审计通过；
这是单次 `rl_gameplay_clear`，未作结局画面人工认证，也不能代表稳定通关率。
