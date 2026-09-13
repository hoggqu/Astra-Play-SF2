# 等血 TIME 平局的同帧 KO 结算（v9）

长时间训练暴露了 Ken 对 Dhalsim 的一种原生结算顺序：计时归零时双方
都剩 12 点血；稍后游戏在同一帧锁定双方的 TIME 血量为 12，并把 Ken
的原始血量设为 -1。血条仍显示 12，双方胜场仍为 0，最终自然进入下一小局。
v8 要求先看到双方原始血量与锁定血量相等，因而漏掉了这次平局，报错
`new round arrived before previous result was resolved`。这属于结果识别遗漏，
不是 PPO 损失、mini-batch 或 worker 数量导致的错误。

v9 保留 v8 的规则，补充这一种有明确时序证据的路径：

- 前一原生帧计时为 0，双方原始血量、血条与随后锁定的正血量一致；
  TIME 锁定值尚为 0，角色和胜场未变。
- 紧接的一帧，双方锁定血量与血条相等，其中一方原始血量变为 -1，
  另一方仍等于锁定血量。
- 之后锁定血量、血条、原始血量和胜场保持一致，通过成熟结算检查。
- 只有自然到达双方满血、相同胜场的下一局开场，才确认上一局为平局。

缺少相邻帧、之前已有 KO、血量不等、胜场变化或过早到达下一局，均不接受。
不通过缺少胜场推断平局，不修改游戏 RAM。动作、观测、奖励和 PPO 更新不变；
奖励仍使用上一局结算状态，而非下一局补满的血量。

新的构建身份为 `astra.rl-screen-perception128-settlement.v9`，自主训练包为
`astra.rl-managed-perception128.v4`。旧运行目录中的冻结源码与失败记录保留。
从仓库重新构建使用 v9；已经运行的训练及其他主机的冻结包装脚本不会自动更新。

## 验证与恢复

`test_settlement_v9` 覆盖上述时序、反例、原生清零过场以及 v8 和更早版本的回归。
`settlement_v9_native_gate` 使用保存的开局和动作重放失败对局，比较批量执行器与
连续原生执行器的逐帧状态、输入端口和模型观测，并检查平局后的自然下一局输入。
这是训练诊断，不能算正式通关。

```sh
python -m unittest experiments.rl.test_settlement_v9 experiments.rl.test_perception128_draw experiments.rl.test_managed
python -m experiments.rl.managed_builder --output NEW_CODE
python -m experiments.rl.settlement_v9_native_gate --source NEW_CODE/astra_sf2_rl_managed --failure FAILURE.json --dataset DATASET/manifest.json --output NEW_PROBE
```

异常运行仍记为 invalid。若运行结果已发布完整更新后的恢复 checkpoint，可使用
`autotrain --resume OLD_RUN --output NEW_RUN` 续训；它恢复网络和优化器，不恢复
中断时的模拟器局面。显式填写之前使用的 rollout、mini-batch 和 workers；
新的时长预算从续训开始计算，不是补足上一次未用完的时长。
