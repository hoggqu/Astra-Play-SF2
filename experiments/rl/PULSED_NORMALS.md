# 普通技末帧松键候选

这是动作语义修正实验，接口为 `ken_actions16_pulsed_normals_v2`，不能冒充原来的 `ken_actions16_lp_mp_uppercut_v1`。共享 PPO 网络仍有 344 个输入、16 个动作，每 12 个原生帧决策一次，没有 V4 战斗策略 fallback。

仅 action 6–11（LP、HP、蹲 LP、蹲 HK、MK、HK）在零起始 frame 11 松开攻击按钮；蹲技 8/9 保留 D，其余普通技末帧为空输入。frame 0–10、方向、其他动作、网络计算、奖励、采样与结算完全继承父包。相同普通技连续被选中时，每个宏都有新的松键边沿；硬直期间仍不保证每次宏都能成功出招。

## 构建与迁移

从包含冻结 `src/` 的 round-chain 父包派生。父包可以含已审计的 settlement revision。训练和完整验证应分别从各自的父包构建，保留其 CLI 功能，尤其完整 20 次验证所需的 `--all-attempts`。

```sh
python -m experiments.rl.pulsed_normals_builder \
  --source .local/rl-code/chain-parent/astra_sf2_rl_round_chain \
  --output .local/rl-code/pulsed-train
python -m experiments.rl.pulsed_normals_builder \
  --source .local/rl-code/reliability-parent/astra_sf2_rl_round_chain \
  --output .local/rl-code/pulsed-verify
python -m experiments.rl.pulsed_normals_migrate \
  --model .local/models/actions16-v1.zip --output .local/models/pulsed-init
```

模型迁移只改 ZIP 的 `data` 成员内两个键：`astra_action_interface` 和新增 `astra_pulsed_normals`。其余 ZIP 成员逐字节保留，并在重新载入后验证权重、全部 Adam 状态、训练步数及更新计数一致。没有重建 optimizer，也没有重置已有优化器年龄。模型数值函数完全不变，但输出标签对应的按键语义改变，因此既有成绩不适用于新接口。

新包保存 `parent-source/` 下完整的原父包与祖先 manifest。验证器先按旧 chain 规则核实父包，再从父字节重新计算唯一允许的派生内容，不能靠重填新文件 hash 允许额外修改。新依赖、父证据、冻结生产源都进入身份检查；native campaign 同时冻结这些依赖。训练启动前检查候选身份，正式验证启动前和结束后检查身份。旧 ZIP 在新模型检查、导出及 Lua 推理入口均被拒绝。

## 运行

每个生成目录有 `launch.py`，自动把对应冻结 `src/` 与隔离包放在导入路径前。使用同一个已安装 RL 依赖的 Python；每项输出目录都必须是新的。

```sh
# 第一轮可先验证迁移模型，不混入任何 PPO 更新。
python .local/rl-code/pulsed-verify/launch.py native_continuous \
  --model .local/models/pulsed-init/ppo-pulsed.zip \
  --output .local/rl-runs/pulsed-initial-full20 \
  --difficulty 3 --attempts 20 --all-attempts --speed fast

# 后续训练从指定迁移模型完整恢复权重和 Adam 状态。
python .local/rl-code/pulsed-train/launch.py batch_train \
  --dataset .local/data/train-dataset.json \
  --init-model .local/models/pulsed-init/ppo-pulsed.zip \
  --output .local/rl-runs/pulsed-train-001 \
  --workers 4 --steps 409600 --block 64 --seed 316

# 小批量训练/自然投币诊断流程；不是完整 20 次可靠性认证。
python .local/rl-code/pulsed-train/launch.py native_campaign \
  --dataset .local/data/train-dataset.json \
  --init-model .local/models/pulsed-init/ppo-pulsed.zip \
  --output .local/rl-runs/pulsed-campaign-001 \
  --workers 4 --steps-per-cycle 409600 --cycles 2 --verification-attempts 3
```

默认继承的批训练限制是 1–8 workers、每 worker 256 决策后 PPO 更新。调整并行上限或 rollout 长度需要另建有独立身份的派生包。

离线测试覆盖全部 16×2×12 波形、连续相同普通技的末帧释放、父包与派生文件篡改拒绝、旧接口拒绝、迁移前后权重/Adam 精确一致。它们不代替原生时序和实际输入端口门禁。门禁及 disposable smoke 通过后，再开展正式自然投币；smoke 权重不能悄悄替代指定冻结模型。

每个候选完整 20 次成绩单独计算，保留此前失败，不拼接不同模型或旧接口批次，不用提前通关结束代替全部 20 次。没有中途读档、continue、重置或修正策略。对照时保持初始权重、训练数据、随机种子、预算和 PPO 参数一致；不能仅凭抽样动作频率断定改动一定提高通关率。
