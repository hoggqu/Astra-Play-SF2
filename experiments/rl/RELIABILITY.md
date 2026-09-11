# Normal：单模型完整 20 次至少通关 10 次

下一阶段目标是提高完整路线的可靠性。首次通关模型及其原始 20 次记录保持冻结，
新训练与验证写入独立目录。当前目标尚未达成；训练小局胜率不替代通关率。

## 验收方式

- 每个候选固定一份 PPO 权重及执行源码，在 Normal（难度 3）预先指定 20 次尝试。
- 同一 MAME 进程自然投币，失败与通关后均继续自然流程；无 continue、读档、重置或对手比赛内暂停。
- 必须完成全部 20 次，并至少有 10 次通过全部 11 名对手。即使前 10 次全部成功，也须继续完成剩余尝试。
- 控制或证据异常立即停止，保留原始记录，不补投来替换无效尝试。不同候选的成功次数不拼接。
- 检查模型、源码、动作接口、原生时间与完整比赛证据。单独的 CLI 退出码 0 只表示至少一次成功。

这衡量的是该模型在这组自然投币流程中的实测表现，不证明其真实通关概率必然达到 50%。
若反复用该验证流程选择模型，这也是开发评估，不能称为从未接触的独立测试集。

## 自动训练循环

`reliability_campaign` 先验证已经完成训练的初始模型，未达标才继续 PPO 训练。
默认最多比较五个候选：初始模型，加上四次各 409600 决策的训练；每个候选独立完整验证 20 次。
模型参数和优化器继续继承，训练阶段的模拟器与随机数状态重新启动，不能称为精确恢复原轨迹。
运行不需要 Agent 逐局发动作或人工筛选胜局。

训练与验证使用分别冻结的派生包。训练包须使用修正后的 round-chain 输入时序；
验证包须支持 `--all-attempts`。不要修改正在执行的包来添加此参数。
以下路径和包名需要替换为实际构建的 16 动作候选：

```sh
python -m experiments.rl.reliability_campaign \
  --dataset DATASET/manifest.json --output NEW_OUTPUT \
  --init-model COMPLETED_TRAIN/ppo-batch.zip \
  --initial-training-result COMPLETED_TRAIN/result.json \
  --training-code FROZEN_TRAIN_CODE --training-package astra_sf2_rl_round_chain \
  --verification-code FROZEN_VERIFY_CODE --verification-package astra_sf2_rl_round_chain \
  --workers 8 --cycles 5 --steps-per-cycle 409600 --seed 129
```

两套训练可以在独立目录和进程中并行，但合计资源必须遵守主机预算。
首先比较均匀采样与已冻结的加权采样；完整路线的主要失败对手与训练逐轮表现共同决定后续调整。
不同时改动奖励、网络结构和输入特征，以便辨别改动是否有效。

输出 `result.json` 的 `goal_achieved` 仅在完整审计的单个 20 次候选达标后为真。
退出码为 0（达标）、1（候选预算耗尽仍未达标）、2（无效执行）。
`reported_initial_budget_steps` 仅记载调用者声明的历史预算，独立记录初始训练结果的实际步数，
不把中断后未保存的更新计为已继承训练。
