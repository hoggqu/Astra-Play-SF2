# 固定弱项采样实验

此候选从已经冻结的 **16 动作 round-chain 包**派生，只改变训练
`BatchEnv.reset_choice()` 选择对手的概率。选中对手后，checkpoint 和 lead
仍按父包原有方式均匀选择；后续小局自然继续。网络、奖励、动作宏、结算及
连续验证玩法沿用父包。所有 Lua 文件保持字节一致。

先完成所用父包的原生时序、跨小局和结算验证，再构建实际实验。builder 不会
升级默认实现，也不会启动 MAME；生成清单仍标为未进行原生验证的 candidate。
相同 chain schema 的已冻结 v4 包可作为 `--source`，不依赖某个历史目录。

## 权重配置

创建本地 JSON，例如 `.local/rl-config/opponent-weights.json`：

```json
{
  "schema": "astra.rl-fixed-opponent-sampling.v1",
  "probabilities": {
    "0": 0.10,
    "1": 0.10,
    "2": 0.05,
    "3": 0.10,
    "5": 0.10,
    "6": 0.10,
    "7": 0.10,
    "8": 0.10,
    "9": 0.10,
    "10": 0.10,
    "11": 0.05
  },
  "formula": {"kind": "user_supplied"},
  "statistics_source": {"kind": "illustrative_configuration", "training_only": true}
}
```

这只是格式示例，不是推荐训练权重。ID 对应：0 Ryu、1 Honda、2 Blanka、
3 Guile、5 Chun-Li、6 Zangief、7 Dhalsim、8 Bison、9 Sagat、10 Balrog、11 Vega。
必须明确提供全部 11 个对手，概率均为有限正数且总和为 1；不做隐式归一化，
也不允许缺失对手的训练池。`formula` 和 `statistics_source` 必须是对象。

一种可比较的方案是 `p_i = α/11 + (1-α) f_i² / Σf_j²`，其中
`f_i = L_i/(W_i+L_i+D_i)`，例如取 `α=0.4`。在训练池统计中计算好概率再写入
JSON，并在 metadata 中记录公式、统计范围及来源哈希；builder 不自动读历史
成绩或随训练更新权重。公共源码不包含私人历史数据或预设弱项权重。
若所有失败率均为零，平方和为零，应直接使用均匀概率 `p_i=1/11`，避免除零。

## 构建及运行

从源码 checkout 操作，先按照 [RL README](README.md) 安装可选依赖、配置
MAME 和兼容 ROM，并激活对应 Python 环境。以下路径均为示例，替换为实际的
已验证父包、数据集及模型；每个输出目录必须全新。

```sh
python -m experiments.rl.weighted_sampling_builder --source .local/rl-code/chain-approved/astra_sf2_rl_round_chain --weights .local/rl-config/opponent-weights.json --output .local/rl-code/weighted-example
```

生成的包名为 `astra_sf2_rl_weighted_chain`。没有额外 launcher；将**构建输出目录**
加入当前终端的 `PYTHONPATH`，其值不是内部包目录。

macOS/Linux：

```sh
export PYTHONPATH="$PWD/.local/rl-code/weighted-example${PYTHONPATH:+:$PYTHONPATH}"
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = (Resolve-Path .local/rl-code/weighted-example).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

单次训练，并单独进行三个自然投币尝试：

```sh
python -m astra_sf2_rl_weighted_chain.batch_train --dataset .local/rl-data/normal-all/manifest.json --init-model .local/rl-runs/parent-train/ppo-batch.zip --output .local/rl-runs/weighted-train-example --workers 8 --steps 204800 --block 64 --seed 42
python -m astra_sf2_rl_weighted_chain.native_continuous --model .local/rl-runs/weighted-train-example/ppo-batch.zip --output .local/rl-runs/weighted-play-example --difficulty 3 --attempts 3 --speed fast
```

或者使用自动 campaign，执行两批训练、每批结束后三个自然投币尝试：

```sh
python -m astra_sf2_rl_weighted_chain.native_campaign --dataset .local/rl-data/normal-all/manifest.json --init-model .local/rl-runs/parent-train/ppo-batch.zip --output .local/rl-runs/weighted-campaign-example --workers 8 --cycles 2 --steps-per-cycle 204800 --block 64 --seed 42 --verification-attempts 3
```

单次训练与 campaign 是两种入口，无需重复执行。连续验证不使用训练采样权重：
自然路线、argmax 神经策略、无 continue／读档的条件保持原样。执行错误保留为
invalid；有效失败不删除，也不通过重试覆盖。不要同时让两个任务占用同一个
输出目录或控制同一个 MAME。

`--init-model` 加载父包兼容的 16 动作 PPO ZIP，继续网络参数及 ZIP 中的
优化器状态，不再次执行 15→16 迁移。campaign 后续周期继续使用上一周期 ZIP。
父训练器会重新初始化每批的环境、随机序列和训练步数/进度调度，因此这不等于
恢复一段完全相同的中断轨迹。结果中的 `optimizer_initialization` 记录初始
优化器状态，`opponent_sampling` 记录精确权重、公式、统计摘要和哈希。

## 对照与审计

用同一个初始 ZIP、同一训练数据集、seed、workers、block 和决策预算分别运行
父包的均匀采样与新包。保持网络、奖励、动作、结算版本及验证配置相同。每组
至少保留完整的训练记录和固定模型的自然投币结果；只使用训练池统计制定权重，
不为选权重查看 holdout。权重冻结后再跑下一组实验。

比较全部 11 个对手及 R1/R2/R3 等实际轮次，既看弱项改善，也看原有强项是否
退步。对手选择概率不是决策时间份额，长对局会占用更多预算；相同决策预算也
不保证相同完成小局数。正概率保证支持范围，有限预算下仍不能保证每个对手
出现固定次数。共享网络可能遗忘，均匀保底只能减少风险，不能保证胜率不降。

构建保存 chain 和原 actions16 祖先清单。新 helper、冻结配置和三个清单文件
进入 campaign 源码冻结检查；训练结果记录执行源码哈希，连续验证仍必须通过
原生时序与接口审计。复制包时保留整个构建目录及清单，不能只复制 Python 包。
修改权重需要生成全新的构建和运行目录，不能改正在使用的配置。

离线验证入口：

```sh
python -m unittest experiments.rl.test_weighted_sampling -v
```

这些测试不运行 MAME，只验证概率约束、reset 参数、祖先与配置篡改拒绝、
审计依赖覆盖及游戏 Lua 保持不变；它们不是原生游戏验证证据。
