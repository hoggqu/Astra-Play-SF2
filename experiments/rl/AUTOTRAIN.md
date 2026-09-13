# 自己运行训练和观看模型

当前入口是 **4516 维结构化可见信息、85 动作、策略与价值网络各128×128的 PPO**。
训练、采样、更新、保存、自动自然投币评估与报告生成均不需要 AI Agent。
该接口的成绩与旧16动作冻结模型分开；当前128模型尚未取得已认证的完整通关。

## 通用安装

在本实验分支的源码根目录，用 Python3.10+ 创建虚拟环境，并安装：

```sh
python -m pip install -e . -r experiments/rl/requirements.txt
astra-sf2 configure --mame /path/to/mame --rom-dir /path/to/roms
astra-sf2 doctor
```

Windows 用所创建环境的 `Scripts/python.exe`，macOS/Linux 用 `bin/python`；其余Python参数相同。
需要 MAME0.288、兼容的 `sf2` World910522 ROM 和 Normal训练开局数据集。已有数据可直接
用 `--dataset` 指定；没有数据时，先用项目的 `python -m experiments.rl.collect --difficulty 3
--opponents all --samples 4 --output NEW_DATASET --max-seconds 1200` 采集。采集会运行游戏并生成
训练存档；训练只读取manifest的train部分，开发/保留集不当训练开局使用。

## 运行多久，或跑多少轮

```sh
# 跑两小时，结束后自动评估最终模型并生成报告
python -m experiments.rl.autotrain --dataset DATASET/manifest.json --hours 2 --output RUN

# 跑五轮；每轮训练40.96万次决策，再自然投币评估最多三次
python -m experiments.rl.autotrain --dataset DATASET/manifest.json --rounds 5 --workers 16 --output RUN

# 继续上一份完整权重及Adam；新输出目录不会覆盖旧记录
python -m experiments.rl.autotrain --dataset DATASET/manifest.json --resume OLD_RUN --hours 2 --output NEW_RUN
```

`--hours` 和 `--rounds` 可同时给，先达到的限制生效。一轮是一次训练批次加自动评估，
不是游戏中的一个小局。默认8个并行环境，`--workers N`支持任意正整数，例如1、6、12、20、24。
默认每轮409600次决策。三个参数分别控制：

- `--workers 16`：并行执行器数量。
- `--rollout-steps 4096`：每次更新策略前收集的全局决策总量，默认4096。
- `--minibatch-size 64`：每次梯度更新的样本数，默认64。

每轮默认100次采样与PPO更新，不随workers改变。`--steps-per-round`和
`--checkpoint-every`向上对齐到rollout-steps整数倍，请求值与实际值均记录在launch.json。
总量不必整除workers：4096步、20个worker时各收集204或205步，额外名额逐次轮换。
每条轨迹独立计算GAE与末端bootstrap，然后合并；不补样本、不丢样本。
mini-batch至少2且必须整除rollout总量，workers不能超过rollout总量。

例如保持每次总采样量4096，只比较mini-batch：

```sh
python -m experiments.rl.autotrain --dataset DATASET/manifest.json --resume BASE_RUN --workers 20 --rollout-steps 4096 --minibatch-size 64 --rounds 5 --output AB_64
python -m experiments.rl.autotrain --dataset DATASET/manifest.json --resume BASE_RUN --workers 20 --rollout-steps 4096 --minibatch-size 256 --rounds 5 --output AB_256
```

两组从同一完整检查点开始，保持seed、workers、训练总量、epochs和对手采样模式相同。
续训保留Adam，但使用本次命令的rollout/minibatch值（省略时使用上述默认值），
覆盖旧ZIP的这两个设置。报告逐轮列出实际参数。旧运行和冻结源码保持旧语义。
workers现在不再改变更新样本总量，但仍影响轨迹长度、组成和随机性，不保证逐位相同的结果。

运行中调整并发：先按Ctrl+C，等待安全保存并退出，再用新的并发数接着训练：

```sh
python -m experiments.rl.autotrain --dataset DATASET/manifest.json --resume OLD_RUN --workers 6 --hours 2 --output NEW_RUN
```

续训保留策略网络、价值网络和Adam状态，重新启动所需数量的MAME环境；训练对局从
训练存档重新抽取。旧输出保持原样，不支持在当前PPO采样批次中热增减执行器。
在上述总采样量限制内，实际吞吐和可用数量受本机CPU、内存限制；更多不一定更快。


时间是软预算，从编排开始计时，包含训练与期间评估，不含前置源码构建。到点不会切断
半次PPO更新：先完成当前更新并保存模型，再完成最后一次自然投币评估。因此实际退出
可能比指定时长晚一些。Ctrl+C请求安全停止：保存当前完整更新，不再新开评估；已经
开始的正式评估会自然结束。重复强制终止不具备同样的收尾保证。

默认跑满预算，即使某轮通关仍继续；`--stop-on-clear`可选择首次通过即停。
模型始终继承上轮最后的完整ZIP和Adam，不按小样本胜率回滚权重。省略`--resume`及
`--init-model`会随机初始化；要接着现有成果训练，请明确指定其一。

输出布局：

- `RUN/report.html`：可离线打开的汇总报告。
- `RUN/summary.json`：结构化统计，适合自行分析。
- `RUN/run/result.json`：原始编排结果及最新可恢复模型路径、SHA。
- `RUN/run/cycle-*/`：逐轮训练、检查点、自动投币和失败记录。
- `RUN/code/`：由公共源码构建的独立执行版本；指定`--code`时复用已验证版本。

报告按每份权重分别列出通关/投币次数、每个对手的小局与整场成绩。训练期间策略
不断变化，训练胜率不能当作固定模型的通关率；不同权重的成功次数也不拼成一个模型
的成绩。运行异常会停止并保留原因；可检查原始记录，选择最后完整检查点续训到新目录。

当前新构建包含[零血量延迟 KO 结算 v10](SETTLEMENT_V10.md)。旧版若遇到
`new round arrived before previous result was resolved`，先保留失败记录并确认具体原因；
若原始结果发布了 `recovery_checkpoint`，`--resume OLD_RUN` 会读取这份完整更新后的权重。
省略 `--code` 可从当前仓库重新构建执行包；不要复用发生错误的旧冻结执行包。
继续使用原来的大批次时，仍须显式填写 `--rollout-steps` 和 `--minibatch-size`。

## 单独看它打

```sh
python -m experiments.rl.watch --campaign RUN --speed normal
python -m experiments.rl.watch --campaign RUN --speed 2x --difficulty 7
python -m experiments.rl.watch --model /path/to/ppo.zip --speed 4x
```

默认一枚自然币、静音、可见窗口、不设为最顶层。可选normal/2x/4x/fast；实际能否达到目标
速度取决于硬件。游玩不读档、不Continue、不在对局中暂停，不进入训练的自动评估统计。
观看前复制固定模型快照，训练后来保存的新权重不会改变正在观看的比赛。
输出单独保存在`.local/rl-watch/`。更多选项见[独立观看](WATCH.md)。

## CPU与4090

默认`--device cpu`。`--device cuda`明确要求可用的CUDA版PyTorch，缺少则报错；
`--device auto`在CUDA可用时使用它，否则使用CPU。CUDA负责Torch计算和PPO更新；
MAME、Lua中的策略推理、游戏采样仍运行在CPU。不要把GPU上的纯更新倍率当作整轮提速。
安装与当前实测见[CUDA说明](CUDA.md)。

远程Linux/WSL用同样的命令。观看需要那台机器的图形桌面；SSH连接本身不会把MAME画面
传回本地。WSLg可在Windows桌面显示Linux MAME；无图形桌面的服务器仍可做自动训练评估。

macOS 和 Linux/WSL 的 `autotrain` 会对子进程设置 `SDL_VIDEODRIVER=dummy`，配合 `-video none`
使并行采样和自动评估都不创建桌面窗口；不需要 DISPLAY。此设置仅作用于本次训练，
不会修改 shell 环境，独立 `watch` 继续使用桌面显示。不要把 dummy 全局 export 到观看终端。

## 自动增加弱项练习

默认 `--opponent-sampling adaptive`。每个对手保存最近128个**已结束小局**的胜负，
平局计为未赢；用20局、50%胜率的先验平滑。每次完整PPO更新后，用
`(1 - 平滑胜率)^2`分配一半开局采样机会，另一半均匀分配；11对手时每个至少4.55%，
单个至多25%。新概率每次只向目标移动25%，减少短期波动。

只改变**下一整场开始时**选择对手的概率；进行中的对局不会被切走。每名对手内部仍
均匀选训练开局并加入原有起步帧变化。对局数据都是当前策略新采集的，PPO小批次、
动作、奖励、模型输入、网络大小与正式自然投币验证流程不变。

完整ZIP同时保存近期胜率窗口和采样概率。`--resume`跨轮数、并发数、CPU/CUDA继续
训练时自动继承；旧模型没有这份状态时从均匀采样开始，随着新结果自动调整。
采样历史只使用训练结果，不读取开发/保留集，也不拿三次自然投币的小样本调概率。

`report.html`与`summary.json`新增各对手实际决策数/占比、平均小局长度、平滑近期胜率
和下一场抽样概率。统计包括本轮未结束小局的已采集决策；概率不是目标决策占比，
本版没有额外按对局时长补偿。每次更新的概率过程保存在训练result.json的iterations中。

需要均匀对照时加 `--opponent-sampling uniform`，它仍统计并保存近期胜率，但均匀选对手。
这只是一种训练机会分配方案，尚未证明能提高固定模型的通关率；比较应使用新的自然投币结果。
