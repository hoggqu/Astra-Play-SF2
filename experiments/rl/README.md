# Ken reinforcement-learning pilot / 强化学习实验

实验分支：`codex/rl-experiment`。正式 V4 策略、`astra-sf2 verify` 和主分支不变。
这是实际 MAME 环境中的 PPO 神经网络训练，不是修改 Lua 规则或模拟胜率。
首轮 4096 次决策的实机结果见[试跑记录](RESULTS.md)；扩大采样与两个种子的
102400 次决策实验见[并行训练记录](SCALED_RESULTS.md)。评估小局不等于通关率。

## 安装与运行

从本分支的源码根目录运行，先按项目安装文档配置 Python 3.10+、MAME 0.288
和 `sf2` World 910522 ROM。实验复用 `astra-sf2 configure` 保存的配置。

```sh
# macOS / Linux
python3 -m venv .local/rl-venv
source .local/rl-venv/bin/activate
python -m pip install -e . -r experiments/rl/requirements.txt
python -m experiments.rl.run --output .local/rl-runs/pilot-001 --steps 4096
```

```powershell
# Windows PowerShell
py -3 -m venv .local/rl-venv
& .\.local\rl-venv\Scripts\python.exe -m pip install -e . -r experiments/rl/requirements.txt
& .\.local\rl-venv\Scripts\python.exe -m experiments.rl.run --output .local/rl-runs/pilot-001 --steps 4096
```

输出目录必须不存在。默认最高难度 7、随机种子 42；可指定 `--difficulty 3..7`
及 `--seed`。`--steps` 是模型决策数，必须是 256 的正整数倍，不是游戏帧数。
默认静音、无窗口、CPU 训练、模拟器不限速；`--show-window` 可显示游戏，仍静音。
模型初始化、PPO 更新和推理均在本机完成，无需 AI Agent 或在线模型。
训练依赖的轮子是否支持本机 Python/OS，由 pip 实际安装结果确定。

程序拥有独立输出目录和 MAME 子进程，只结束自己创建的实例；不要给这个实例
附加第二个发送者。正式验证监听器只在实验实例完成自然开局后解除，随后进入
训练专用 RPC；不会修改发行版文件或其他正在运行的模拟器。

## 扩大训练：多个开局、并行采样、定期评估

### 普通难度全对手自动训练与通关实验

现在支持一套权重混合训练全部 11 名对手，以及 Lua 中每 12 个原生帧执行一次神经网络
推理的[连续游玩](CONTINUOUS.md)。新的训练优先使用
[round-chain 派生包](ROUND_CHAIN_DESIGN.md)，它已修正暂停恢复时的输入提前锁存。
[原生时间说明](NATIVE_TIMING.md) 区分这次修正和早期有限样本的 parity；
下面的 R1-only `native_campaign` 命令及旧暂停 RPC 保留作历史对照，
它们没有获得新的 chain 时序认证。战斗不调用 AI Agent，也不在中途暂停向 Python
请求动作。选择角色、奖励关和地图过场仍复用固定脚本；实际交战没有 V4 回退。

原生结算采用[证据协议](SETTLEMENT.md)，包括自然进入下一小局才确认的特殊
时间平局；训练奖励使用上一局的成熟状态，防止补血污染奖励。
[16 动作候选](ACTIONS16.md)保留原动作并增加中升龙，由同一网络选择。
它使用独立生成包、明确的模型迁移和接口审计，先通过原生一致性及短 PPO
更新检查，再运行自动训练与自然投币验证。候选成绩与原 15 动作模型分开统计，
已取得[首次真正的普通难度通关](FIRST_NORMAL_CLEAR.md)：冻结 PPO 模型在一批
20 次自然投币中，第 20 次连胜全部 11 名对手，无 continue、读档或对局内暂停。
这达成了第一步目标，但尚不代表稳定通关。

下一阶段目标是[固定模型完整 20 次至少通关 10 次](RELIABILITY.md)。
自动流程先完整验证候选，未达标才继续训练；全部失败保留，不拼接不同模型的成功次数。

后续对照包括只改变训练对手概率的[固定加权采样](WEIGHTED_SAMPLING.md)，
以及固定权重的 [16 动作随机策略验证](STOCHASTIC16.md)。
[飞行道具观测调查](PROJECTILE_OBSERVATION.md)记录了只读实机证据和字段限制，
尚未加入当前模型输入。

```sh
python -m experiments.rl.collect --difficulty 3 --opponents all --samples 4 --output .local/rl-data/normal-all-001 --max-seconds 1200
python -m experiments.rl.native_campaign --dataset .local/rl-data/normal-all-001/manifest.json --output .local/rl-runs/normal-native-campaign-001 --init-model .local/rl-runs/train-001/best-dev.zip --workers 8 --cycles 10 --steps-per-cycle 102400 --block 64 --verification-attempts 1 --seed 101
```

`--init-model` 可省略以随机初始化；示例模型路径需要替换为实际已有模型。
每个对手独立按采集顺序拆分 2 train / 1 dev / 1 holdout。训练时先均匀选对手，
再选该对手的训练存档。每批新增 102400 次决策，继承批末模型和优化器，
不因单次小样本开发评估回滚；开发迭代不打开 holdout。固定执行源码在每个
阶段前后核验，新建无关实验文件不影响批次。

每批结束固定模型、自然投币验证；`--verification-attempts` 支持 1～3，默认 3。
上例每批只验证一次，适合通关率仍低时减少重复验证耗时。成功完成 11 场对手比赛后自动停止，
否则自动继续下一批，以上命令最多十批。所有败局与中断保留；控制异常立即停止。
通关结果为独立的 `rl_gameplay_clear`，不混入冻结 V4 的认证或历史胜率。
初步接口检查和计划见 [Normal 实验设计](PLAN_NORMAL.md)。仅一个开局的局部胜率
不能替代完整路线的实测。初次 R1-only 数据仍不覆盖所有 R2/R3 条件。

两个 campaign 可使用不同 seed 和独立输出目录并行运行；必须把两个进程组的
资源合计计入主机预算。每个 campaign 都不需要 Agent 按小局下指令。

批量训练约每 20480 次已完成 PPO 更新的决策保存一份模型检查点，路径、步数、
哈希记入训练结果的 `checkpoints` / `last_checkpoint`。若阶段异常，仍记录为
无效；可用该检查点在**新的输出目录**明确指定 `--init-model` 续训。它恢复网络和
优化器，不恢复模拟器局面或 RNG，也不代表失败阶段已经完成。

只读统计可重复传入多个运行目录，输出每个模型的逐对手训练小局成绩、连续游玩
小局成绩和整路线失败对手。运行中的尝试记为待定；Lua 与 Python 的同一小局不重复计数。

```sh
python -m experiments.rl.report_normal --campaign .local/rl-runs/normal-native-campaign-001 --output .local/rl-reports/normal-001
```

输出目录须为新目录，并位于输入运行目录之外。报告不启动模拟器、不控制训练，
也不代替原始连续游玩审计。

独立批量训练可用 `--training PATH` 纳入报告，独立原生连续验证可用
`--continuous PATH`；三个输入选项均可重复组合，已覆盖的子阶段自动去重。

默认连续验证选概率最大的动作。另有两个独立的验证对照，均固定网络权重：

```sh
python -m experiments.rl.stochastic_continuous --model MODEL.zip --output .local/rl-runs/sample-001 --difficulty 3 --attempts 3 --policy-seed 42
python -m experiments.rl.fast_fire_continuous --model MODEL.zip --output .local/rl-runs/fire-001 --difficulty 3 --attempts 3 --variant fast
```

前者按网络 softmax 概率采样，使用独立策略 RNG，并复算完整动作与随机数序列；
后者保持 argmax，只把波动拳改为 2/2/2 帧输入加 6 帧空输入。`--variant original`
可运行原宏对照。它们分别记录选择方式、种子或动作接口，不能因 ZIP 相同就
当作同一种策略。首批各三次验证均未通关，尚未据此替换默认训练方式。
结果识别的修复及原生证据边界见 [SETTLEMENT.md](SETTLEMENT.md)。

旧 `campaign` 保留为历史对照：逐决策 RPC、每对手一个 lead 的 dev 选模并
继承 best-dev。它可能回滚优化进展，且暂停恢复的帧回调不严格等于推进的原生帧；
不要把其计数与新的 native-time 训练混合。时序检查和历史失败见上述说明。

### 首轮 Blanka 多开局实验

先用冻结 V4 走自然投币流程，采集六个 Blanka R1，按采集顺序固定分为
4 个训练、1 个阶段评估（dev）、1 个最终检查（holdout）。这是训练采集，允许
保存；不读档、不重置来挑选路线。遇到败局会记录，并在自然结束后重新投币。
不同存档哈希仍不保证统计独立，dev/holdout 只有一个开局的限制要保留。

```sh
python -m experiments.rl.collect --output .local/rl-data/blanka-001 --samples 6 --max-seconds 900
python -m experiments.rl.train --dataset .local/rl-data/blanka-001/manifest.json --output .local/rl-runs/train-001 --workers 4 --steps 102400 --eval-every 10240
```

每个采样进程拥有自己的 MAME 和存档副本，从训练池中抽样；四个历史观察和
每次 12 帧的动作设置与首版相同。PPO 每个 worker 收集 256 次决策后更新，
总 rollout 为 `256 × workers`。使用 CPU 推理/训练，当前不需要 CUDA。
默认静音无窗口，最多支持八个采样 worker。

每 10,240 次总决策保存模型，并在固定 dev 开局及 2/6/10 帧等待条件下评估。
按 dev 胜率优先、平均回报次优选择模型；结束训练、冻结选择并重载模型后，
才打开 holdout 做最终比较。dev 是用于选模的开发评估，不是最终泛化证明。
每个阶段还保留 V4 基线，全部失败、未完成片段和模型快照分别保存。

`train.py` 的总步数和评估间隔必须是 `256 × workers` 的正整数倍，且总步数
可被评估间隔整除。`--seed` 用于重复实验，不能据一次训练判断稳定性。
不同系统/架构生成的 MAME 存档未承诺互通，应在运行训练的主机上采集。
加载前验证文件哈希及难度，加载后检查原生角色、满血开局和内部难度。

只测采样吞吐，不更新模型或查看 dev/holdout：

```sh
python -m experiments.rl.train --dataset .local/rl-data/blanka-001/manifest.json --output .local/rl-runs/bench-4 --workers 4 --benchmark --steps 4096
python -m experiments.rl.train --dataset .local/rl-data/blanka-001/manifest.json --output .local/rl-runs/bench-8 --workers 8 --benchmark --steps 4096
```

两个 benchmark 顺序运行，比较总决策/秒。不要把不同等待条件计作独立样本。
有界 worker 关闭和异常日志用于保留失败；Linux 服务的整个进程组还应设置
总资源上限和退出清理，避免父进程故障留下模拟器。

### 共享 Linux 主机的资源限制

本次远端为 13900KS / 32 个逻辑 CPU、WSL 可见约 47 GiB RAM。经用户调整，
初始单任务测试上限为 12 个逻辑 CPU / 16 GiB。用户进一步允许增加资源后，
采用两组各 8 个 worker 的独立种子实验，每组设 `CPUQuota=1000%`、
`MemoryMax=12G`、`MemorySwapMax=0`、`Nice=10`、`TasksMax=512`，
合计上限为 20 个逻辑 CPU / 24 GiB。这是各组全部 worker 共用的上限。
使用 CPU-only PyTorch，4090 不参与本轮训练。保持桌面应用有资源余量。

在有 systemd 的 Linux 上，可将完整 Python 命令放入单个 transient service；
预先设置 `ASTRA_SF2_CONFIG` 为本机实际配置路径，并使用未用过的 unit/output：

```sh
systemd-run --unit=astra-rl-train-001 \
  --property=CPUQuota=1000% --property=MemoryMax=12G \
  --property=MemorySwapMax=0 --property=Nice=10 --property=TasksMax=512 \
  --property=RuntimeMaxSec=3600 --working-directory="$PWD" \
  --setenv=ASTRA_SF2_CONFIG="$ASTRA_SF2_CONFIG" \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  "$PWD/.local/rl-venv/bin/python" -m experiments.rl.train \
  --dataset .local/rl-data/blanka-001/manifest.json \
  --output .local/rl-runs/train-001 --workers 8 --steps 102400 --eval-every 10240
journalctl -u astra-rl-train-001 -f
```

这是可选的 Linux 运维方式，需要运行者有创建服务的权限。Windows/macOS 的
直接 Python 入口保持可用，但这段 Linux 资源限制命令不适用于它们。
采集和 benchmark 顺序执行；两个训练服务可并行，但需合并计算资源上限，
不能给每个服务都分配整台机器的预算。

## 首版 pilot 的实验设计

- **环境：** 自然投币选 Ken，遇到第一位真实对手后保存 R1。没有修改游戏 RAM
  或强制对手。当前只支持这个开局的一回合训练，不是任意指定对手的采集工具。
- **状态：** 双方 HP、位置、离地状态、动作类别、对手身份、计时，合计 86 个
  数值，堆叠最近四次观察。模型不接收 CPU AI 排名、内部决策参数或未来状态。
- **动作：** 15 种动作，包括前进、后退、蹲防、跳跃、普通攻击及波动拳/升龙拳
  输入序列；每次固定执行 12 帧。必杀技只是普通按键序列，不保证每次发动成功。
- **模型：** Stable-Baselines3 PPO，策略和值函数各两层 64 单元 MLP。每 256
  次决策更新，batch 64、4 epochs、学习率 0.0003、gamma 0.99、熵系数 0.01。
  首轮从随机初始化训练，尚未加入模仿学习。
- **奖励：** 每步 `0.25 × (造成伤害 − 承受伤害) / 144`，成熟胜局 +1、负局 −1、
  平局 0。HP 限制到 0..144 后计算伤害，避免负血值增加奖励。
- **结算：** 每个 native frame 运行冻结的 Core，等待胜局标记、姿态及至少
  360 帧结算成熟。结算期间连续执行，中间不给模型发空决策；不以 HP=0 猜胜负。
  原脚本基线保留其 timeout guard；模型不额外继承该防守规则。
- **对照：** 同一存档，训练用额外等待 0/4/8/12 帧，评估用 2/6/10 帧。
  顺序固定为 V4 基线三回合、未训练模型三回合、指定步数 PPO 训练、保存并重载
  模型后评估三回合。评估使用确定性动作，不再更新参数。

这三种评估等待值是预先保留的局部条件，**不是三个独立开局，也不是通关率**。
相同存档改变等待仍可能重复轨迹；不要把局部胜率直接推广到新路线或其他难度。
SB3 seed 控制模型和等待抽样，不会重新设置游戏内部随机种子。
模型一次训练预算结束时可能正在回合中，该片段单列，不假装输赢。

训练允许读档、暂停与同步步进。正式游戏仍要求不读档、不 continue、不在对局
中暂停；本实验不会生成发行版的正式认证。并行版本已经按自然开局分离
训练/评估，但样本规模仍小；还需扩大开局覆盖，再研究连续推理和全路线验证。

## 输出与耗时

- `manifest.json`：环境、真实难度、对手、存档哈希、运行时和实验源码哈希。
- `episodes.jsonl`：全部完整回合，区分 baseline/untrained/train/trained，保留
  胜负平、回报、游戏帧数、时间和原生 stop/settled/score 结算信息。
- `partial-episodes.jsonl`：预算或关闭打断的片段，未结算，不计输赢。
- `ppo-ken.zip`：训练后模型；`result.json`：前后对照、参数是否更新、模型哈希、
  训练步数和耗时。`startup-error.json` 记录初始化失败。
- `mame.log`、开局截图、存档与实验诊断只存本地，不提交 Git。

RPC 使用独立编号的终态文件，每个文件发布一次；Python 完整读完再移除。
超时使实验无效并结束自己创建的 MAME，不重放不确定的按键或恢复请求。
`rpc_seconds` 包含游戏执行和 IPC 等待，不能把它全部当作纯通信开销；
`train_seconds` 包含采样、恢复及 PPO 更新，`wall_seconds` 还包括启动和对照。
首版 `run.py` 是单实例原型；新 `train.py` 支持单机多进程采样。两台机器联合
更新同一模型的分布式训练尚未实现，可以分别承担采集、实验和评估。

离线检查（无需模拟器或 ROM）：

```sh
python -m pip install lupa==2.8
python -m unittest experiments.rl.test_rl experiments.rl.test_runtime experiments.rl.test_collect experiments.rl.test_dataset experiments.rl.test_train -v
```

接口设计参考 [Gymnasium 自定义环境](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/)、
[Stable-Baselines3 自定义环境](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/custom_env.html)
和 [MAME Lua API](https://docs.mamedev.org/luascript/ref-core.html)。
