# Ken reinforcement-learning pilot / 强化学习实验

实验分支：`codex/rl-experiment`。正式 V4 策略、`astra-sf2 verify` 和主分支不变。
这是实际 MAME 环境中的 PPO 神经网络训练，不是修改 Lua 规则或模拟胜率。
首轮实机结果见[实验记录](RESULTS.md)：闭环跑通，模型尚未提高评估胜率。

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

## 本轮实验设计

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
中暂停；本实验不会生成发行版的正式认证。下一阶段应收集多个自然开局、按
开局分离训练/评估，再研究连续推理和全路线验证。

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
目前是单实例原型，尚无多进程采样；先测吞吐再决定并行和传输优化。

离线检查（无需模拟器或 ROM）：

```sh
python -m pip install lupa==2.8
python -m unittest experiments.rl.test_rl experiments.rl.test_runtime -v
```

接口设计参考 [Gymnasium 自定义环境](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/)、
[Stable-Baselines3 自定义环境](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/custom_env.html)
和 [MAME Lua API](https://docs.mamedev.org/luascript/ref-core.html)。
