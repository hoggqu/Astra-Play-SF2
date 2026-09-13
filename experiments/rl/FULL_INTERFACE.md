# Full85 与可见执行反馈实验

这是独立接口版本，保留已经冻结的 10/20 通关版本。它提供 85 个动作、双方可见眩晕状态，以及此前请求动作的实际执行反馈；不会把旧模型的成绩继承到新接口。

- 动作身份：`ken_actions85_full_v1`。前 16 项与 pulsed16 完全相同，新增普通方向/按钮组合及不同强度必杀技，见 [完整动作](FULL_ACTIONS.md)。每次决策仍执行 12 个原生帧，无动作合法性遮罩、强制等待或禁止重复动作规则。
- 观测身份：`sf2_state86_visible_feedback114_history4_v1`。每帧原 86 项状态加 114 项反馈，四帧历史合计 **800 项**，详见 [可见反馈](VISIBLE_FEEDBACK.md)。这些是四次决策时刻的观测，不是最近四个相邻原生帧。
- 每个推进的原生帧先更新反馈，再计算当前决策观测。选出动作以后才登记新请求，登记不修改已保存观测；因此当前请求不会泄漏到选择它的输入。新对局、读训练存档和自然新小局均清空反馈与历史。
- 反馈明确区分请求、观察到的动作开始、其他动作和未知；没有观察到开始不等于证明失败。双方可见眩晕来自已验证的动画身份，不读取隐藏累积眩晕值。正常技可以识别的强度被记录；视觉无法区分的必杀技强度保留未知。
- PPO 仍使用同一共享标准 MLP。奖励、折扣、GAE、采样方法和原生成熟判胜逻辑保留。战斗输入全部来自 PPO，没有 V4 战斗 fallback。

## 构建与运行

需要安装项目与 RL 可选依赖。公共 bootstrap 直接从 checkout 的源代码构建全部父级，不需要私有目录、ROM、模型或历史数据。下面的相对路径可直接用于新 checkout；各输出目录必须不存在，训练命令要求已备好 `full85-openings-001/manifest.json`：

```sh
python -m experiments.rl.full_interface_bootstrap --output .local/rl-code/full85-001
python .local/rl-code/full85-001/launch.py initialize \
  --seed 851 --output .local/rl-runs/full85-initial-001
python .local/rl-code/full85-001/launch.py batch_train \
  --dataset .local/rl-data/full85-openings-001/manifest.json \
  --init-model .local/rl-runs/full85-initial-001/ppo-initial.zip \
  --workers 4 --steps 2048 --seed 851 --output .local/rl-runs/full85-short-001
python .local/rl-code/full85-001/launch.py export \
  --model .local/rl-runs/full85-short-001/ppo-batch.zip --output .local/rl-runs/full85-export-001
python .local/rl-code/full85-001/launch.py native_continuous \
  --model .local/rl-runs/full85-short-001/ppo-batch.zip --difficulty 3 --attempts 1 \
  --speed fast --output .local/rl-runs/full85-validation-001
```

在源 checkout 运行 builder 时，若未安装主项目，先设置 `PYTHONPATH=src`。已有冻结的 pulsed 或 specialist-pulsed 父包也可用 `python -m experiments.rl.full_interface_builder --source "$PARENT_SOURCE" --output "$NEW_CODE"` 构建独立子包。生成的 `launch.py` 包含标准主入口保护，适用于多进程 spawn；也可用 `PYTHONPATH="$NEW_CODE/src:$NEW_CODE" python -m astra_sf2_rl_full85.batch_train ...`。MAME 配置沿用项目 CLI，可通过 `ASTRA_SF2_CONFIG` 明确指定。

初始化不会运行 MAME 或训练；会建立新的 800→64→64→85 策略头与独立价值网络、空 Adam 状态。旧 344/16 模型不能直接加载，修改名称也无法通过维度和双接口检查。这里没有迁移入口。短训练仅验证实现可运行，不代表已获得能通关的策略。

训练 loader 接受完成的 `astra.rl-openings.v1/v2` 集合，每个声明对手必须恰有 8 个唯一训练开局，以及独立的 dev/holdout；可以声明全部 11 名对手或其子集。它保留所有 split，核对 ID、文件路径和 SHA；只有 train 集合传入环境，不执行模型筛选。用已有 specialist 父包构建时，也保留该父包的单对手 manifest 支持；公共 bootstrap 使用上述通用集合。训练中的保存、加载和暂停只属于训练模式。

已有兼容集合可直接复用。需要自建数据时，原有 collector 的 `--samples 10` 按每名对手的收集顺序预分为 8 train、1 dev、1 holdout，可在配置 MAME/ROM 后单独运行：

```sh
python -m experiments.rl.collect --difficulty 3 --opponents all --samples 10 \
  --max-attempts 20 --max-matches 220 --max-seconds 3600 \
  --output .local/rl-data/full85-openings-001
```

这是会实际启动 MAME、使用既有脚本策略采集训练开局的独立步骤。达到预算不保证取得全部状态；只有 manifest 为 `complete` 且 loader 校验通过才可训练。不要把 `budget_exhausted` 改名为完成，也不要把 dev/holdout 补进训练集。由脚本策略得到的开局分布与学习策略自身到达的分布可能不同。

连续验证使用一个固定新模型、自然投币和原生逐帧控制，不训练、不读写存档、不 continue、不在对局中暂停。除原有生命周期、成熟比分、完整 11 对手、时钟和来源审核外，每条原生帧及决策记录还保存可见反馈。动作、观测、模型、源包和两份观察器运行文件均绑定 SHA。新接口完整批次应单独统计，不与 pulsed16 或其他模型拼接。

## 实现检查

离线测试覆盖 Python/Lua 的 200 项单帧编码一致、800 项历史模型形状、85 个输出 logits、反馈时序及重置、严格旧模型拒绝、源与映射篡改拒绝、staging Lua 编译，以及实际 SB3 PPO 更新后保存/导出。它们不替代原生端口与计时门禁，也不证明胜率。原生门禁与短环境训练结果应由各自独立结果记录报告；不把离线通过标成原生验证。

首个独立原生集成门禁已覆盖全部 85 个动作：batch 初始加载、显式 reset 和不中断的 native reference 三种路径，各比较 1,332 个原生帧，逐帧状态、反馈、800 项决策观测及实际输入端口一致。随后完成单 worker、256 次决策的原生 PPO smoke，包含 2 个自然小局和 2 场对局；参数与 Adam 状态实际更新，保存后可导出。Lua 与 Torch 的旧策略 log-prob 最大差为 `6.07e-7`，执行约 21.9 秒。这些是接口实现检查，未做长训练，不能据此报告游戏通关率。该门禁使用 specialist-pulsed 父源；公共构建路径还须保持关键运行文件与其逐字节一致，不能仅凭相同接口名称继承证据。

该短训模型又完成一次实际自然投币验证：0:2 败给 Blanka，作为有效失败保留。全场 4,011 原生帧均记录反馈、没有局中暂停，原生和接口审核通过。独立检查从决策记录重建 800 项历史观测，并用 Torch 重新推理，253 次决策全部匹配 Lua。它证明新模型可以贯通训练、导出和正式连续入口；不证明新策略已学会通关。

最终公共构建包也完成了一次从零开始的原生短训：注册全部 11 名对手、88 个 train 开局，单 worker 完成 256 次决策及一次 PPO 更新，保存完整模型和 Adam 状态，约 21.7 秒。未把 dev/holdout 传入训练环境；这段短训只采到了训练集的一部分，不代表逐一验证了 11 个对手的胜率。最终离线回归共 382 项通过，包含无私有目录的构建、移动产物后的身份验证及启动检查。
