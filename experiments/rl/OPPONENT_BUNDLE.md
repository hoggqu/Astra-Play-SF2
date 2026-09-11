# 固定对手路由的神经网络 bundle（实验候选）

这是新的 `ppo_opponent_bundle` 策略：一个 ZIP 包含 11 个完整 PPO actor/critic 分支，
按当前对手 ID 路由。它不是原来的共享 64×64 模型。全部分支、路由和输入接口在
启动前固定并整体计算 SHA-256；正式游玩不训练、不加载外部权重、不回退到 V4。
每个逻辑分支拥有完整模型文件，来源相同时文件内容可以相同。

本原型只构建、导出和验证冻结分支，不实现多分支 PPO 训练。原有正在运行的训练包、
生产 V4 和共享策略入口均不变。输入仍为四个决策时刻的 344 维历史观测；输出为
同一套 pulsed16 动作，每 12 个原生帧决策一次，固定朝向和最后一帧普通技松键语义不变。
离线验证和原生兼容性检查已通过；通关能力须由独立完整 20 次验证确定。

## 三份独立候选

1. `baseline-all-f26`：11 路均来自 f26，用于确认路由包装和原模型行为等价。
2. `dev-selected`：四个来源模型在相同 dev22 上逐对手择优，依次比较整场胜数、
   小局胜数，再按固定顺序 f26、local-C2、remote85、local-C3 解开平手。
   选中 Honda/Zangief 的 C2 和 Sagat 的 C3，其余用 f26。
   该算法不使用正式统计（批量构建工具同时读取另一候选所需的旧正式审计）。
   回顾性拼选结果为 22/22，属于选择数据上的描述，不是新策略胜率或通关证明。
3. `prior-formal-informed`：把四个旧模型各自完整 20 次中的所有小局 W/L/D
   作为开发输入，逐对手按 Wilson 下界公式（z=1.96）排序；平局计非胜、零样本最后、
   完全平手按上述固定顺序。该数值只是保守排序启发式：小局和路线并不独立，
   不提供置信保证。旧正式批次不能再被称为此候选的独立测试。

| 对手 | prior-formal-informed 来源 |
|---|---|
| Ryu、Honda、Guile、Sagat、Bison | remote85 |
| Blanka、Zangief、Dhalsim | local-C2 |
| Chun-Li、Vega | f26 |
| Balrog | local-C3 |

来源统计见 [旧模型完整成绩](PULSED_RELIABILITY.md)。私有选择产物同时保存
输入审计文件 SHA、各模型 SHA、全部候选分项统计、选择规则和完整 bundle SHA。
不同模型/候选的局数不得合并，不得用未来正式结果临时修改路由。
目标仍是某个固定 bundle 在它自己的一批完整 20 次自然投币中至少 10 次通关。

## 构建与验证入口

```sh
python -m experiments.rl.opponent_bundle_builder \
  --source PULSED_VERIFY_CODE/astra_sf2_rl_round_chain --output NEW_BUNDLE_CODE
python -m experiments.rl.opponent_bundle_select \
  --config selection-inputs.json --output NEW_BUNDLES
python NEW_BUNDLE_CODE/launch.py \
  --model NEW_BUNDLES/prior-formal-informed.zip --output FRESH_RUN \
  --difficulty 3 --attempts 20 --all-attempts --speed fast
```

最后一条须在原生门禁通过后使用；需要项目已配置的 MAME/ROM，默认静音无窗口。
`selection-inputs.json` 中 `candidates` 必须依次包含 `f26`、`local-c2`、`remote85`、
`local-c3`，每项声明 `model`、`model_sha256`、`dev`、`formal`、`formal_selector`。
前两个来源的 selector 为 `1`/`2`（既有 C1/C2 合并审计），后两个为 `dual`；
顶层 `public_formal_summary` 指向旧模型成绩文档。输入文件/原始日志及权重留在 `.local/`。
工具拒绝复用输出、混用不同 dev 状态、缺失/无效 dev 或未完成的旧正式批次。

代码 builder 完整复制并验证父快照及其生产依赖，只在新包中更换导出器、
网络路由包装与显式策略身份。原生计时、生命周期、有效胜负和动作接口检查保留。
新包格式为 `astra.rl-opponent-bundle.v1`，Lua payload 为
`astra.rl-opponent-bundle-policy.v1`，结果为 `astra.rl-continuous.opponent-bundle.v1`。
包内各 PPO ZIP 保留 critic 和优化器，但游玩只计算当前路由 actor。

`bundle_audit` 在原生/动作接口审计之外验证全部分支启动身份、逐场路由、
每次决策记录中的来源模型 SHA 和动作编号。未知/歧义对手、任何权重或源码变更、
缺少路由证据均为 invalid，不能因为旧共享模型审计不认识新 schema 而绕过检查。
路由 SHA 同时绑定对手到分支的映射与分支来源模型 SHA；完整 ZIP SHA 还绑定
选择依据与完整权重。运行前后 hash 守卫覆盖父源、生产依赖、原始 NN 和新路由 NN。

离线测试覆盖全部 11 路 Torch/Lua logits 和 argmax 一致、全 f26 等价、
真实 Lua 序列化、ZIP/actor/路由篡改、错误输入接口、逐决策身份、
审计失败使最终结果 invalid，以及 dev/Wilson 的平手与零样本规则。

## 原生兼容性结果

2026-09-12，全 f26 包装与 prior-formal-informed 各完成 22 场开发集比赛。
全部 11 个分支、共 245,395 个原生帧的状态、事件、输入文本、动作及成熟结算，
均与各分支原始模型的已保存轨迹严格一致。新记录的实际 IN1/IN2 玩家按键位
也逐帧通过检查；旧轨迹没有这些端口记录，因此没有声称新旧实际端口相互比较。
两组运行文件哈希终检通过，自有 MAME 均退出。

两组都是 19 胜 3 败，只用于验证包装没有改变模型行为，不能计入正式通关率。
prior-formal-informed 已冻结后开始独立 20 次自然投币验证；结果未完成前不宣称达标。
