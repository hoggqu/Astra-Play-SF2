# 六个飞行道具标量：独立观察候选（仅离线验收）

独立 368 维 builder 和零列迁移已实现并通过离线检查，尚未进行原生
验收或启动 PPO；现行 344 维包保持不变。目标是检验一项新增
信息能否改善跨对手稳定性；不能把缺少波位置宣称为现有网络的硬上限。
最新完整 20 币对照中，旧观察 weighted 模型的 Ryu 小局为 22 胜 2 负，
但整条路线仅 1/20 通关；其失利集中于 Zangief、Guile。uniform 为 0/20。
这种差异也可能来自采样和跨角色泛化取舍，新增波特征只是一个对照候选。

## 已有证据及范围

[只读观察报告](PROJECTILE_OBSERVATION.md) 覆盖 Ryu/Ken、Guile、Sagat、
Dhalsim 的有限 train 样本，未使用 dev/holdout 选模。

- type 0：Ryu/Ken，Y96。
- type 1：Dhalsim，Y80，有向前移动的完整区间。
- type 3：Guile，Y90。
- type 4：Sagat，Y118/Y78；同一 type 不区分高低。
- Dhalsim 还出现 type 2，位置和高度变化与上述平移对象不同，疑似近身
  持续火焰。在 5 个退出前的原生帧中，玩家关联已经清零，对象却仍然
  `status=0x0101, HP=256`。这反驳“这两个字段足以定义所有活动攻击”的说法。

只实际观察了对象槽 6/7。非零 HP 的数量、生命周期或所有碰撞语义均没有
被完全逆向；对象的中心点也不是 hitbox 的边界。

## 建议冻结的第一版定义

每个原生状态保留原 86 个特征，末尾追加以下六个值，顺序不可变：

| 新列 | 含义 |
|---:|---|
| 86 | Ken 自己有一个符合条件的波：1，否则 0 |
| 87 | 该波 `(x - Ken.x) / 512`，裁剪到 [-1,1] |
| 88 | 该波 `(y - Ken.y) / 256`，裁剪到 [-1,1] |
| 89 | CPU 有一个符合条件的波：1，否则 0 |
| 90 | 该波 `(x - Ken.x) / 512`，裁剪到 [-1,1] |
| 91 | 该波 `(y - Ken.y) / 256`，裁剪到 [-1,1] |

两组都以当前 Ken 的原生位置为坐标基准，不随朝向反转。无波时该组完整
编码为 `(0,0,0)`；标志位可区分“无波”和“波恰好与 Ken 同坐标”。不增加
速度、类型、距离奖励或按对手切换网络。

建议第一版仅接受同时满足下列条件的槽：

1. 槽属于 `0xFF938A + slot*0xC0` 的八槽池。
2. status 恰为 `0x0101`，HP 恰为 256。
3. type 属于 `{0,1,3,4}`。**暂不纳入 type 2**，避免把尚未确认的近身攻击
   对象当成飞行波。若要包含，应先另行定义语义，不默默扩展本接口。
4. 对象 `+0x26` 的 u16 与 Ken/CPU 原生地址低字匹配；所属玩家 `+0x1D4`
   的 u16 必须又指向该对象槽低字。没有把整个 u32 强行转换成指针。

只读探针保留原始对象诊断；共享编码器返回拒绝总数和选中槽，
目前没有逐原因计数或把这些计数加入模型。拒绝对象不混入有效对象，
也不因此修改 RAM 或按对手改招。这些是**观察接口的操作定义**，不是
“可造成伤害”的完整判定。

双向关联通常只允许每方一个对象。仍明确多候选的确定性规则：按
`(abs(object.x-Ken.x), abs(object.y-Ken.y), slot)` 升序取一个；不得依赖 Lua table 遍历顺序。实际双向单指针过滤下
每个 owner 最多一个不同槽，尚无多槽同时通过的实机证据。第一项不是碰撞到达时间预测，没有
速度信息时不宣称它选的是最危险的一颗。

## 最小 builder 与迁移

从固定的 16 动作 round-chain 包生成新的独立包，不覆盖原包。动作、
Core 结算、12 原生帧节拍和奖励保持原字节。新观察模块只有一个共享 Lua
实现，训练快照与连续部署都调用它；Python 特征编码与 Lua 用同组金样本
对照，避免训练和部署各自解释 owner/无波/排序。

观察历史仍为四帧，最旧到最新，**每帧 92 维，共 368 维**。迁移分别处理
策略首层和价值首层（通常都是 `64 × 344` → `64 × 368`）：

```
对 h = 0..3, j = 0..85：
    new_weight[:, 92*h+j] = old_weight[:, 86*h+j]
对每个 h 的新列 92*h+86 .. 92*h+91：
    new_weight[:, new_column] = 0
```

不能把旧 344 列直接拷到新矩阵最前面，那会错位后三个历史块。原偏置、
后续层、动作头和值头全部复制。六列零初始化使任何新波值都不影响迁移
瞬间的函数；数学上旧 logits/value 不变。浮点 GEMM 的形状变更可能造成
极小舍入差异，验证应记录最大误差及近乎平局的 action margin，不能伪称
任意输入都逐位相同。

若继承 Adam，首层 `exp_avg`、`exp_avg_sq`（以及启用时的 AMSGrad 最大
二阶矩）按同一列映射；新列矩为零，step 与其他层状态不变。应按参数名
对照映射，不能假定序列化整数参数 ID 永远不变。其余 PPO 超参、更新计数
和原模型 SHA 都记录。对照组继续使用同一父模型/优化器；若选择重置
优化器，则两组都重置并声明，不能只让候选组获得额外变化。

新包需明确 `observation_interface` 身份与特征顺序/尺度/过滤协议，不能
复用 344 维模型的身份。builder 同步更新 Gym 空间、SB3 首层、导出维度、
Lua history、批采样载荷、连续策略标识和接口审计；动作身份仍是现有
16 动作接口。新增共享观察模块也必须进入 source hashes 与 staging 审计。
旧模型只通过明确迁移命令进入新包，不能静默接受 shape 不符的权重。

## 启动训练前的有限验证

- 离线列映射测试：四个历史块分别放不同值，覆盖 pi/vf 与 Adam 矩；
  实际旧 train 观察和随机合法新增值下，检查旧/新 logits/value、贪心动作。
- 原始波 trace 的金样本：owner 错配、HP=-2/-1/0、status0100、无波、
  两侧/不同高度、多候选稳定排序、type2 排除及历史重置。
- 新旧模型冻结零新增权重，在同 train checkpoint/强制动作下核对 Lua
  与 Python 的完整 368 观察，并确认 batch 与连续执行逐原生帧一致。
- 最后小额真实 PPO 更新和一次独立自然币接口审计，再考虑同预算训练。

对照须看全部对手和完整 20 自然币通关率，不能只比较 Ryu 或有波角色。
即使观察更丰富，也可能损害无波角色、改变跨角色权重分配或增加过拟合。
低虎波命中语义、type2 范围、未见过的槽/初始化状态、遮挡与碰撞边界仍是
未解风险。当前不据此承诺通关率提升。


## 已实现的离线入口

仅从已有冻结的 **344 维、16 动作 round-chain 包**派生，创建全新目录：

```sh
python -m experiments.rl.projectile_builder --source PATH/TO/CHAIN_PACKAGE --output NEW_CODE_DIR
python -m experiments.rl.projectile_migrate --model OLD_ACTIONS16.zip --output NEW_MODEL_DIR
python -m unittest experiments.rl.test_projectiles6 -v
```

包名为 `astra_sf2_rl_projectiles6`；模型身份为
`sf2_projectiles6_owner_reciprocal_v1`。独立 schema 为
`astra.rl-projectiles6-build.v1` / `astra.rl-policy.projectiles6.v1` /
`astra.rl-continuous.projectiles6.v1`。原 action interface 不变。
Gym 92×4、Lua features、export、batch live policy、native 部署与逐 match
身份检查同步扩展。两个 runtime 调用同一个 `projectile_features.lua`；
模块也进入 staged hash 和 campaign source hash。

builder 记录父 manifest、父文件 SHA 与派生文件 SHA；动作、原生 Core 和
结算文件必须保持父包字节。训练 runtime 仅插入共享只读 snapshot 字段，
奖励函数、采样、输入节拍不改。生产依赖仍由已安装 `astra_play_sf2` 提供，
进入每次 runtime/campaign 的现有来源审计；本 builder 不安装 MAME 或 ROM。

`projectile_acceptance` 可使用已有训练诊断和只读 probe JSON，无须启动 MAME：

```sh
python -m experiments.rl.projectile_acceptance \
  --model OLD_ACTIONS16.zip --package NEW_CODE_DIR/astra_sf2_rl_projectiles6 \
  --output NEW_CHECK_DIR --diagnostics TRAIN_CASE_1.json TRAIN_CASE_2.json \
  --probes TRAIN_PROBE_1.json TRAIN_PROBE_2.json
```

输入必须声明 `training_only=true`，诊断历史按真实 reset 标志重建，文件 SHA
写入证据。此工具不查找或打开 holdout。它保存迁移前后 logits/value 最大误差、
逐样本 argmax 一致性、原 margin、Adam 来源、模型 SHA、原生 staged Lua 语法与
身份检查。任何 argmax 改变（包括接近平局）都拒绝离线迁移验收。

在实际 a06 父模型与 12 个既有 train-only 诊断上：

- 4,446 个真实历史加 515 个随机/边界样本；batch 1/32/256 和保存重载全部
  argmax 一致。Torch logits 最大差 `2.3842e-6`，value 最大差 `4.7684e-7`。
- 195 个真实/随机/边界样本，Lua 旧新 actor/value 在这组样本上精确相等。
  Torch/Lua logits 最大差 `3.2470e-6`，value `6.1248e-7`，logprob `5.0024e-6`；
  使用原 PPO Lua/Torch gate 的 `2e-5` 绝对限，未放宽运行时检查。
- 六个既有 probe 共 21,139 原生帧与独立字段 oracle 一致；2,774 个有效
  owner-frame 包括 type 0/1/3/4，排除 type 2。这里没有新增游戏运行或胜率样本。
- 首层权重与 Adam 的一阶/二阶矩（含可选 AMSGrad）按历史列映射；新列零，
  后层和 step 保留。测试包含序列化重载。未通过训练更新来“修补”迁移误差。

这些结果仅证明有限离线数值与接口检查。它们不证明同一 ROM 的所有波类型、
原生读取时序等价、吞吐或胜率改善。启用前仍需独立原生观测/动作一致性验收，
随后另立冻结来源和父模型一致的对照；当前容量试验继续使用 344 维，不混合变量。
