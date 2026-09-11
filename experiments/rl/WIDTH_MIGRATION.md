# 共享 PPO 网络的保函数扩宽候选

`width_migrate` 是离线工具，将同一 16 动作 PPO 的 actor、critic 从
`[64,64]` 扩为 `[128,128]`。仍是普通 Linear/Tanh 网络，输入 344、动作 16、
12 帧宏、观察、奖励和对手采样全部不变。它不启动 MAME 或正式训练。
这是检验容量是否影响持续角色取舍的候选，不是容量硬上限的结论。

## 初始化与优化器

对 actor 和 critic 分别做相同的块嵌入：

```text
W1_new = [ W1_old ]       b1_new = [ b1_old ]
         [ random ]                [ fresh  ]

W2_new = [ W2_old   0 ]   b2_new = [ b2_old ]
         [ random random ]         [ fresh  ]

Wout_new = [ Wout_old  0 ],  bout_new = bout_old
```

新增隐藏单元保留标准 SB3 初始化产生的随机输入权重；通往旧隐藏输出路径及
最终输出头的必要连接置零。旧函数在实数代数下保持不变。新隐藏特征不全为零，
输出连接可先获得梯度，之后新增隐藏参数也能学习，避免把所有新增参数都设零。

按参数名称嵌入旧 Adam 的 `exp_avg`、`exp_avg_sq`，若启用 AMSGrad 则同时
嵌入 `max_exp_avg_sq`；新增坐标的矩为零。保留每参数的 `step`、各 param group
设置、算法更新计数及原动作身份，不重新初始化整个优化器。
**标准 Adam 的 step 属于整个参数张量**，不能同时让其旧坐标维持历史年龄、
新坐标单独从 step0 开始；新增坐标使用零矩和继承的偏差校正年龄，这点明确
记录在迁移元数据中，不冒称为完全新 Adam。后续训练会因新增参数及梯度裁剪
产生不同优化轨迹，不要求训练后的旧坐标与小网络继续一致。

## 调用

使用 RL 可选依赖及 `lupa`。`--code` 指向已经冻结的 actions16 运行包根目录，
必须含 `build.json` 和其声明的 package，工具复用该包的 exporter 和 `nn.lua`
进行离线一致性验证。

```sh
python -m experiments.rl.width_migrate \
  --model .local/models/parent.zip \
  --output .local/rl-runs/width128-candidate \
  --observations .local/rl-runs/train-demonstrations/demonstrations.npz \
  --observation-result .local/rl-runs/train-demonstrations/result.json \
  --code .local/rl-code/frozen-actions16 \
  --seed 316 --lua-samples 256
```

NPZ 必须含有限、已裁剪的 `observations[N,344]`。其配套 result 必须声明
`status=complete`、`training_only=true`、`split=train`，并通过
`demonstrations_sha256` 绑定 NPZ。工具不打开最终预留数据集。

输出新目录包含 `ppo-width128.zip` 与 `migration.json`。后者记录原/新模型、
结构、来源、观察与验证代码哈希，以及每一项检查；只有 `status=complete`
才是通过验收的迁移产物。网络结构与迁移身份也保存在 PPO ZIP 的
`astra_architecture`、`astra_width_migration` 属性中。

验收包括：旧参数与 Adam 矩逐元素完全一致、新增矩为零；真实 train 观察、
随机和边界观察上的 logits/value 浮点容差及 argmax 一致；SB3 保存/加载后的
参数和优化器完全一致；真实导出器和 Lua 推理一致；在独立模型副本上做短
Tensor 更新，确认新增 head 和隐藏参数获得非零梯度。最后一项不改保存的产物。
浮点比较使用 `atol=1e-5, rtol=1e-6`，argmax 不允许不一致。样本验收不等于
所有可能输入的浮点 argmax 证明，近似并列的动作仍可能对计算舍入敏感。

函数验收默认还对同一观察集按 batch1、32、256 分块（含不足整块的尾部），
同时比较原/扩宽网络以及各自分块结果与全批结果。逐项记录 logits/value
最大差和 argmax 一致性，避免只验证某一种 GEMM 批大小。Python 调用
`check_function(..., batch_sizes=(1, 4, 32, 256))` 可加入实际 worker 批大小。
对既有冻结产物补充检查时另存 acceptance 文件，不修改 ZIP 或原 migration.json。

## 对照与兼容

现有 actions16 exporter、`live_policy` 和 `nn.lua` 按实际矩阵尺寸遍历，没有
隐藏层 64 的硬限制；`PPO.load` 恢复 ZIP 的 128 结构。因此无需修改动作 schema
或游戏运行 Lua。原 `actions16_migrate` 专门处理 15→16 动作并重置优化器，
不能用来做此容量迁移。

配对实验从同一父模型分出原尺寸 A 和扩宽 B，保留相同运行代码、数据集、
采样、奖励、训练种子与决策预算。分别冻结最终模型，各完成 20 次自然投币，
不能跨尺寸或候选合并成绩。参数量约增为 2.31 倍，Lua actor 矩阵乘加量也约
增为 2.30 倍；这不是整个 MAME 训练耗时的倍率承诺，需以实际采样吞吐为准。
