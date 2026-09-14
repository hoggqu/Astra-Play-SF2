# 同起点学习率对照

目标是比较已有 Normal 模型继续训练时的学习率，不把新模型从随机初始化重新训练。
两组各使用一份相同的完整 checkpoint（包括 Adam 和对手采样历史），默认参数如下：

| 参数 | 高学习率组 | 低学习率组 |
|---|---:|---:|
| learning rate | 0.0003 | 0.0001 |
| workers / device | 12 / CPU | 12 / CPU |
| rollout / mini-batch | 16384 / 256 | 16384 / 256 |
| 轮数 / 每轮决策 | 10 / 409600 | 10 / 409600 |
| 总训练决策 | 4096000 | 4096000 |
| 每輪固定权重验证 | 完整 3 次自然投币 | 完整 3 次自然投币 |
| 最终固定权重验证 | 完整 20 次自然投币 | 完整 20 次自然投币 |

均为 Normal（3）、seed 42、自适应对手采样；网络、奖励、动作、PPO epochs 等继承
同一 checkpoint。两个预算都按决策数，而非时间。跨主机/平台可能有数值差异，单个种子
的实验结果也不能替代多种子结论；首先比较最终通关率和失败对手，而不是完成速度。

从源码运行（替换路径和两个哈希；另一组只改 learning-rate）：

```sh
python -m experiments.rl.lr_comparison \
  --checkpoint MODEL.zip --checkpoint-sha256 MODEL_SHA256 \
  --dataset DATASET/manifest.json --dataset-sha256 DATASET_SHA256 \
  --learning-rate 0.0003 --rounds 10
```

默认每次创建唯一输出目录。开始前校验 checkpoint ZIP 和数据集 SHA256，再复制
不变的起点到输出目录；训练包也独立冻结。加载后显式更新学习率调度和 optimizer
参数组，保留 Adam 的步数与一、二阶矩。每轮结果核验实际生效的 learning rate。
省略普通 autotrain 的 `--learning-rate` 会继承 checkpoint，不自动恢复为默认值。

所有周期使用 `--all-attempts`，成功后仍打满三次。十轮完成后，固定最终权重，
额外打满二十次；不挑选最佳 checkpoint，不合并不同权重的通关。真实异常立即停止，
保留 invalid、日志和已保存权重。Ctrl+C 可安全停止训练，未完成预算不进入最终验证。

输出目录包含：

- `comparison.json`：参数、起点与最终模型哈希、最终通关次数和平均击败人数。
- `training/report.html`：各轮训练与验证成绩，包含实际学习率。
- `final-evaluation/result.json`：最终二十次自然投币的完整结果和审计。
- `final-evaluation.log`：最终验证日志。

执行包使用 managed v7，保留 settlement v11 的普通 KO 计时递减修复。
原有冻结运行和模型不修改。主项目冻结 V4 的策略与验证入口不受影响。
