# 离线训练报告

`training_report.py` 用 Python 标准库读取已落盘结果，生成可直接打开的 `report.html` 和机器可读的 `summary.json`。HTML 自包含，没有网络请求、CDN 或 JavaScript。它不会启动 MAME、加载模型或读取巨大的逐帧对战文件。

```sh
# autotrain 顶层目录，包含 run/：默认将报告写在这个顶层目录
python -m experiments.rl.training_report .local/my-training

# 原始 campaign 目录：将衍生报告写到独立目录
python -m experiments.rl.training_report .local/my-campaign \
  --output-dir .local/my-campaign-report
```

Python 调用：

```python
from experiments.rl.training_report import write_report
result = write_report(campaign_path, report_directory)
print(result['report_path'])
```

输入可以是含 `run/` 的训练顶层，也可以是原 campaign 目录。输出不得位于实际 campaign 内；训练顶层包含 `run/` 时，可以将报告写在顶层。生成器只替换自己生成的两个报告文件，拒绝输出文件符号链接及覆盖其他已有文件。运行中的结果可以重新汇总，原记录不会改变。

报告按训练轮列出完成决策数、训练单局与整场胜率、平均回报、末模型 SHA、每次自然投币击败对手数与失败对手，并提供各对手的训练和自然投币明细。训练采样过程中的多个策略与轮末固定模型的评估分开；不同模型不合并计算通关率。最新模型路径和 SHA 取自结果记录，报告不重新审计 ZIP。

训练统计只接受完整的 Python `episodes.jsonl` 记录，核对重复 episode、原生单局结果和成熟比分；不读取或重复计算 Lua 副本。未完成、采样中、无效阶段单独显示，不加入已完成训练分母。自然投币只有通过既有时序、接口及逐次尝试审计的结果才计入成绩；逐对手数据来自小型状态文件。原生 `round` 和最终比分齐全时，两者差值给出平局数；细节缺失时明确显示未知或覆盖范围，不补零。

这是一份已有证据的汇总，不代替完整原生轨迹审计。源记录核对失败会保留问题说明，CLI 返回 2；没有核对问题返回 0。八项测试覆盖原统计语义、侧车明细、未完成状态、HTML 转义与输入文件保护。

报告逐轮展示实际 workers、全局 rollout 决策数、mini-batch、epochs 和采样更新次数，供 A/B 对照。旧记录缺少参数时显示未知，不反推默认值。
