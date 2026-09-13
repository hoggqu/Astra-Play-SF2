# 从干净的 Git 副本构建新接口

完整动作与可见反馈实验可以只用本分支的公开源码构建。
不需要原作者的 `.local/rl-code/`、历史训练目录、冻结 PPO 权重、ROM 或存档。
先安装本实验 README 中的 Python 可选依赖，再执行：

```sh
python -m experiments.rl.full_interface_bootstrap --output .local/rl-code/full85-001
```

该命令只构建和校验代码，不启动 MAME、不收集对局、不进行 PPO 更新。
输出目录必须是新目录。命令使用 Python 标准库和项目代码，无 shell 专用操作，
Windows、Linux、macOS 使用相同参数。

构建过程从公开的 actions16 构建器开始，依次生成 round-chain、显式声明的
settlement-v5 修订、pulsed-normal，最后生成 **85 动作、800 维输入**的新接口。
中间目录在系统临时目录内创建；最终产物保留校验所需的父版本源码、清单和
生产代码副本。它可以移动到其他目录，不依赖构建时的临时路径。
没有复制私人 specialist 构建器或历史数据；它是新的公开来源构建身份，
不能冒用旧冻结模型的通关成绩。

可以随后创建一个尚未训练的新 PPO：

```sh
python .local/rl-code/full85-001/launch.py initialize --output .local/rl-runs/full85-initial-001 --seed 42
```

该步骤只初始化网络和写入 `ppo-initial.zip`，训练步数为零。
实际训练还需要 MAME 0.288、兼容 ROM 和经过校验的训练数据集；本构建命令不
下载或生成这些数据，也不把随机初始化模型当作已有胜率的模型。

已有审核过的父代码快照时，仍可使用底层构建器：

```sh
python -m experiments.rl.full_interface_builder --source PATH_TO_PARENT_ROOT --output NEW_OUTPUT
```

底层父目录必须包含其原始 `build.json`、执行源码及完整校验依赖，不能只复制
一个 Lua 文件并重新标记接口。新用户优先使用不需要父目录的 bootstrap 命令。

## 独立性测试

```sh
python -m unittest experiments.rl.test_full_interface_bootstrap -v
```

测试在临时目录复制公开源码，明确排除 `.local`、`MAME`、私人 Skill 和 Git 历史；
在子进程中构建，再删除整个源码副本、移动结果目录，重新验证全部构建身份，
并运行独立 launcher 的帮助入口。测试不启动模拟器或训练模型。
