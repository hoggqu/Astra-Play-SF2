# Astra-Play-SF2

用冻结的 V4 策略操控 Ken，在 MAME 中自动验证《Street Fighter II: The World Warrior》通关结果。运行时只需要 Python、MAME 和你自己的兼容 ROM，**不依赖 AI Agent、模型或 API Key**。

[English](README.md) · [Agent 接手](AGENTS.md) · [实现说明](docs/architecture.md)

## 安装和首次运行

先安装 **Python 3.10+**、**MAME 0.288**，准备兼容 `sf2 / World 910522` 的 ROM。[安装文档](docs/installation.md)提供官方软件下载入口。项目不包含模拟器、ROM 或存档。

macOS / Linux，在下载并解压的项目目录中执行：

```sh
bash scripts/bootstrap.sh
source .venv/bin/activate
astra-sf2 configure --mame "/模拟器路径/mame" --rom-dir "/ROM目录"
astra-sf2 doctor
astra-sf2 verify --difficulty 3
```

如果系统默认 Python 太旧，可指定解释器：

```sh
ASTRA_PYTHON=/path/to/python3.12 bash scripts/bootstrap.sh
```

Windows PowerShell：

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\astra-sf2.exe configure --mame "C:\MAME\mame.exe" --rom-dir "C:\MAME\roms"
.\.venv\Scripts\astra-sf2.exe doctor
.\.venv\Scripts\astra-sf2.exe verify --difficulty 3
```

安装脚本创建虚拟环境并安装 CLI。MAME、Python 和 ROM 由使用者准备；`doctor` 会检查版本、ROM 校验和输出目录。无需修改脚本中的路径。

## 验证规则和批量运行

默认普通难度、正常速度、一次挑战。每次 `verify` 创建独立实例；同一次验证中的后续挑战沿用该实例，等待游戏自然结束后重新投币，不 continue、不重置、不读写存档、不在整场对战中暂停或修改打法。

```sh
# 最高难度，高速验证五轮
astra-sf2 verify --difficulty 7 --attempts 5 --speed fast

# Normal 到最高难度，每档五轮，全部计入成绩
astra-sf2 verify --difficulty all --attempts 5 --speed fast

# 每档最多二十轮，达到连续五次通关即结束该档
astra-sf2 verify --difficulty all --attempts 20 --consecutive 5 --speed fast
```

难度范围是 3–7。`--attempts` 是**每个难度**的上限。失败会保留并中断连胜；达到连续五次不意味着包含之前失败在内的总通关率为 100%。

退出码：`0` 达成此次目标；`1` 正常输掉挑战或未达到指定连胜；`2` 环境、控制或证据校验异常。异常不会自动重发指令或把中断记录补成通关。

## 成绩与复核

终端会输出本次运行目录，内含 `report.html`、`report.md`、`report.json`、逐帧记录、逐局胜负、截图和文件校验清单。报告按难度和对手分别统计；失败和异常不会被删除。

```sh
astra-sf2 audit RUN_DIR
astra-sf2 report RUN_DIR
```

自动结果 `gameplay_clear` 表示 Ken 在同一次挑战中取得 11 场完整对手战的胜利。视觉复核另外记录：人工或 Agent 必须实际查看该轮的选人图、Bison 结果图和三张结局图，再执行：

```sh
astra-sf2 review RUN_DIR --attempt l3-001 --reviewer "审阅者姓名" --decision approve
```

有矛盾则使用 `reject`。脚本不会把“自动判胜”写成“已看过图片”。详见[证据规则](docs/verification.md)。

## 平台与历史成绩

Python 工具和启动脚本按 macOS、Linux、Windows 设计；CI 配置覆盖三个系统。实机测试结果见[验证记录](docs/validation.md)。没有在某个平台实际运行过 MAME，就不声称已验证该平台的游戏兼容性。

V4 原训练环境共 55 次挑战、45 次通关，最终每档达到连续五次通关。这是历史样本，不能作为新安装环境或未来胜率的保证。原始历史资料与当前可分发项目分开保存。
