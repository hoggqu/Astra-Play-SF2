# Astra-Play-SF2

本项目全程由 Astra 自主完成，人工未约束实现方法和思路，只参与讨论和设置验证目标。

![GPT-Astra 在街机前使用 Ken 玩街霸 II 的卡通插画](docs/images/astra-plays-sf2.png)

**0.1.3 Windows 修复：** 对局进度与终态文件分开发布，解决轮询读取和
状态文件替换相撞时的 `Permission denied` 问题。[说明](docs/windows-status-io.md)

**0.1.1 难度修复：** 旧启动方式可能出现 DIP 显示最高难度、游戏内部仍为 Normal。
现在改为启动前配置，并检查游戏内部 RAM。历史跨难度结论需要复核，详见
[问题诊断与证据规则](docs/difficulty-fix.md)。

**0.1.2 开局优化：** 按原生游戏状态判断投币和 Start 时机，不再固定空等
9000 帧，详见[就绪检查与实测](docs/entry-readiness.md)。

用冻结的 V4 策略操控 Ken，在 MAME 中自动验证《Street Fighter II: The World Warrior》通关结果。运行时只需要 Python、MAME 和你自己的兼容 ROM，**不依赖 AI Agent、模型或 API Key**。

[English](README.md) · [Agent 接手](AGENTS.md) · [实现说明](docs/architecture.md)

速度支持 `--speed normal`（默认）、`--speed 2x`、`--speed 4x` 和
`--speed fast`（不限速）。2 倍和 4 倍为 0.1.4 新增，实际速度受机器性能限制。

## 安装和首次运行

先安装 **Python 3.10+**、**MAME 0.288**，准备兼容 `sf2 / World 910522` 的 ROM。[安装文档](docs/installation.md)提供官方软件下载入口。项目不包含模拟器、ROM 或存档。

本地缺少依赖时，允许 Agent 按 [AGENTS.md](AGENTS.md#missing-dependencies-automatic-project-local-setup) 自由寻找 MAME 0.288 和兼容 ROM 的下载来源，自行下载、安装到项目的 `.local/` 目录并校验，无需玩家先提供地址或逐个确认来源。推荐链接只是搜索起点；只有搜索并尝试合适的替代方案后仍无法完成安装，才请玩家协助，并说明已尝试的办法和具体障碍。

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

安装脚本创建虚拟环境并安装 CLI。MAME、Python 和 ROM 可由使用者准备，或交给 Agent 按安装说明补齐；bootstrap 本身不包含模拟器或 ROM 下载器。`doctor` 会检查版本、ROM 校验和输出目录，无需修改脚本中的路径。

## 验证规则和批量运行

默认普通难度、正常速度、一次挑战。每个难度在开机前配置好，并创建独立实例；同一难度的后续挑战沿用该实例，等待游戏自然结束后重新投币，不 continue、不重置、不读写存档、不在整场对战中暂停或修改打法。

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

## 从第一次试玩到 V4 冻结

仓库同时收录了从旧 Skill 提炼的[训练方法与 Agent 接手](docs/training-method.md)、
[训练演进历史](docs/training-history.md)和[对手策略与失败经验](docs/matchup-lessons.md)。
内容包含真实败局采样、配对实验、预留开局、批量提速、淘汰候选及冻结验证，
不包含历史存档、截图、日志和逐次数据。旧训练执行器未随 CLI 移植；新读者可
直接复验 V4，后续训练工具和实验按文档另建。

项目始于 **2026 年 9 月 8 日**，最初只是“找一个 Mac 版 MAME，用 Ken 玩街霸 II”，随后逐步变成了一套可以独立复验的固定策略：

1. **先学会打。** 观察位置、血量和动作，建立按游戏帧执行的 Lua 输入，把规律整理为其他 Agent 能接手的说明。
2. **针对弱项训练。** 用对局开局存档反复练习，逐小局记录胜负；重点强化 Honda、Blanka、Vega、Sagat、Bison 等对手，并让子 Agent 协助失败复盘和训练工具优化。
3. **明确正式规则。** 训练与游玩分开：正式挑战不读档、不 continue、不在整场对战中暂停。验证从重置开局改为等待游戏自然结束后重新投币，覆盖更多实际开局。
4. **冻结并检验 V4。** 用同一套选定打法验证 Normal（3）到最高难度（7），保留全部失败，最终各标签档位均达到**自然投币连续五次通关**；当时只有 DIP 校验，内部难度尚待复核。最终 V4 验证批次共 **55 次挑战、45 次通关，累计通关率 81.8%**；逐小局 **1,114 胜、96 负、3 平**。这些数字属于最终批次，不是此前所有训练的总和，也不是未来胜率保证。
5. **整理为独立项目。** 保留冻结的出招策略、判胜核心和对手模式配置，加入 Git、跨平台安装脚本、统一 CLI、Agent 接手说明及证据报告，让人工和 Agent 都能直接复验，运行时无需 AI 模型。

冻结文件身份、分难度成绩及独立 CLI 的实机测试见[验证记录](docs/validation.md)。

### 时间与 Token 消耗

以下是**本任务从最初提问到 2026 年 9 月 11 日 13:28:38（北京时间）的累计统计**，涵盖训练、验证与 CLI 工程整理，不仅是 V4 的训练成本。数据来自去重后的本地逐响应日志，包含主 Agent、六个工作子 Agent 和四个自动审批审查实例；不含其他独立任务，封面图的生成任务也不在此统计内。

| 统计口径 | 累计 |
|---|---:|
| 从首次提问起的自然时间 | 72 小时 52 分 47 秒 |
| 实际任务工作时段，合并并行时间 | 51 小时 21 分 46 秒 |
| 各 Agent 工作时长相加，包含并行工作 | 79 小时 20 分 50 秒 |
| **Token 总计：输入 + 输出** | **1,345,709,004（约 13.46 亿）** |
| 其中：缓存输入 | 1,314,050,816 |
| 非缓存输入 | 27,393,080 |
| 输出，已包含推理部分 | 4,265,108 |

工作时段包含分析、工具执行和任务内等待。缓存占输入的 **97.96%**；非缓存输入加输出为 **31,658,188 Token**。这是使用量统计，不是费用估算。最后一次 Git／跨平台整理耗时 **28 分 58 秒**，包含子 Agent 共 **16,332,890 Token**，已计入上述总数；之后的统计和 README 编辑不在这份快照内。
