# Windows 状态文件读写（0.1.3）

用户在 PC 运行最高难度首场 Blanka 时遇到：

```text
Invalid continuous match: controller error: training/runtime/play.lua:83:
training/attempts/l7-001/m01-blanka-status.json: Permission denied
```

0.1.2 的该行先删除旧状态 JSON，再把临时文件改名。Python 同时轮询读取
该状态文件；Windows 上，未允许共享删除的打开句柄会阻止删除或替换。
仅凭用户日志不能认定具体占用者，但这条代码存在读者与写者冲突的路径。
Windows 的相关规则见 [Microsoft DeleteFile 文档](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-deletefilea)。

## 修复

- 对局进行中，Lua 只向 `PREFIX-progress.jsonl` 追加进度，不删除该文件。
- Python 轮询的 `PREFIX-status.json` 在对局结束前不存在。
- 完整对局日志和截图写好之后，将关闭的临时 JSON 文件一次性发布为
  终态文件，之后不再更新或替换它。再次调用状态查询也不会重写终态。
- 写入失败仍判运行无效；不隐藏错误，不重放按键，不暂停对局等待重试。

由此移除了 Python 正在读取状态文件、Lua 却需要删除它的冲突路径。
冻结策略、难度设置及判胜规则不变。

## 验证范围

新增 Windows 专用测试使用真实 Windows 文件句柄及 Lua 5.4 I/O，验证
旧方案在读者持有句柄时删除失败，并对新模块执行并发追加、终态轮询、
完整 JSON 检查和拒绝覆盖测试。测试依赖 `lupa` 只安装在测试环境，
运行 CLI 不需要它。无该依赖时，普通测试会明确跳过相关项目；Windows
专用 CI 入口则必须具备 Windows 与 Lua，不能以跳过代替通过。

这些是实际操作系统上的文件通信测试，不是 Windows MAME 的完整通关
测试，也不替代用户机器上对相同版本的复验。此前的 Windows CI 只覆盖
Python 和打包流程，未覆盖 Lua 与 Python 的并发文件读写。
