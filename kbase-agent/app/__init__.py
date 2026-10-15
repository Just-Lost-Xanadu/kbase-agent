"""kbase-agent 应用包。

导入本包时统一把标准输出/错误流切到 UTF-8。

为什么要在包的入口做这件事：代码与日志里有大量中文，而 Windows 上
`sys.stdout` / `sys.stderr` 的编码来自 ANSI 代码页（英文/其他区域是 cp1252、cp437），
`print` 中文会直接抛 `UnicodeEncodeError`。而 MCP 子进程的 stderr 与
`python ... > log.txt` 这类重定向**本身就是管道**，走的正是这条路径。
实测后果是误导性的：`pipeline.index()` 在索引**已经写成功之后**于最后一句 print 上抛异常，
服务把它报成"建索引失败 / Agent 引擎未就绪"，而盘上的索引其实是好的。
`scripts/eval_e2e.py` 早就单独做过同样的 reconfigure，app 侧一直漏着。

这里只改编码、不碰缓冲与重定向，因此对 MCP stdio（stdout 走 JSON-RPC）是安全的：
JSON-RPC 本来就要求 UTF-8。
"""

from __future__ import annotations

import contextlib
import sys

# 抑制异常是刻意的：pytest 会把 sys.stdout 换成自己的捕获对象（可能没有 reconfigure），
# 关闭了 stdout 的宿主进程也会在这里失败。编码修不了，不该反过来把导入搞崩。
with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
