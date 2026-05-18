"""统一日志/控制台输出。

为了保持终端反馈干净，这里手写了一个简易封装：
- info/warn/error 三个等级
- 关键阶段使用 section 横线
- 自动带时间戳
"""

from __future__ import annotations

import sys
import time
from typing import Optional


_LEVEL_TAGS = {
    "info": "[INFO ]",
    "warn": "[WARN ]",
    "error": "[ERROR]",
    "ok": "[ OK  ]",
    "step": "[STEP ]",
}


def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _emit(level: str, msg: str) -> None:
    tag = _LEVEL_TAGS.get(level, "[INFO ]")
    stream = sys.stderr if level in ("warn", "error") else sys.stdout
    stream.write(f"{_ts()} {tag} {msg}\n")
    stream.flush()


def info(msg: str) -> None:
    _emit("info", msg)


def warn(msg: str) -> None:
    _emit("warn", msg)


def error(msg: str) -> None:
    _emit("error", msg)


def ok(msg: str) -> None:
    _emit("ok", msg)


def step(msg: str) -> None:
    _emit("step", msg)


def section(title: str, width: int = 72) -> None:
    bar = "=" * width
    sys.stdout.write(f"\n{bar}\n  {title}\n{bar}\n")
    sys.stdout.flush()


def subsection(title: str, width: int = 72) -> None:
    bar = "-" * width
    sys.stdout.write(f"\n{bar}\n  {title}\n{bar}\n")
    sys.stdout.flush()


def pause(msg: Optional[str] = None) -> None:
    """暂停等待用户回车继续。"""
    prompt = msg or "已暂停。请处理后按回车继续 (Ctrl+C 退出)..."
    try:
        input(f"\n>>> {prompt} ")
    except (EOFError, KeyboardInterrupt):
        sys.stdout.write("\n用户终止。\n")
        sys.exit(1)
