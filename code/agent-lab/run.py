#!/usr/bin/env python
"""免安装直接跑：python code/agent-lab/run.py eval

把 code/ 与 code/agent-lab 加进 sys.path 后转发给对应模块。
这样读者 clone 下来不用 pip install、不用设 PYTHONPATH 就能跑。
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # code/  → agentkit
sys.path.insert(0, str(HERE))  # code/agent-lab → agentlab

from agentlab.__main__ import main  # noqa: E402

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 ACP=936，中文会花屏
    sys.exit(main())
