#!/usr/bin/env python
"""免安装直接跑：python code/hr-agent/run.py eval"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # code/  → agentkit
sys.path.insert(0, str(HERE))  # code/hr-agent → hragent

from hragent.__main__ import main  # noqa: E402

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
