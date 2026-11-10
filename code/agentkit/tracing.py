"""轨迹记录：Agent 的每一步都必须可回放、可审计、可算账。

评测与排障都依赖轨迹，所以这里刻意做得很薄：
- 事件追加写 JSONL，进程崩了也不会丢已写内容；
- 每条事件带 `run_id`、`seq`、`ts`，回放时能按序还原；
- 敏感内容默认**不落盘原文**，只记长度与哈希 —— 简历筛查场景尤其重要。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def fingerprint(text: str) -> str:
    """文本指纹：用于在不落原文的前提下比对「是不是同一份内容」。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class TraceEvent:
    run_id: str
    seq: int
    ts: float
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": round(self.ts, 6),
            "kind": self.kind,
            **self.fields,
        }


class Tracer:
    """JSONL 轨迹写入器。"""

    def __init__(self, path: Path | str | None = None, *, run_id: str = "run", echo: bool = False):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.echo = echo
        self.events: list[TraceEvent] = []
        self._seq = 0
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def event(self, kind: str, **fields: Any) -> TraceEvent:
        self._seq += 1
        ev = TraceEvent(
            run_id=self.run_id,
            seq=self._seq,
            ts=time.time(),
            kind=kind,
            fields=fields,
        )
        self.events.append(ev)
        line = json.dumps(ev.to_dict(), ensure_ascii=False)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.echo:
            print(line)
        return ev

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MemoryTracer(Tracer):
    """只留在内存里的 tracer，测试用。"""

    def __init__(self, run_id: str = "test"):
        super().__init__(path=None, run_id=run_id)

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def find(self, kind: str) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == kind]
