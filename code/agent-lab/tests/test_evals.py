"""评测集回归：任何一个用例挂掉都算代码回归。"""

from __future__ import annotations

from agentlab.evals.cases import CASES
from agentlab.evals.metrics import aggregate
from agentlab.evals.run_eval import run_case


def test_all_cases_pass():
    results = [run_case(case) for case in CASES]
    failures = [f"{r.name}: {r.reasons}" for r in results if not r.passed]
    assert not failures, "\n".join(failures)


def test_aggregate_shape():
    results = [run_case(case) for case in CASES]
    summary = aggregate(results)
    assert summary["cases"] == len(CASES)
    assert summary["pass_rate"] == 1.0
    assert set(summary["by_profile"]) == {"bot", "agent", "superagent"}
