# hr-agent：AI 简历筛查与模拟面试

> 对应笔记：[`06-AI招聘应用`](../../06-AI招聘应用/)

## 跑起来

```bash
python code/hr-agent/run.py eval                     # 离线评测（5 份合成简历 + 一轮模拟面试）
python code/hr-agent/run.py eval --show-report       # 打印完整报告样例
python code/hr-agent/run.py model JD.txt             # 打印 JD 的能力模型
python code/hr-agent/run.py screen 简历.txt --jd JD.txt
python code/hr-agent/run.py interview 简历.txt --auto
```

## 筛查流水线（四段）

```
屏蔽敏感字段 → 硬性门槛过滤 → 语义匹配打分 → 结论与理由
```

| 段 | 做什么 | 关键约束 |
|---|---|---|
| 屏蔽 | 年龄/性别/婚育/民族/籍贯/政治面貌/照片/身份证/健康 | 生产必须开启；屏蔽记录进报告 |
| 硬过滤 | 学历、年限、城市、薪资 | 一票否决，**不进加权分数** |
| 打分 | 逐能力维度评分 + 原文证据 | 证据必须能在简历里逐字找到 |
| 结论 | pass / review / reject + 理由 | 证据覆盖不足时强制降级为 review |

## 离线评测结果（当前基线）

| 指标 | 结果 | 含义 |
|---|---|---|
| `hard_filter_accuracy` | 100% | 硬门槛判对（R002 年限不够、R003 学历不够都被拦下） |
| `decision_accuracy` | 100% | 结论与人工标注一致 |
| `evidence_grounding_rate` | 100% | 每条证据都能回指原文，无幻觉证据 |
| `bias_score_delta` | 0.0 | R005（多敏感字段）与 R004（对照）得分**完全一致** |
| `rank_concordance` | 100% | 与人工排序一致 |
| `top2_recall` | 100% | 前二名与人工期望一致 |

模拟面试：6 题 / 6 个能力维度全覆盖 / 每题都有原话引用 / 0 个 issue。

## 为什么用「规则后端 + 模型后端」两套

- `RuleMatchBackend` / `RuleInterviewerBackend`：确定性、可离线、可当基线；
- `LlmMatchBackend` / `LlmInterviewerBackend`：真实能力上限更高，但**必须配证据回指校验**。

生产建议：**模型抽、规则校**。模型抽出来的能力 key 对不上词典的直接丢弃并记账。

## 敏感字段处理

`bias.mask_sensitive()` 逐字段（而不是一把大正则）替换，因此：

- 能精确记账「屏蔽了什么」，写进报告供合规审计；
- 新增敏感字段只改词典，不动主逻辑；
- `audit_masking()` 用屏蔽前后各打一次分，**差值必须为 0** —— 这是回归防线，
  一旦有代码路径绕过屏蔽，审计立刻报警。

## 数据声明

`hragent/evals/fixtures.py` 中的 JD 与简历**全部为合成虚构数据**，不含任何真实个人信息、
不来自任何真实招聘渠道，仅用于跑通流程与回归测试。
