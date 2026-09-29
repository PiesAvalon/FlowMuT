# MS-001 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`2`
- 退出码 0：`2`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative`
- `python reproducer.py --mode graph`

代表性控制命令：

- 复现器内部独立参考或跨模式对照

复验说明：复现器同时计算 NumPy float64 参考值；两种模式得到完全相同的异常误差，且返回 dtype 均为 float64。
## FlowMuT candidate

- Mutation site (`tau`): the float64 tensor edge entering `log_softmax`.
- Original transformation (`m`): replace `ops.log_softmax(x, -1)` with the
  equivalent `ops.log(ops.softmax(x, -1))`.
- Original oracle (`o`): approximate-value and metadata agreement.
- Localized boundary: the input predicate holds; the first violated
  observation is the `log_softmax` output.

The original metamorphic alert was reclassified as an API-contract issue.  The
2.9.0 documentation limits the input to float16/float32 and explicitly lists a
`TypeError` for other dtypes.  Therefore float64 is not a legal numerical
mutation for an equivalence bug, but accepting it silently is itself a
checkable termination/metadata inconsistency.

## Observations

For the deterministic 2x3 float64 input, both PyNative and Graph mode return a
float64 tensor.  The maximum absolute error against a stable NumPy float64
reference is `1.4674482509136055e-06`.  In contrast,
`ops.log(ops.softmax(x, -1))` agrees with the same reference within
`2.220446049250313e-16` for this bounded input.

The same result is present in MindSpore 2.7.1 (the paper's target version) and
2.9.0. It was observed in the full 150-candidate campaign and in three later
isolated runs per execution mode.  See `repetitions/` for captured JSON.

## Environment

- MindSpore 2.7.1 and 2.9.0 binary wheels
- Python 3.11.15
- Ubuntu 24.04.4 LTS under WSL2
- CPU: Intel Core i7-14700K
- Device target: CPU

## Duplicate search

The open and closed issues in `mindspore-ai/mindspore` were searched on
2026-08-01 using three queries recorded in `metadata.json`.  No issue matching
this float64 acceptance/precision symptom was found.  A broad `log_softmax
dtype` query returned issue #143, which is unrelated (BERT checkpoint shape).
