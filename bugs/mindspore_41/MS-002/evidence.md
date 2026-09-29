# MS-002 evidence

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

复验说明：两种模式均成功执行，但当前输出直接显示返回嵌套结构与 indices 是否存在的差异。
## FlowMuT candidate

- Mutation site (`tau`): a rank-2 float32 tensor edge before a reduction.
- Transformation (`m`): execute the positional `Tensor.max(0)` or
  `Tensor.min(0)` overload in PyNative and Graph modes.
- Oracle (`o`): identical values, indices, nested return structure and dtypes
  across modes.
- Localized boundary: the first violated observation is the Tensor reduction
  method's return boundary.

MindSpore 2.9.0 documents `Tensor.max(dim, keepdim=False)` and
`Tensor.min(dim, keepdim=False)` as returning `(values, indices)`.  PyNative
follows that contract.  The same `nn.Cell` in Graph mode returns only the value
Tensor for each call, silently dropping indices and changing the nested return
structure.

The cross-mode difference also reproduces with MindSpore 2.7.1, the framework
version used by the paper's experimental setup.

The result is independent of max versus min and is therefore treated as one
likely overload-resolution root cause, not two issues.  It was reproduced in
three isolated processes per mode.  See `repetitions/summary.json`.

## Duplicate search

No matching open or closed issue was found in `mindspore-ai/mindspore` on
2026-08-01 using the queries in `metadata.json`.
