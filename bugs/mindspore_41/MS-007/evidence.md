# MS-007 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`6`
- 退出码 0：`4`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --rhs selected`
- `python reproducer.py --mode pynative --rhs full`

代表性控制命令：

- `python reproducer.py --mode pynative --rhs selected`
- `python reproducer.py --mode graph --rhs full`
- `python reproducer.py --mode graph --rhs scalar`

复验说明：同一索引操作在两模式呈现互相相反的 RHS 接受规则，标量控制排除了赋值路径整体不可用。
## FlowMuT candidate

- Mutation site (`tau`): Tensor boolean-mask setitem.
- Transformation (`m`): assign either a scalar, a 1-D tensor with one value per
  selected element, or a full-shape tensor through the same boolean mask in
  PyNative and Graph modes.
- Oracle (`o`): boolean-mask setitem should use the same RHS shape rules across
  execution modes.
- Localized boundary: Graph lowering/inference for boolean-mask Tensor setitem.

Observed with MindSpore 2.9.0 CPU:

- Scalar RHS works in both modes.
- A 1-D RHS with length equal to the number of `True` mask elements works in
  PyNative, but Graph fails during compile-time broadcast inference:

```text
operands could not be broadcast together with remapped shapes
[original->remapped]: (6,1) and requested shape (3,4)
```

- A full-shape `(3, 4)` RHS fails in PyNative, but Graph accepts it and writes
  the full tensor's values at masked positions.

This points to a Graph-mode implementation that treats boolean-mask setitem as
full-tensor masked update, while PyNative follows selected-view shape rules.
The issue is a shape/behavior contract mismatch, not numerical precision.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T205330Z-p313415`
- Triggering cases: `TSET-004`, `TSET-006`, `TSET-007`
- Additional confirmation run: `artifacts/behavioral/20260801T105800Z-p1017291`
- Additional confirmation case: `TSET-012`, a Python bool-list row setitem
  variant that fails in Graph with the same full-shape broadcast rule mismatch.
- Minimal script: `findings/MS-007/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
in `metadata.json`.
