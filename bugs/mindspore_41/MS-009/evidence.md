# MS-009 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`10`
- 退出码 0：`9`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant method_default`

代表性控制命令：

- `python reproducer.py --mode pynative --variant method_default`
- `python reproducer.py --mode graph --variant method_positional_offset`
- `python reproducer.py --mode graph --variant functional_default`

复验说明：失败仅由 Graph Tensor 方法省略默认参数触发，显式 `offset=0` 和功能式入口均通过。
## FlowMuT candidate

- Mutation site (`tau`): Tensor method wrapper for `diagonal_scatter`.
- Transformation (`m`): compare Tensor method default arguments,
  Tensor method with explicit `offset=0`, and functional
  `ops.diagonal_scatter`.
- Oracle (`o`): `Tensor.diagonal_scatter(src)` should honor the documented
  default `offset=0`, just like `ops.diagonal_scatter(input, src)`.
- Localized boundary: Graph lowering/default argument generation for the
  Tensor method wrapper.

Observed with MindSpore 2.9.0 CPU:

- PyNative `x.diagonal_scatter(src)` succeeds and returns:

```text
[[9. 4. 2.]
 [3. 8. 5.]]
```

- Graph `x.diagonal_scatter(src)` fails at compile time:

```text
RuntimeError: Miss argument input for parameter:offset
mindspore/core/ir/func_graph_extends.cc:249 GenerateDefaultValue
```

- Graph `x.diagonal_scatter(src, 0)` succeeds.
- Graph `x.diagonal_scatter(src, offset=0)` succeeds.
- Graph `ops.diagonal_scatter(x, src)` succeeds.

This isolates the bug to the Tensor method wrapper's default-argument handling
in Graph mode. It is not numerical precision.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T212733Z-p344546`
- Triggering case: `TDIA-003`
- Minimal script: `findings/MS-009/reproducer.py`
- Repetition summary: `findings/MS-009/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
in `metadata.json`.
