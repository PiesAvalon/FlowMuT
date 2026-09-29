# MS-006 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`4`
- 退出码 0：`3`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --api tensor`

代表性控制命令：

- `python reproducer.py --mode pynative --api tensor`
- `python reproducer.py --mode graph --api ops`

复验说明：相同输入下只有 Graph Tensor 方法路径失败，功能式算子与 PyNative Tensor 方法均成功。
## FlowMuT candidate

- Mutation site (`tau`): a deprecated-but-documented Tensor method,
  `Tensor.index_add(indices, y, axis)`.
- Transformation (`m`): execute the same method in PyNative and Graph modes,
  and compare it with the corresponding functional API `ops.index_add`.
- Oracle (`o`): the Tensor method is documented as CPU-supported and works in
  PyNative; the functional API works in Graph; therefore Graph should either
  compile the Tensor method or report a documented unsupported limitation.
- Localized boundary: Graph parser registration for deprecated Tensor methods.

MindSpore 2.9.0 documents `Tensor.index_add(indices, y, axis, ...)` as CPU
supported.  The same docstring warns that this overload will be removed after
2.9.0, but the overload is still present in 2.9.0 and its example uses
`x.index_add(indices, y, axis=1)`.

Observed behavior:

- `Tensor.index_add` works in PyNative mode on CPU.
- `ops.index_add` works in Graph mode on CPU.
- `Tensor.index_fill`, another deprecated Tensor method, works in Graph mode in
  the neighboring behavioral case.
- `Tensor.index_add` in Graph mode fails before kernel execution with:

```text
As a deprecated Tensor method, 'index_add' should be registered in
graph/_parse/deprecated/deprecated_tensor_method.py::deprecated_tensor_method_map
```

The error is reported as a framework unexpected exception and asks the user to
create an upstream issue.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T194833Z-p253662`
- Triggering case: `TUPD-001`
- Minimal script: `findings/MS-006/reproducer.py`

## Duplicate search

No exact matching GitHub or Gitee issue was found on 2026-08-01 using the
queries in `metadata.json`.
