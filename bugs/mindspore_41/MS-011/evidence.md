# MS-011 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`4`
- 退出码 0：`3`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_method`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_method`
- `python reproducer.py --mode graph --variant functional`

复验说明：仅 Graph Tensor 方法入口失败，功能式算子与 PyNative Tensor 方法均返回正确对角矩阵。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.diag()` method wrapper.
- Transformation (`m`): compare Tensor method with functional `ops.diag`.
- Oracle (`o`): Tensor method documentation refers to `ops.diag`, and
  `ops.diag` is documented as CPU-supported.
- Localized boundary: Graph lowering/registration of the deprecated Tensor
  method.

Observed with MindSpore 2.9.0 CPU on:

```text
[1, 2, 3]
```

- PyNative `x.diag()` succeeds and returns a `(3, 3)` int32 diagonal matrix.
- Graph `ops.diag(x)` succeeds and returns the same `(3, 3)` int32 diagonal
  matrix.
- Graph `x.diag()` fails during compile with:

```text
As a deprecated Tensor method, 'diag' should be registered in
graph/_parse/deprecated/deprecated_tensor_method.py::deprecated_tensor_method_map
```

The same Graph error also says this is a framework unexpected exception and
asks users to create an issue. This is not a numerical precision issue: the
failure happens before execution and is isolated to Tensor method registration.

## Documentation basis

- `mindspore.Tensor.diag` in the 2.9.0 docs: "For details, please refer to
  mindspore.ops.diag()."
- `mindspore.ops.diag` in the 2.9.0 docs returns a tensor with the input on the
  diagonal and lists supported platforms as `Ascend` `GPU` `CPU`.
- `mindspore.ops.nonzero(as_tuple=True)` was explicitly discarded from this
  pass because the 2.9.0 docs note that `as_tuple=True` is currently Ascend-only.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T221027Z-p382885`
- Triggering case: `TMAT-009`
- Minimal script: `findings/MS-011/reproducer.py`
- Repetition summary: `findings/MS-011/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
in `metadata.json`.
