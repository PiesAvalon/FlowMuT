# MS-038 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`2`；非零：`6`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant scalar_max_output_size`
- `python reproducer.py --mode graph --variant scalar_thresholds`
- `python reproducer.py --mode graph --variant all_scalar_args`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_args_control`
- `python reproducer.py --mode graph --variant tensor_args_control`

复验说明：CPU 算子由 all-Tensor 控制验证，失败集中在公开声明的 Python 标量参数。
## FlowMuT candidate

- Mutation site (`tau`): direct `ops.NonMaxSuppressionWithOverlaps` primitive
  scalar argument handling.
- Transformation (`m`): pass documented Python scalar arguments for
  `max_output_size`, `overlap_threshold` and `score_threshold` instead of
  scalar Tensors.
- Oracle (`o`): the primitive documents these arguments as Tensor-or-scalar
  unions and lists CPU support. The all-Tensor scalar control succeeds.

## Observed behavior

Inputs:

```text
overlaps = Tensor([[1.0, 0.2, 0.7],
                   [0.2, 1.0, 0.1],
                   [0.7, 0.1, 1.0]], float32)
scores = Tensor([0.9, 0.8, 0.7], float32)
```

Failing `max_output_size` scalar:

```text
ops.NonMaxSuppressionWithOverlaps()(overlaps, scores, 2, Tensor(0.5), Tensor(0.0))

PyNative -> TypeError: input argument[max_output_size]:Int64; valid type list: {Tensor[Int32]}.
Graph    -> TypeError: input argument[max_output_size]:Int64; valid type list: {Tensor[Int32]}.
```

Failing threshold scalars:

```text
ops.NonMaxSuppressionWithOverlaps()(overlaps, scores, Tensor(2, int32), 0.5, 0.0)

PyNative -> RuntimeError: The method 'GetShapeVector()' doesn't implement
Graph    -> RuntimeError: The method 'GetShapeVector()' doesn't implement
```

## Passing control

This passes in both PyNative and Graph mode:

```text
ops.NonMaxSuppressionWithOverlaps()(
    overlaps,
    scores,
    Tensor(2, int32),
    Tensor(0.5, float32),
    Tensor(0.0, float32),
)
```

The control returns selected indices with shape `(2,)` and dtype `int32`.

## Why this is a real issue

This is not a precision issue. The operation fails before returning selected
indices.

This is not a CPU unsupported issue. CPU support is documented and the
all-Tensor scalar control passes on CPU.

The threshold-scalar path also reports an internal `GetShapeVector()` RuntimeError
rather than a clear documented unsupported-input error.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260802T044717Z-p1653353`
- Exploratory run: `artifacts/behavioral/20260802T044122Z-p1648872`
- Triggering cases: `TNMS-001`, `TNMS-002`, `TPRIMSC-008`
- Passing controls: `TNMS-003`, `TPRIMSC-009`
- Minimal script: `findings/MS-038/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
