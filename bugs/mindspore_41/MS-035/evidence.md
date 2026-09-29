# MS-035 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`10`
- 退出码 0：`8`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant scalar_tensor_axis0`
- `python reproducer.py --mode graph --variant scalar_tensor_axis1`

代表性控制命令：

- `python reproducer.py --mode pynative --variant scalar_tensor_axis0`
- `python reproducer.py --mode graph --variant int_control`
- `python reproducer.py --mode graph --variant single_element_tensor_control`
- `python reproducer.py --mode graph --variant vector_tensor_control`

复验说明：失败被定位到 0-D Tensor repeats，而不是 CPU、Tensor repeats 整体或特定 axis。
## FlowMuT candidate

- Mutation site (`tau`): `ops.repeat_interleave` argument handling for
  `repeats`.
- Transformation (`m`): pass `repeats=ms.Tensor(2, ms.int32)` instead of the
  Python integer `2`, while keeping the same input, axis, dtype and device.
- Oracle (`o`): `ops.repeat_interleave` documents `repeats` as
  `Union[int, tuple, list, Tensor]` and CPU support. PyNative accepts the
  scalar Tensor form, and Graph accepts nearby int, list/tuple and Tensor-vector
  controls.

## Observed behavior

Input:

```text
x = Tensor([[3, 1, 2],
            [6, 5, 4]], int32)
```

Failing cases:

```text
ops.repeat_interleave(x, Tensor(2, int32), axis=0)

PyNative -> Tensor([[3, 1, 2],
                    [3, 1, 2],
                    [6, 5, 4],
                    [6, 5, 4]], int32)
Graph    -> RuntimeError: Convert data failed
```

```text
ops.repeat_interleave(x, Tensor(2, int32), axis=1)

PyNative -> Tensor([[3, 3, 1, 1, 2, 2],
                    [6, 6, 5, 5, 4, 4]], int32)
Graph    -> RuntimeError: Convert data failed
```

Representative Graph error:

```text
RuntimeError: Convert data failed

Framework Unexpected Exception Raised:
This exception is caused by framework's unexpected error.
mindspore/ccsrc/frontend/jit/ps/static_analysis/prim.cc:729 RunPyInferValue
```

## Passing controls

All of these pass in both PyNative and Graph mode:

```text
ops.repeat_interleave(x, 2, axis=0)
ops.repeat_interleave(x, Tensor([2], int32), axis=0)
ops.repeat_interleave(x, Tensor([1, 2, 1], int32), axis=1)
ops.repeat_interleave(x, [1, 2, 1], axis=1)
ops.repeat_interleave(x, (1, 2, 1), axis=1)
```

This isolates the issue to Graph-mode handling of a 0-D/scalar Tensor
`repeats` argument.

## Why this is a real issue

This is not a precision issue. The reproducer uses exact int32 data and Graph
mode fails before returning a tensor.

This is not a CPU unsupported issue. CPU is documented for
`ops.repeat_interleave`, PyNative succeeds on CPU with scalar Tensor repeats,
and Graph mode succeeds on CPU with the int and Tensor-vector controls.

This is not caused by an invalid repeat count. The repeat count is positive and
the same value succeeds as a Python int and as a 1-D Tensor with one element.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T214316Z-p1483705`
- Exploratory run: `artifacts/behavioral/20260801T214106Z-p1481835`
- Triggering cases: `TRI-001`, `TRI-002`, `TARG3-004`
- Passing controls: `TRI-003`, `TRI-004`, `TRI-005`, `TARG3-003`,
  `TARG3-005`, `TARG3-006`
- Minimal script: `findings/MS-035/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
