# MS-025 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`24`
- 退出码 0：`14`；非零：`10`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --op min --variant tensor_i32`
- `python reproducer.py --mode graph --op max --variant tensor_i64`
- `python reproducer.py --mode pynative --op prod --variant tensor_i32`

代表性控制命令：

- `python reproducer.py --mode pynative --op min --variant int`
- `python reproducer.py --mode graph --op prod --variant tensor_i32`
- `python reproducer.py --mode pynative --op sum --variant tensor_i32`

复验说明：已按实际模式差异收窄：不再声称 prod 在 Graph 失败；sum 与 int 控制排除一般 CPU/输入问题。
## FlowMuT candidate

- Mutation site (`tau`): `ops.unsorted_segment_min/max/prod`
  `num_segments` argument handling.
- Transformation (`m`): pass `num_segments=ms.Tensor(2, ms.int32)`
  instead of the Python integer `2`, while keeping the same input tensor,
  `segment_ids`, device target and execution mode.
- Oracle (`o`): the public API should accept the documented
  `Union[int, Tensor]` form for `num_segments`, or document that these
  operators do not support Tensor-valued `num_segments`.
- Localized boundary: Python primitive validators and C++ infer logic for
  unsorted segment arithmetic.

## Observed behavior

Input:

```text
x = Tensor([[1, 2, 3], [4, 5, 6], [4, 2, 1]], int32)
segment_ids_min_max = Tensor([0, 1, 1], int32)
segment_ids_prod = Tensor([0, 1, 0], int32)
num_segments = Tensor(2, int32)
```

Passing controls:

```text
ops.unsorted_segment_sum(x, segment_ids_min_max, Tensor(2, int32))
PyNative -> Tensor([[1, 2, 3], [8, 7, 7]], int32)
Graph    -> Tensor([[1, 2, 3], [8, 7, 7]], int32)

ops.unsorted_segment_min/max/prod(..., 2)
PyNative -> succeeds
Graph    -> succeeds
```

Failing cases:

```text
ops.unsorted_segment_min(x, segment_ids_min_max, Tensor(2, int32))
PyNative -> ValueError: num_segments value must be greater than 0, but got: -1
Graph    -> TypeError: num_segments must be Number, but got Tensor[Int32]

ops.unsorted_segment_max(x, segment_ids_min_max, Tensor(2, int32))
PyNative -> ValueError: num_segments value must be greater than 0, but got: -1
Graph    -> TypeError: num_segments must be Number, but got Tensor[Int32]

ops.unsorted_segment_prod(x, segment_ids_prod, Tensor(2, int32))
PyNative -> ValueError: num_segments value must be greater than 0, but got: -1
Graph    -> succeeds and matches the Python-int control
```

## Why this is a real issue

This is not a precision issue. The failures occur before returning a numeric
result.

This is not a general CPU unsupported issue. The same operators succeed with
Python-int `num_segments`, and `ops.unsorted_segment_sum` succeeds with a
0-D Tensor `num_segments` on CPU.

This is not an invalid input shape issue. The Tensor input is scalar
(`shape == ()`) and has a positive value.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T163222Z-p1205982`
- Repetition runs:
  - `artifacts/behavioral/20260801T163714Z-p1209581`
  - `artifacts/behavioral/20260801T163827Z-p1210522`
- Triggering cases: `TSEGX-002`, `TSEGX-003`, `TSEGX-004`
- Passing control: `TSEGX-001`
- Minimal script: `findings/MS-025/reproducer.py`
- Repetition summary: `findings/MS-025/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
