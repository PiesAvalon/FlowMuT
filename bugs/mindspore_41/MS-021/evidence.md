# MS-021 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`7`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant map_tensor`

代表性控制命令：

- `python reproducer.py --mode pynative --variant map_tensor`
- `python reproducer.py --mode graph --variant map_tuple_control`
- `python reproducer.py --mode graph --variant zip_tensor_control`
- `python reproducer.py --mode graph --variant sum_tensor_control`

复验说明：lambda 和 Tensor 行运算由 tuple-map 控制验证，Tensor 可迭代性由 zip/sum 控制验证。
## FlowMuT candidate

- Mutation site (`tau`): Python builtin `map` applied to Tensor iteration
  inside `construct`.
- Transformation (`m`): compare `map(lambda, x)` with `map(lambda, (x[0],
  x[1]))`, `zip(x, x + 1)`, and `sum(x)` controls.
- Oracle (`o`): if Graph mode supports Tensor iteration and supports `map`
  over a tuple of Tensor rows, `map` over the Tensor iterator should not be
  rejected simply because the sequence object is a Tensor.
- Localized boundary: Graph builtin `map` lowering accepts list/tuple
  non-leaf sequences but rejects Tensor even though other Graph Tensor
  iterator paths work.

## Observed behavior

Input:

```text
x = Tensor([[1, 4, 2], [3, 0, 5]], float32)
```

Failing case:

```text
tuple(map(lambda row: row + 1, x))

PyNative -> (Tensor([2, 5, 3]), Tensor([4, 1, 6]))
Graph    -> RuntimeError: Map can only be applied to list, tuple, but got Tensor[Float32].
```

Passing controls:

```text
tuple(map(lambda row: row + 1, (x[0], x[1])))
tuple(zip(x, x + 1))
sum(x)
len(x)
```

All of these controls match PyNative and Graph mode.

## Localized source evidence

MindSpore 2.9.0 maps Python builtin `map` to `C.Map()` in
`graph/_parse/resources.py`.

The local `Map` composite header initializes `nonleaf_` with only list and
tuple object types:

```text
include/mindspore/ccsrc/frontend/operator/composite/map.h

nonleaf_({kObjectTypeList, kObjectTypeTuple})
```

That matches the observed Graph error: Tensor is rejected as the sequence type
even though Tensor iteration is available in the surrounding Graph syntax.

## Why this is a real issue

This is not a precision issue. It is a compile-time termination mismatch.

This is not a missing CPU kernel. The failing path errors during Graph builtin
lowering, and the controls run on CPU.

This is not caused by the lambda body. The same lambda succeeds in Graph mode
when applied to `(x[0], x[1])`.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T090152Z-p873310`
- Repetition runs:
  - `artifacts/behavioral/20260801T090152Z-p873316`
  - `artifacts/behavioral/20260801T090400Z-p876265`
- Triggering case: `TPYB-004`
- Passing controls: `TPYB-001`, `TPYB-003`, `TPYB-005`, `TPYB-007`
- Minimal script: `findings/MS-021/reproducer.py`
- Repetition summary: `findings/MS-021/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
