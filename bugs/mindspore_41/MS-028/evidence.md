# MS-028 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`8`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_ops_shape_full`
- `python reproducer.py --mode graph --variant tensor_ops_shape_reduced`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_ops_shape_full`
- `python reproducer.py --mode graph --variant tensor_shape_property_full`
- `python reproducer.py --mode graph --variant ops_ops_shape_full`

复验说明：这是退出码为 0 的元数据结构错误，定位到 Graph Tensor.svd wrapper 输出与 ops.shape 的组合。
## FlowMuT candidate

- Mutation site (`tau`): `ops.shape(...)` applied to tensors returned by
  `Tensor.svd(...)` inside Graph mode.
- Transformation (`m`): compare Tensor-method `svd` shape metadata with the
  functional `ops.svd` control and with the Tensor outputs' `.shape`
  properties.
- Oracle (`o`): `Tensor.svd` documents the same output contract as
  `ops.svd`, and `ops.shape` documents a tuple shape metadata return. Shape
  metadata should not change from tuple metadata to scalar Tensor values only
  because the SVD call used the Tensor method wrapper.
- Localized boundary: Graph-mode metadata handling for tensors produced by the
  `Tensor.svd` method wrapper.

## Observed behavior

Input is a static CPU tensor:

```text
x = Tensor([[1, 2], [3, 4], [5, 7]], float32)
```

Failing cases:

```text
s, u, v = x.svd(full_matrices=True, compute_uv=True)
return ops.shape(s), ops.shape(u), ops.shape(v)

PyNative -> ((2,), (3, 3), (2, 2))
Graph    -> (Tensor(2, int64), Tensor(3, int64), Tensor(2, int64))

s, u, v = x.svd(full_matrices=False, compute_uv=True)
return ops.shape(s), ops.shape(u), ops.shape(v)

PyNative -> ((2,), (3, 2), (2, 2))
Graph    -> (Tensor(2, int64), Tensor(3, int64), Tensor(2, int64))
```

Passing controls:

```text
s, u, v = x.svd(full_matrices=True, compute_uv=True)
return s.shape, u.shape, v.shape

PyNative -> ((2,), (3, 3), (2, 2))
Graph    -> ((2,), (3, 3), (2, 2))

s, u, v = ops.svd(x, full_matrices=True, compute_uv=True)
return ops.shape(s), ops.shape(u), ops.shape(v)

PyNative -> ((2,), (3, 3), (2, 2))
Graph    -> ((2,), (3, 3), (2, 2))
```

The SVD tensors are produced successfully. The mismatch is the
type/structure returned by `ops.shape` for the Tensor-method outputs in Graph
mode.

## Why this is a real issue

This is not a numerical precision issue. The reproducer only checks shape
metadata and does not compare singular vector or singular value contents.

This is not a CPU unsupported issue. The direct `ops.svd` control works on CPU
in both modes, and the Tensor method's `.shape` properties expose the expected
metadata in both modes.

This is not a keyword-binding issue. Both keyword and positional Tensor-method
calls reproduce the same Graph `ops.shape` metadata mismatch.

This is not the same root as the older transient SVD value alerts. The current
issue is stable over repeated runs and isolated to `ops.shape` applied to
`Tensor.svd` outputs; the `.shape` property and functional `ops.svd` controls
pass.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T183200Z-p1321296`
- Repetition runs:
  - `artifacts/behavioral/20260801T182908Z-p1317748`
  - `artifacts/behavioral/20260801T182908Z-p1317757`
  - `artifacts/behavioral/20260801T182908Z-p1317763`
- Triggering cases: `TSVD-001`, `TSVD-002`, `TSVD-003`
- Passing controls: `TSVD-004`, `TSVD-005`
- Exploratory candidate run: `artifacts/behavioral/20260801T182138Z-p1311294`
- Minimal script: `findings/MS-028/reproducer.py`
- Repetition summary: `findings/MS-028/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
