# MS-037 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`12`
- 退出码 0：`8`；非零：`4`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant primitive_int_main_diag`
- `python reproducer.py --mode graph --variant primitive_int_offset_diag`

代表性控制命令：

- `python reproducer.py --mode graph --variant primitive_tensor_main_control`
- `python reproducer.py --mode graph --variant primitive_tensor_offset_control`
- `python reproducer.py --mode graph --variant functional_int_main_control`

复验说明：Tensor primitive 和功能式 int 控制均成功，失败局限于直接 MatrixDiagV3 的 Python 标量处理。
## FlowMuT candidate

- Mutation site (`tau`): direct `ops.MatrixDiagV3` primitive argument
  handling.
- Transformation (`m`): pass Python int arguments for `k`, `num_rows`,
  `num_cols` and `padding_value` instead of equivalent scalar Tensors.
- Oracle (`o`): `MatrixDiagV3` documents `k` as `Union[int, Tensor]`, documents
  scalar-compatible row/column/padding arguments, and lists CPU support.

## Observed behavior

```text
x = Tensor([1, 2, 3], int32)
```

Failing call:

```text
ops.MatrixDiagV3()(x, 0, 3, 3, 0)

PyNative -> TypeError: For primitive[MatrixDiagV3], the input[1] should be a Tensor, but got Int64.
Graph    -> TypeError: For primitive[MatrixDiagV3], the input[1] should be a Tensor, but got Int64.
```

The same happens for a nonzero diagonal offset:

```text
ops.MatrixDiagV3()(x, 1, 3, 4, 0)
```

## Passing controls

These pass in both PyNative and Graph mode:

```text
ops.MatrixDiagV3()(x, Tensor(0, int32), Tensor(3, int32), Tensor(3, int32), Tensor(0, int32))
ops.MatrixDiagV3()(x, Tensor(1, int32), Tensor(3, int32), Tensor(4, int32), Tensor(0, int32))
ops.matrix_diag(x, k=0, num_rows=3, num_cols=3, padding_value=0)
ops.matrix_diag(x, k=1, num_rows=3, num_cols=4, padding_value=0)
```

The passing wrapper controls show that the Python-int form is valid and can be
handled by the public API layer. The direct primitive rejects the documented
form before execution.

## Why this is a real issue

This is not a precision issue. The failure is a deterministic TypeError before
any tensor is returned.

This is not a CPU unsupported issue. CPU support is documented and the
Tensor-valued primitive controls pass on CPU.

This is related to MS-024 but not the same API surface. MS-024 covers
`ops.matrix_diag_part`; this issue covers direct `ops.MatrixDiagV3`.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260802T044122Z-p1648872`
- Exploratory run: `artifacts/behavioral/20260802T042828Z-p1638588`
- Triggering cases: `TPRIMSC-001`, `TPRIMSC-002`
- Passing controls: `TARG5-023`, `TARG5-024`
- Minimal script: `findings/MS-037/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
