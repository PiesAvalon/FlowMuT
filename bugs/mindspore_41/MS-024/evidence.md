# MS-024 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`4`
- 退出码 0：`2`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant int_k`
- `python reproducer.py --mode graph --variant int_k`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_k_control`
- `python reproducer.py --mode graph --variant tensor_k_control`

复验说明：相同矩阵与参数只有 Python int 形式失败，Tensor 形式在两模式都成功。
## FlowMuT candidate

- Mutation site (`tau`): `ops.matrix_diag_part` argument binding for `k`.
- Transformation (`m`): pass `k=0` as a Python `int` versus
  `ms.Tensor(0, ms.int32)` while keeping the same input and padding value.
- Oracle (`o`): the public API should accept the documented `Union[int,
  Tensor]` type for `k`, or document that only Tensor-valued `k` is accepted.
- Localized boundary: argument type conversion/checking before
  `MatrixDiagPartV3` execution.

## Observed behavior

Input:

```text
x = Tensor([[3, 1, 2], [6, 5, 4]], int32)
padding_value = Tensor(0, int32)
```

Failing case:

```text
ops.matrix_diag_part(x, 0, padding_value)
PyNative -> TypeError: input[1] should be a Tensor, but got Int64
Graph    -> TypeError: input[1] should be a Tensor, but got Int64
```

Passing control:

```text
ops.matrix_diag_part(x, Tensor(0, int32), padding_value)
PyNative -> Tensor([3, 5], int32)
Graph    -> Tensor([3, 5], int32)
```

Additional nonzero-offset confirmation:

```text
base = ops.matrix_diag(x[0], Tensor(1, int32), Tensor(3, int32),
                       Tensor(4, int32), Tensor(0, int32))

ops.matrix_diag_part(base, 1, Tensor(0, int32))
PyNative -> TypeError: input[1] should be a Tensor, but got Int64
Graph    -> TypeError: input[1] should be a Tensor, but got Int64

ops.matrix_diag_part(base, Tensor(1, int32), Tensor(0, int32))
PyNative -> Tensor([3, 1, 2], int32)
Graph    -> Tensor([3, 1, 2], int32)
```

## Why this is a real issue

This is not a precision issue. The call fails at argument type checking before
any numeric result is computed.

This is not a CPU kernel availability issue. The Tensor-valued `k` control
executes successfully on CPU in both PyNative and Graph mode.

The failure is specific to the documented Python-int `k` form.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T112640Z-p1041829`
- Repetition runs:
  - `artifacts/behavioral/20260801T112754Z-p1042663`
  - `artifacts/behavioral/20260801T112754Z-p1042675`
- Triggering case: `TARR-018`
- Passing control: `TARR-021`
- Minimal script: `findings/MS-024/reproducer.py`
- Repetition summary: `findings/MS-024/repetitions/summary.json`
- Additional confirmation runs:
  - `artifacts/behavioral/20260801T192928Z-p1366948`
  - `artifacts/behavioral/20260801T193048Z-p1367984`
  - `artifacts/behavioral/20260801T193048Z-p1367990`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
