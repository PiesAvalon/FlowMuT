# MS-031 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`10`
- 退出码 0：`8`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_mv`
- `python reproducer.py --mode graph --variant ops_csr_mv`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_mv`
- `python reproducer.py --mode graph --variant csr_tuple_control`
- `python reproducer.py --mode graph --variant csr_abs_control`
- `python reproducer.py --mode graph --variant csr_add_control`

复验说明：维度由 PyNative 结果验证，CSR 基础控制通过，失败定位到 CSRMV CPU kernel metadata。
## FlowMuT candidate

- Mutation site (`tau`): CSR sparse matrix-vector multiplication with a dense
  vector operand.
- Transformation (`m`): compare `x.to_csr().mv(dense_vector)` across PyNative
  and Graph modes.
- Oracle (`o`): `CSRTensor.mv`/`csr_mv` documents CPU support and returns a
  dense Tensor.
- Localized boundary: Graph-mode CPU kernel metadata selection for `CSRMV`.

## Observed behavior

Input is fully static:

```text
x = Tensor([[0., 2., 0.],
            [3., 0., 4.]], float32)
dense_vector = Tensor([[1.], [2.], [3.]], float32)
```

Failing case:

```text
x.to_csr().mv(dense_vector)
PyNative -> Tensor(shape=(2, 1), dtype=float32)
Graph    -> RuntimeError: Parsed metadata of op[CSRMV] failed.
```

Passing controls:

```text
x.to_csr().to_tuple()
x.to_csr().astype(ms.float64).to_tuple()
x.to_csr().abs().to_tuple()
(x.to_csr() + x.to_csr()).to_tuple()
```

Each control returns the expected CSR tuple structure in Graph mode.

## Why this is a real issue

This is not a numerical precision issue. Graph mode raises before returning a
result.

This is not a general CSR/CPU unsupported issue. Several CSR controls pass on
the same input in Graph mode on CPU, and `CSRTensor.mv`/`csr_mv` lists CPU as a
supported platform without the LLVM precondition found on some other sparse
methods.

This is not an invalid shape issue. The CSR input shape is `(2, 3)`, the dense
vector shape is `(3, 1)`, and PyNative returns a dense Tensor of shape `(2, 1)`.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T203408Z-p1421780`
- Repetition runs:
  - `artifacts/behavioral/20260801T203854Z-p1425588`
  - `artifacts/behavioral/20260801T203854Z-p1425594`
- Triggering cases: `TSP2-005`, `TSPC-002`
- Passing controls: `TSP2-001`, `TSP2-002`, `TSP2-003`, `TSP2-007`, `TSPC-005`, `TSPC-006`, `TSPC-007`
- Minimal script: `findings/MS-031/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
