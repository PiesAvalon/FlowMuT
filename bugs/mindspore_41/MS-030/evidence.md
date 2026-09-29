# MS-030 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`6`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant ops_csr_div_return`
- `python reproducer.py --mode graph --variant tensor_div_to_tuple`

代表性控制命令：

- `python reproducer.py --mode pynative --variant ops_csr_div_return`
- `python reproducer.py --mode graph --variant csr_div_dtype_control`
- `python reproducer.py --mode graph --variant csr_relu_control`

复验说明：PyNative 可执行两条入口，Graph 的其他 CSR 控制正常，失败局限于 CSRDiv 结果展开。
## FlowMuT candidate

- Mutation site (`tau`): CSR sparse division with a dense Tensor broadcast
  operand.
- Transformation (`m`): compare `ops.csr_div(csr, dense)` and the documented
  `csr / dense` operator path across PyNative and Graph modes.
- Oracle (`o`): the operation is accepted in PyNative mode, the local docstring
  lists CPU support, and Graph mode can infer the result dtype.
- Localized boundary: Graph-mode return/unfold handling for the `CSRDiv` result.

## Observed behavior

Input is fully static:

```text
x = Tensor([[0., 2., 0.],
            [3., 0., 4.]], float32)
csr = x.to_csr()
dense = ops.ones_like(x)
```

Failing function path:

```text
ops.csr_div(csr, dense)
PyNative -> CSRTensor(shape=(2, 3), values=[2., 3., 4.])
Graph    -> RuntimeError: Tuple to TupleUnfold pattern should have TupleGetItem as user node, but got Default/CSRDiv-op0
```

Failing operator path:

```text
(csr / dense).to_tuple()
PyNative -> (indptr=[0, 1, 3], indices=[1, 0, 2], values=[2., 3., 4.], shape=(2, 3))
Graph    -> RuntimeError: Tuple to TupleUnfold pattern should have TupleGetItem as user node, but got Default/CSRDiv-op0
```

Passing controls:

```text
ops.csr_div(csr, dense).dtype
PyNative -> Float32
Graph    -> Float32

ops.csr_relu(csr).to_tuple()
PyNative -> same CSR tuple structure
Graph    -> same CSR tuple structure
```

## Why this is a real issue

This is not a numerical precision issue. Graph mode raises while compiling or
returning the `CSRDiv` result.

This is not a general sparse/CPU unsupported issue. The same input converts to
CSR, the CSR unary control passes, and Graph mode can evaluate the `CSRDiv`
result dtype.

This is not only a documentation mismatch. The local `csr_div` docstring has a
known return-type contradiction for non-scalar dense operands: it says dense
Tensor/non-zero values, but the implementation returns a `CSRTensor`. Either
way, Graph mode should not fail with an internal TupleUnfold error for an input
that PyNative accepts, and the explicitly suggested `/` operator path fails with
the same `CSRDiv-op0` error.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T202303Z-p1411696`
- Repetition runs:
  - `artifacts/behavioral/20260801T202602Z-p1414014`
  - `artifacts/behavioral/20260801T202602Z-p1414023`
- Additional operator-path confirmation:
  - `artifacts/behavioral/20260801T202853Z-p1416934`
- Triggering cases: `TSPD-001`, `TSPD-005`
- Passing controls: `TSPD-002`, `TSPD-004`
- Minimal script: `findings/MS-030/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
