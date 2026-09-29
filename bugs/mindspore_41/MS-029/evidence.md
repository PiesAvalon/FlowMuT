# MS-029 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`6`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant ops_csr_mul_to_tuple`
- `python reproducer.py --mode graph --variant tensor_mul_to_tuple`

代表性控制命令：

- `python reproducer.py --mode pynative --variant ops_csr_mul_to_tuple`
- `python reproducer.py --mode graph --variant csr_dense_control`
- `python reproducer.py --mode graph --variant csr_relu_control`

复验说明：相同 CSR 元数据在 PyNative 和 Graph CSR 基础控制中有效，失败定位到 CSRMul CPU kernel metadata。
## FlowMuT candidate

- Mutation site (`tau`): CSR sparse multiplication with a dense Tensor
  broadcast operand.
- Transformation (`m`): compare `ops.csr_mul(csr, dense)` and the equivalent
  `csr * dense` operator path across PyNative and Graph modes.
- Oracle (`o`): `csr_mul` accepts `CSRTensor` and `Tensor`, returns a
  `CSRTensor`, and lists CPU support in the local MindSpore 2.9 docstring.
- Localized boundary: Graph-mode CPU kernel metadata selection for `CSRMul`.

## Observed behavior

Input is fully static:

```text
x = Tensor([[0., 2., 0.],
            [3., 0., 4.]], float32)
csr = x.to_csr()
dense = ops.ones_like(x)
```

Failing cases:

```text
ops.csr_mul(csr, dense).to_tuple()
PyNative -> (indptr=[0, 1, 3], indices=[1, 0, 2], values=[2., 3., 4.], shape=(2, 3))
Graph    -> RuntimeError: Parsed metadata of op[CSRMul] failed.

(csr * dense).to_tuple()
PyNative -> (indptr=[0, 1, 3], indices=[1, 0, 2], values=[2., 3., 4.], shape=(2, 3))
Graph    -> RuntimeError: Parsed metadata of op[CSRMul] failed.
```

Graph stderr includes the more specific metadata warning:

```text
ParseMetadata] The size of inputs in OpIOInfo should be great than real input.
Inputs size in OpIOInfo:4, real input num: 6, node: Default/CSRMul-op0
```

Passing controls:

```text
ops.csr_relu(csr).to_tuple()
PyNative -> same CSR tuple structure
Graph    -> same CSR tuple structure

csr.to_dense()
PyNative -> Tensor(shape=(2, 3), dtype=float32)
Graph    -> Tensor(shape=(2, 3), dtype=float32)
```

## Why this is a real issue

This is not a numerical precision issue. Graph mode raises before a result is
returned.

This is not a general sparse/CPU unsupported issue. The CSR input conversion,
CSR unary operation, and CSR dense conversion controls all pass in Graph mode on
CPU. The failure is specific to `CSRMul` with a dense Tensor operand.

This is not a dependency/setup issue. An earlier exploratory run exposed a
missing optional `decorator` package on the PyNative AKG path; after installing
it, PyNative succeeds and the Graph `CSRMul` metadata failure remains stable.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T201940Z-p1408183`
- Repetition runs:
  - `artifacts/behavioral/20260801T202133Z-p1409648`
  - `artifacts/behavioral/20260801T202133Z-p1409654`
- Exploratory confirmation after dependency fix:
  - `artifacts/behavioral/20260801T201845Z-p1407449`
- Triggering cases: `TSPB-001`, `TSPB-002`
- Passing controls: `TSPB-004`, `TSPB-005`
- Minimal script: `findings/MS-029/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
