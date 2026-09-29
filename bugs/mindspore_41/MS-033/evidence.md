# MS-033 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`7`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant csr_softmax`

代表性控制命令：

- `python reproducer.py --mode pynative --variant csr_softmax`
- `python reproducer.py --mode graph --variant csr_tuple_control`
- `python reproducer.py --mode graph --variant coo_concat_control`
- `python reproducer.py --mode graph --variant coo_add_control`

复验说明：PyNative 与 Graph 稀疏基础控制证明输入有效，错误指向 csr_softmax Python 实现中的 free variable 路径。
## FlowMuT candidate

- Mutation site (`tau`): `ops.csr_softmax` on a valid `CSRTensor`.
- Transformation (`m`): compare `ops.csr_softmax(csr, ms.float32).to_tuple()`
  across PyNative and Graph modes.
- Oracle (`o`): `ops.csr_softmax` documents CPU support and returns a
  `CSRTensor`.
- Localized boundary: Graph-mode compilation of the Python implementation path
  for `csr_softmax`.

## Observed behavior

Input is fully static:

```text
x = Tensor([[0., 2., 0.],
            [3., 0., 4.]], float32)
csr = x.to_csr()
```

Failing case:

```text
ops.csr_softmax(csr, ms.float32).to_tuple()
PyNative -> (indptr, indices, values, shape) CSR tuple
Graph    -> RuntimeError: The Map operator don't support Closure with free variable yet.
```

Passing controls:

```text
x.to_csr().to_tuple()
x.to_coo().to_tuple()
ops.coo_concat((coo_a, coo_b), 1).to_tuple()
ops.coo_add(coo_a, coo_b, thresh).to_tuple()
```

## Why this is a real issue

This is not a numerical precision issue. The reproducer does not compare
softmax values; Graph mode raises before returning.

This is not an invalid sparse input issue. The same `CSRTensor` works in
PyNative mode, and CSR tuple controls pass in Graph mode.

This is not a CPU unsupported issue. The local `ops.csr_softmax` docstring lists
CPU support. The error is an internal Graph compilation/runtime error rather
than a documented unsupported-platform rejection.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T204331Z-p1430881`
- Repetition runs:
  - `artifacts/behavioral/20260801T204552Z-p1432792`
  - `artifacts/behavioral/20260801T204552Z-p1432798`
- Triggering case: `TSPN-002`
- Passing controls: `TSPN-003`, `TSPN-004`, `TSPN-005`, `TSPN-006`, `TSPN-007`
- Minimal script: `findings/MS-033/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
