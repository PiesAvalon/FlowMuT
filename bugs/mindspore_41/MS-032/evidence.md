# MS-032 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`7`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant csr_gather`

代表性控制命令：

- `python reproducer.py --mode pynative --variant csr_gather`
- `python reproducer.py --mode graph --variant csr_tuple_control`
- `python reproducer.py --mode graph --variant coo_concat_control`
- `python reproducer.py --mode graph --variant coo_add_control`

复验说明：输入由 PyNative 和 Graph CSR tuple 控制验证，失败局限于 Graph CSRGather CPU metadata。
## FlowMuT candidate

- Mutation site (`tau`): `ops.csr_gather` using CSR metadata from a valid
  dense-to-CSR conversion.
- Transformation (`m`): compare `ops.csr_gather(indptr, indices, dense, shape)`
  across PyNative and Graph modes.
- Oracle (`o`): `ops.csr_gather` documents CPU support and returns a dense
  Tensor.
- Localized boundary: Graph-mode CPU kernel metadata selection for `CSRGather`.

## Observed behavior

Input is fully static:

```text
x = Tensor([[0., 2., 0.],
            [3., 0., 4.]], float32)
csr = x.to_csr()
```

Failing case:

```text
ops.csr_gather(csr.indptr, csr.indices, x, x.shape)
PyNative -> Tensor(shape=(3,), dtype=float32)
Graph    -> RuntimeError: Parsed metadata of op[CSRGather] failed.
```

Passing controls:

```text
x.to_csr().to_tuple()
x.to_coo().to_tuple()
ops.coo_concat((coo_a, coo_b), 1).to_tuple()
ops.coo_add(coo_a, coo_b, thresh).to_tuple()
```

## Why this is a real issue

This is not a numerical precision issue. Graph mode raises before returning a
Tensor.

This is not an invalid sparse input issue. The CSR metadata is produced from a
valid dense Tensor, PyNative returns the expected non-zero gathered values, and
CSR/COO tuple controls pass in Graph mode.

This is not a CPU unsupported issue. The local `ops.csr_gather` docstring lists
CPU support.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T204331Z-p1430881`
- Repetition runs:
  - `artifacts/behavioral/20260801T204552Z-p1432792`
  - `artifacts/behavioral/20260801T204552Z-p1432798`
- Triggering case: `TSPN-001`
- Passing controls: `TSPN-003`, `TSPN-004`, `TSPN-005`, `TSPN-006`, `TSPN-007`
- Minimal script: `findings/MS-032/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
