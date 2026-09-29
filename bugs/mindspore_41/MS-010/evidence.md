# MS-010 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`8`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_default`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_default`
- `python reproducer.py --mode graph --variant tensor_dim_none`
- `python reproducer.py --mode graph --variant ops_default`
- `python reproducer.py --mode graph --variant tensor_axis_minus1`

复验说明：这是静默的值与形状错误，进程退出码为 0；显式 None 和功能式控制给出正确标量。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.argmax()` no-argument overload.
- Transformation (`m`): compare no-argument Tensor method, explicit
  `dim=None`, explicit `axis=-1`, and functional `ops.argmax`.
- Oracle (`o`): the documented no-argument/default behavior should compute the
  argmax over all elements and return a scalar index.
- Localized boundary: Graph lowering of the Tensor method no-argument overload.

Observed with MindSpore 2.9.0 CPU on:

```text
[[3, 1, 2],
 [6, 5, 4]]
```

- PyNative `x.argmax()` returns scalar `3`, the flattened index of value `6`.
- Graph `x.argmax()` returns vector `[0, 0]`, equivalent to `x.argmax(axis=-1)`.
- Graph `x.argmax(dim=None)` returns scalar `3`.
- Graph `ops.argmax(x)` returns scalar `3`.

The MindSpore 2.9.0 documentation says that `axis=None` computes all elements in
the tensor, and its example shows no-argument `x.argmax()` returning a scalar.
This issue is therefore a Graph-only Tensor method overload/default dispatch
bug, not numerical precision.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T212956Z-p346202`
- Triggering case: `TDEF-002`
- Minimal script: `findings/MS-010/reproducer.py`
- Repetition summary: `findings/MS-010/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
in `metadata.json`.
