# MS-008 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`6`
- 退出码 0：`4`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant statement`
- `python reproducer.py --mode pynative --variant statement`

代表性控制命令：

- `python reproducer.py --mode graph --variant return_value`
- `python reproducer.py --mode graph --variant functional`

复验说明：问题被严格限定为 Graph 语句形式的静默 no-op；不把 CPU 原地操作未支持本身误报为 bug。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.masked_fill_`, the in-place form of
  `Tensor.masked_fill`.
- Transformation (`m`): compare the same masked fill as a Graph statement,
  Graph return-value expression, PyNative expression, and functional
  `Tensor.masked_fill`.
- Oracle (`o`): a statement-form in-place operation should either mutate the
  target consistently or fail clearly. It should not compile and return an
  unchanged tensor while the return-value form performs the fill.
- Localized boundary: in-place Tensor method lowering/materialization for
  `masked_fill_`.

Observed with MindSpore 2.9.0 CPU:

- `y.masked_fill(mask, value)` works in both modes and returns:

```text
[[9. 4. 9.]
 [3. 9. 5.]]
```

- In Graph mode, `return y.masked_fill_(mask, value)` also returns the filled
  tensor.
- In Graph mode, using the same call as a statement and returning `y` silently
  returns the original tensor:

```text
[[1. 4. 2.]
 [3. 0. 5.]]
```

- In PyNative mode, materializing the result of `masked_fill_` fails with:

```text
RuntimeError: The kernel InplaceMaskedFillTensor unregistered.
```

The MindSpore 2.9.0 Tensor page notes that CPU/GPU do not support in-place
operations yet, so this is not being counted as "CPU must support in-place
masked_fill_". The reportable issue is narrower: Graph mode silently accepts a
statement-form unsupported/in-place call and leaves the tensor unchanged, while
the return-value lowering produces the filled tensor and PyNative fails with an
explicit kernel error. This is a semantic/termination issue, not numerical
precision.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T212601Z-p343003`
- Triggering case: `TINP-001`
- Minimal script: `findings/MS-008/reproducer.py`
- Repetition summary: `findings/MS-008/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
in `metadata.json`.
