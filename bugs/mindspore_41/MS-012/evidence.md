# MS-012 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`8`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant statement`
- `python reproducer.py --mode pynative --variant paired`
- `python reproducer.py --mode graph --variant paired`

代表性控制命令：

- `python reproducer.py --mode pynative --variant return_value`
- `python reproducer.py --mode graph --variant functional`

复验说明：问题限定为成功调用后的不一致语义，而不是把文档已说明的不支持状态本身当作 bug。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.add_`, the in-place form of `Tensor.add`.
- Transformation (`m`): compare statement-form, return-value-form, paired
  statement/return-value, and non-in-place `Tensor.add`.
- Oracle (`o`): an in-place operation should either mutate the target
  consistently or fail clearly when unsupported. It should not silently return
  contradictory successful results across call shapes and execution modes.
- Localized boundary: CPU in-place Tensor method lowering/materialization for
  `add_`.

Observed with MindSpore 2.9.0 CPU on:

```text
[[1. 4. 2.]
 [3. 0. 5.]]
```

The non-in-place control `Tensor.add(2)` succeeds in both PyNative and Graph
modes and returns:

```text
[[3. 6. 4.]
 [5. 2. 7.]]
```

`Tensor.add_` then shows inconsistent successful behavior:

- PyNative statement form:

```python
y = x + 0
y.add_(2)
return y
```

returns the original unmodified tensor:

```text
[[1. 4. 2.]
 [3. 0. 5.]]
```

- PyNative return-value form:

```python
y = x + 0
return y.add_(2)
```

returns the add result:

```text
[[3. 6. 4.]
 [5. 2. 7.]]
```

- Graph statement and return-value forms each return the add result when run
  alone.
- Graph paired form:

```python
y = x + 0
y.add_(2)
z = x + 0
r = z.add_(2)
return y, r
```

returns two tensors with the value of applying the update twice:

```text
[[5. 8. 6.]
 [7. 4. 9.]]
```

for both `y` and `r`.

This is not a numerical precision issue: all values are exact small float32
integers. The issue is control-flow/API semantics. The same `Tensor.add_` call
is a silent no-op in PyNative statement form, a normal add in PyNative
return-value form, and an apparent aliasing/double-update in Graph paired form.

## Documentation basis

- `mindspore.Tensor.add_` in the 2.9.0 docs is described as the in-place
  version of `mindspore.Tensor.add()`.
- The 2.9.0 `mindspore.Tensor` page also notes that all CPU/GPU modes do not
  support in-place operations yet.

Given that documented limitation, this report does not claim that CPU must
support `add_`. The reportable issue is narrower: CPU execution should not
silently succeed with mutually inconsistent semantics. A clear unsupported
operation error would be preferable to a PyNative no-op or a Graph double
update.

## Related FlowMuT cases

`TINP-003` through `TINP-007` show the same class of problem for `sub_`,
`mul_`, `div_`, `floor_divide_`, and `remainder_`. This issue uses `add_` as
the minimal representative reproducer.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T222657Z-p401704`
- Triggering case: `TINP-002`
- Minimal script: `findings/MS-012/reproducer.py`
- Repetition summary: `findings/MS-012/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
in `metadata.json`.
