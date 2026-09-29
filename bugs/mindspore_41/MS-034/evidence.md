# MS-034 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`10`
- 退出码 0：`8`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant python_then_tensor_while`
- `python reproducer.py --mode graph --variant tensor_then_python_while`

代表性控制命令：

- `python reproducer.py --mode pynative --variant python_then_tensor_while`
- `python reproducer.py --mode graph --variant pure_python_while_control`
- `python reproducer.py --mode graph --variant pure_tensor_while_control`
- `python reproducer.py --mode graph --variant mixed_if_control`

复验说明：失败只出现在 mixed-bool while 组合，不是 Tensor bool、Python bool 或一般控制流不支持。
## FlowMuT candidate

- Mutation site (`tau`): Graph-mode `while` conditions that combine a Python
  scalar boolean guard with a scalar Tensor boolean predicate.
- Transformation (`m`): compare PyNative and Graph behavior for:
  `while i < 1 and out.sum() > 0`.
- Oracle (`o`): Graph mode already accepts pure Python `while`, scalar Tensor
  `while`, and mixed Python/Tensor `if` controls on the same input.
- Localized boundary: Graph static-analysis type joining for `and` inside a
  `while` condition.

## Observed behavior

Input is fully static:

```text
x = Tensor([[1., 4., 2.],
            [3., 0., 5.]], float32)
```

Failing cases:

```text
i = 0
out = x[0]
while i < 1 and out.sum() > 0:
    out = out + 1
    i = i + 1
return out

PyNative -> Tensor([2., 5., 3.], float32)
Graph    -> TypeError: Cannot join the return values of different branches
```

The reverse short-circuit order fails as well:

```text
while out.sum() > 0 and i < 1:
    ...

PyNative -> Tensor([2., 5., 3.], float32)
Graph    -> TypeError: Cannot join the return values of different branches
```

Representative Graph error:

```text
Type Join Failed: Abstract type AbstractTensor cannot join with AbstractScalar.
```

The reverse-order confirmation reports the same underlying mismatch:

```text
Type Join Failed: dtype1 = Bool, dtype2 = Any.
```

## Passing controls

All of these pass in both modes:

```text
while i < 1:
    ...

while out.sum() > 0 and i < limit:
    ...

if 0 < 1 and x[0].sum() > 0:
    ...

if x[0].sum() > 0:
    ...
```

This excludes a general CPU backend failure, a general Tensor-bool control-flow
failure, and a general Python `and` failure.

## Why this is a real issue

This is not a numerical precision issue. The Graph path fails during
compilation before producing an output.

This is not an unsupported CPU kernel issue. The failing operation is Graph
control-flow analysis, and the neighboring controls compile and execute on CPU.

This is not caused by dynamic Tensor getitem in the loop body. `TBWH-003` fails
even though the loop body only performs `out = out + 1`, and `TBWH-004` fails
with only a constant `x[0]` getitem.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T205206Z-p1439079`
- Repetition runs:
  - `artifacts/behavioral/20260801T184630Z-p1333036`
  - `artifacts/behavioral/20260801T184809Z-p1334445`
  - `artifacts/behavioral/20260801T184948Z-p1333030`
- Additional mixed-order run: `artifacts/behavioral/20260801T211505Z-p1459974`
- Triggering cases: `TBWH-001`, `TBWH-003`, `TBWH-004`, `TBMIX-002`
- Passing controls: `TBWH-002`, `TBWH-005`, `TBWH-006`, `TBWHV-001` to
  `TBWHV-005`, `TBMIX-001`, `TBMIX-003`
- Minimal script: `findings/MS-034/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
