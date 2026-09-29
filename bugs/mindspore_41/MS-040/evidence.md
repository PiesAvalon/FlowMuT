# MS-040 Evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`18`
- 退出码 0：`8`；非零：`10`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant functional_scalar_abs`
- `python reproducer.py --mode graph --variant functional_scalar_angle`
- `python reproducer.py --mode graph --variant primitive_scalar_abs`

代表性控制命令：

- `python reproducer.py --mode graph --variant functional_tensor_abs_control`
- `python reproducer.py --mode graph --variant functional_tensor_angle_control`
- `python reproducer.py --mode graph --variant functional_scalar_tensor_pair_control`
- `python reproducer.py --mode graph --variant primitive_tensor_angle_control`

复验说明：CPU complex 结果与 Polar primitive 由 Tensor 控制验证，失败仅是 Python float 处理。
## Source alerts

Run:

```text
artifacts/behavioral/20260802T052041Z-p1682614
```

Cases:

```text
TPRIMSEQ-006 Polar primitive scalar angle matches Tensor angle
TPRIMSEQ-007 Polar primitive scalar abs matches Tensor abs
TPRIMSEQ-008 ops.polar scalar angle matches Tensor angle
TPRIMSEQ-009 ops.polar scalar abs matches Tensor abs
```

Observed in both PyNative and Graph mode:

```text
ops.Polar()(x, 2.0)
PyNative: TypeError: valid calling should be Polar()(abs=<Tensor>, angle=<Tensor>)
Graph:    TypeError: Failed calling Polar with "angle=Float32".

ops.Polar()(2.0, x)
PyNative: TypeError: valid calling should be Polar()(abs=<Tensor>, angle=<Tensor>)
Graph:    TypeError: Failed calling Polar with "abs=Float32".
```

The same errors are raised by `ops.polar(x, 2.0)` and `ops.polar(2.0, x)`.

## Controls

Run:

```text
artifacts/behavioral/20260802T052605Z-p1686659
```

All controls passed in both PyNative and Graph mode:

```text
TPRIMSEQ-010 ops.Polar()(x, ops.ones_like(x) * 2.0) -> tensor(3,):complex64
TPRIMSEQ-011 ops.polar(x, ops.ones_like(x) * 2.0) -> tensor(3,):complex64
TPRIMSEQ-012 ops.polar(ops.ones_like(x) * 2.0, x) -> tensor(3,):complex64
TPRIMSEQ-013 ops.polar(Tensor(2.0), Tensor(2.0)) -> tensor():complex64
```

This localizes the failure to Python float argument handling rather than the
Polar kernel or complex output path.

## Discarded control attempt

Run:

```text
artifacts/behavioral/20260802T052430Z-p1685540
```

The first control attempt used a 0-D Tensor with a vector Tensor, for example
`ops.polar(x, Tensor(2.0))`. That correctly failed with a same-shape Tensor
requirement and is not part of the issue.
