# MS-039 Evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`10`
- 退出码 0：`4`；非零：`6`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant primitive_dims_list`
- `python reproducer.py --mode graph --variant primitive_dims_tuple`
- `python reproducer.py --mode pynative --variant primitive_default_dims`

代表性控制命令：

- `python reproducer.py --mode pynative --variant functional_axis_list_control`
- `python reproducer.py --mode graph --variant functional_axis_list_control`
- `python reproducer.py --mode graph --variant functional_default_control`

复验说明：相同 axis 由功能式 CPU 路径成功验证，失败源于 constructor attribute 与生成 primitive call-time input 的冲突。
## Source alerts

Run:

```text
artifacts/behavioral/20260802T051055Z-p1674669
```

Case:

```text
TPRIM2-008 CountNonZero primitive list dims matches tuple dims
```

Observed:

```text
PyNative: RuntimeError: For Operator[CountNonZero], the inputs number should be 2 but got 1.
Graph:    RuntimeError: Unsupported op [CountNonZero] on CPU
```

The Graph failure shows the lowered node as:

```text
PrimFunc_CountNonZero, param_x, None
```

which indicates the documented constructor `dims=[1]` was not passed into the
active CountNonZero operator path.

## Controls

Run:

```text
artifacts/behavioral/20260802T051055Z-p1674675
```

Case:

```text
TPRIM2-016 count_nonzero functional axis list control succeeds
```

Observed in both PyNative and Graph mode:

```text
Tensor(shape=[2], dtype=Int32, value=[2, 1])
```

The larger control run also kept adjacent Primitive probes green:

```text
artifacts/behavioral/20260802T051056Z-p1674697
```

`AdaptiveAvgPool2D`, `AdaptiveAvgPool3D`, `AdaptiveMaxPool3D`,
`ScalarToTensor`, `TupleToArray`, and the functional `count_nonzero` control
all passed in both modes.

## Local source/doc evidence

Installed 2.9 source in `operations/array_ops.py` still documents and defines:

```text
class CountNonZero(Primitive)
__init__(self, dims=None)
init_prim_io_names(inputs=['x'], outputs=['y'])
```

The same docstring includes:

```python
countnonzero = ops.CountNonZero(dims=[1])
y = countnonzero(x)
```

Installed 2.9 generated source in `auto_generate/gen_ops_prim.py` defines a
different active call shape:

```text
prim = ops.CountNonZero()
out = prim(input, dim)
__call__(self, input, dim=None)
```

This explains the observed one-input failure for the documented constructor
form.
