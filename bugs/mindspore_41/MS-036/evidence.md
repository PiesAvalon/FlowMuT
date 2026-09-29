# MS-036 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`12`
- 退出码 0：`8`；非零：`4`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant primitive_int_value`
- `python reproducer.py --mode graph --variant primitive_float_value`

代表性控制命令：

- `python reproducer.py --mode graph --variant primitive_tensor_value_control`
- `python reproducer.py --mode graph --variant functional_int_value_control`
- `python reproducer.py --mode graph --variant tensor_method_int_value_control`

复验说明：CPU 与底层 IndexFill 可由 Tensor value 控制运行，失败仅是直接 primitive 的标量参数契约。
## FlowMuT candidate

- Mutation site (`tau`): direct `ops.IndexFill` primitive argument handling.
- Transformation (`m`): pass a documented scalar `value` (`9` or `9.0`)
  instead of an equivalent scalar Tensor.
- Oracle (`o`): `IndexFill` documents `value` as
  `Union[bool, int, float, Tensor]` and lists CPU support. The Tensor-valued
  control succeeds.

## Observed behavior

Input:

```text
x = Tensor([[3, 1, 2],
            [6, 5, 4]], int32)
index = Tensor([0, 2], int32)
```

Failing primitive call:

```text
ops.IndexFill()(x, 1, index, 9)

PyNative -> TypeError: For Primitive[IndexFill], the type of input argument[value] must be Tensor but got Int64.
Graph    -> TypeError: For Primitive[IndexFill], the type of input argument[value] must be Tensor but got Int64.
```

Float values show the same problem on float32 input:

```text
ops.IndexFill()(x_float32, 1, index, 9.0)

PyNative -> TypeError: ... value must be Tensor but got Float32.
Graph    -> TypeError: ... value must be Tensor but got Float32.
```

## Passing controls

These pass in both PyNative and Graph mode:

```text
ops.IndexFill()(x, 1, index, Tensor(9, int32))
ops.IndexFill()(x, Tensor(1, int32), index, Tensor(9, int32))
ops.index_fill(x, 1, index, 9)
x.index_fill(1, index, 9)
```

The functional and Tensor-method controls show that scalar `value` can be
handled correctly by the public wrapper layer. The direct primitive rejects the
same documented scalar value before execution.

## Why this is a real issue

This is not a precision issue. The reproducer uses exact integer replacement
and a fail-before-result TypeError.

This is not a CPU unsupported issue. CPU support is documented for the
primitive, Tensor-valued primitive controls pass on CPU, and scalar-valued
functional/Tensor wrappers pass on CPU.

This is not a Graph-only limitation. The same direct primitive call fails in
PyNative and Graph mode, while the wrapper controls succeed in both modes.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260802T043617Z-p1645097`
- Exploratory run: `artifacts/behavioral/20260802T042828Z-p1638588`
- Triggering cases: `TIFILL-001`, `TIFILL-002`, `TARG5-022`
- Passing controls: `TIFILL-003`, `TIFILL-004`, `TARG5-021`
- Minimal script: `findings/MS-036/reproducer.py`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
