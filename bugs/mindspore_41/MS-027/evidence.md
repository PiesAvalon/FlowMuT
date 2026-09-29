# MS-027 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`6`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_keywords`
- `python reproducer.py --mode graph --variant tensor_mixed`

代表性控制命令：

- `python reproducer.py --mode graph --variant tensor_positional`
- `python reproducer.py --mode graph --variant ops_keywords`
- `python reproducer.py --mode pynative --variant tensor_keywords`

复验说明：底层算子、输入和位置参数均由控制验证，失败局限于 Graph Tensor 方法关键字绑定。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.col2im(...)` argument binding in Graph mode.
- Transformation (`m`): call `Tensor.col2im` with documented keyword
  arguments instead of positional arguments.
- Oracle (`o`): the Tensor method's Python signature exposes
  `output_size`, `kernel_size`, `dilation`, `padding_value` and `stride`; the
  keyword call should match the positional call and the direct `ops.col2im`
  keyword control.
- Localized boundary: Graph method wrapper / keyword binding for
  `Tensor.col2im`.

## Observed behavior

Input:

```text
x.shape == (1, 4, 9)
x4d = x.expand_dims(1)
output_size = Tensor([4, 4], int32)
kernel_size = [2, 2]
dilation = [1, 1]
padding_value = [0, 0]
stride = [1, 1]
```

Passing controls:

```text
x4d.col2im(output_size, kernel_size, dilation, padding_value, stride)
PyNative -> Tensor(shape=(1, 1, 4, 4), dtype=float32)
Graph    -> Tensor(shape=(1, 1, 4, 4), dtype=float32)

ops.col2im(input_x=x4d, output_size=output_size, kernel_size=kernel_size,
           dilation=dilation, padding_value=padding_value, stride=stride)
PyNative -> Tensor(shape=(1, 1, 4, 4), dtype=float32)
Graph    -> Tensor(shape=(1, 1, 4, 4), dtype=float32)
```

Failing cases:

```text
x4d.col2im(output_size=output_size, kernel_size=kernel_size,
           dilation=dilation, padding_value=padding_value, stride=stride)
PyNative -> succeeds and matches positional Tensor.col2im
Graph    -> RuntimeError: Got an unexpected keyword argument 'output_size'

x4d.col2im(output_size, kernel_size=kernel_size, dilation=dilation,
           padding_value=padding_value, stride=stride)
PyNative -> succeeds and matches positional Tensor.col2im
Graph    -> RuntimeError: Got an unexpected keyword argument 'kernel_size'
```

## Why this is a real issue

This is not a numerical precision issue. The Graph failures occur at argument
binding before returning a tensor.

This is not an invalid input or CPU unsupported issue. The positional
`Tensor.col2im` call succeeds in both modes, and the direct `ops.col2im`
keyword call succeeds in both modes with the same input.

This is not caused by missing keyword support in the underlying function:
`ops.col2im` has the same public keyword names and accepts them in Graph mode.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T175105Z-p1287732`
- Repetition runs:
  - `artifacts/behavioral/20260801T175228Z-p1288833`
  - `artifacts/behavioral/20260801T175339Z-p1289826`
- Exploratory source run: `artifacts/behavioral/20260801T174432Z-p1282507`
- Triggering cases: `TCOL2-002`, `TCOL2-003`
- Passing controls: `TCOL2-001`, `TCOL2-004`
- Minimal script: `findings/MS-027/reproducer.py`
- Repetition summary: `findings/MS-027/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
