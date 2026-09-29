# MS-014 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`6`
- 退出码 0：`4`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant expand_varargs`
- `python reproducer.py --mode graph --variant expand_tuple`

代表性控制命令：

- `python reproducer.py --mode pynative --variant expand_varargs`
- `python reproducer.py --mode pynative --variant expand_tuple`
- `python reproducer.py --mode graph --variant broadcast_to`

复验说明：相同数据和目标形状可由 PyNative expand 与 Graph broadcast_to 正确处理，失败局限于 Graph Tensor.expand 参数降低。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.expand`.
- Transformation (`m`): compare `x.expand(2, 3)` and `x.expand((2, 3))`
  with `x.broadcast_to((2, 3))` under PyNative and Graph mode.
- Oracle (`o`): `Tensor.expand` is documented locally as forwarding to
  `broadcast_to` with the same target shape, so it should compile and return
  the same broadcasted tensor as `broadcast_to`.
- Localized boundary: Graph parsing/lowering of `Tensor.expand` arguments.

Input:

```text
[[1. 2. 3.]]
```

Expected broadcast result:

```text
[[1. 2. 3.]
 [1. 2. 3.]]
```

## Observed behavior

PyNative succeeds for both supported call shapes:

```bash
python findings/MS-014/reproducer.py --mode pynative --variant expand_varargs
python findings/MS-014/reproducer.py --mode pynative --variant expand_tuple
```

Both return shape `(2, 3)` with the expected values.

Graph mode fails for `expand(2, 3)`:

```bash
python findings/MS-014/reproducer.py --mode graph --variant expand_varargs
```

Failure:

```text
TypeError: The parameters number of the function is 2, but the number of provided arguments is 3.
FunctionGraph : expand_2
standard_method.py:4204~4209
def expand(input, size):
```

Graph mode also fails for `expand((2, 3))`:

```bash
python findings/MS-014/reproducer.py --mode graph --variant expand_tuple
```

Failure:

```text
RuntimeError: The method 'GetShapeVector()' doesn't implement
standard_method.py:4208 -> size = TensorToTuple()(size)
```

The direct control API works in Graph mode:

```bash
python findings/MS-014/reproducer.py --mode graph --variant broadcast_to
```

It returns the expected `(2, 3)` tensor.

## Why this is a real issue

This is not a precision issue. The operation is exact broadcasting of a small
row tensor.

This is not a missing CPU broadcast kernel. `Tensor.broadcast_to((2, 3))`
compiles and runs in Graph mode on the same input.

The failure is localized to the `Tensor.expand` Graph wrapper: local Tensor
source defines `expand(self, *size)`, but Graph `standard_method.expand` accepts
only one `size` argument and then tries to convert the tuple shape through
`TensorToTuple`.

`expand(2, 3)` and `expand((2, 3))` are the same root cause, so this is counted
as one issue.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T234230Z-p484249`
- Triggering cases: `TSHP-015`, `TSHP-017`
- Minimal script: `findings/MS-014/reproducer.py`
- Repetition summary: `findings/MS-014/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
