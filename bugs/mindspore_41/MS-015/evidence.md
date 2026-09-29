# MS-015 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`12`
- 退出码 0：`11`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_scatter_div`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_scatter_div`
- `python reproducer.py --mode graph --variant ops_tensor_scatter_div`
- `python reproducer.py --mode graph --variant tensor_scatter_mul`

复验说明：异常直接暴露注册名拼写错误，且同一底层算子的功能式入口成功。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.scatter_div(indices, updates)`.
- Transformation (`m`): compare the Tensor method with `ops.tensor_scatter_div`
  and sibling Tensor methods `scatter_sub`, `scatter_mul`, `scatter_max`, and
  `scatter_min`.
- Oracle (`o`): the Tensor method should compile and run in Graph mode when the
  underlying `ops.tensor_scatter_div` operation compiles and runs on the same
  input.
- Localized boundary: Graph `standard_method` registration for
  `tensor_scatter_div`.

Input:

```text
x =
[[1. 4. 2.]
 [3. 2. 5.]]

indices = [[0, 1],
           [1, 2]]
updates = [2., 4.]
```

Expected tensor-scatter division result:

```text
[[1.   2.   2.  ]
 [3.   2.   1.25]]
```

## Observed behavior

PyNative `Tensor.scatter_div` succeeds and matches `ops.tensor_scatter_div`:

```bash
python findings/MS-015/reproducer.py --mode pynative --variant tensor_scatter_div
python findings/MS-015/reproducer.py --mode pynative --variant ops_tensor_scatter_div
```

Graph `ops.tensor_scatter_div` succeeds:

```bash
python findings/MS-015/reproducer.py --mode graph --variant ops_tensor_scatter_div
```

Graph `Tensor.scatter_div` fails:

```bash
python findings/MS-015/reproducer.py --mode graph --variant tensor_scatter_div
```

Failure:

```text
AttributeError: module 'mindspore.graph._parse.standard_method' has no attribute 'tensor_scatter_div'
```

The Python error hint also points at the typo:

```text
Did you mean: 'tensor_sactter_div'?
```

Sibling Tensor wrappers work in Graph mode on the same indices and updates:

```bash
python findings/MS-015/reproducer.py --mode graph --variant tensor_scatter_sub
python findings/MS-015/reproducer.py --mode graph --variant tensor_scatter_mul
python findings/MS-015/reproducer.py --mode graph --variant tensor_scatter_max
python findings/MS-015/reproducer.py --mode graph --variant tensor_scatter_min
```

## Localized source evidence

In MindSpore 2.9.0 local source, `Tensor.scatter_div` routes to
`tensor_scatter_div`:

```text
common/tensor.py:1296
return tensor_operator_registry.get('tensor_scatter_div')(self, indices, updates)
```

But Graph `standard_method.py` contains a misspelled definition:

```text
standard_method.py:3087
def tensor_sactter_div(input_x, indices, updates):
```

So Graph lookup for `tensor_scatter_div` fails, while the sibling definitions
`tensor_scatter_sub`, `tensor_scatter_mul`, `tensor_scatter_max`, and
`tensor_scatter_min` are present and work.

## Why this is a real issue

This is not a precision issue. The output values are exact simple divisions.

This is not a missing CPU kernel. `ops.tensor_scatter_div` compiles and runs in
Graph mode on CPU with the same input.

The failure is localized to the Tensor method Graph registration name:
`tensor_sactter_div` is misspelled, so the expected `tensor_scatter_div` symbol
does not exist in `standard_method`.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T233947Z-p479849`
- Triggering case: `TSCAT-023`
- Minimal script: `findings/MS-015/reproducer.py`
- Repetition summary: `findings/MS-015/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
