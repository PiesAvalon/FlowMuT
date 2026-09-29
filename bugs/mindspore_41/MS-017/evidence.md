# MS-017 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`28`
- 退出码 0：`22`；非零：`6`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_amax_initial`
- `python reproducer.py --mode graph --variant tensor_amin_where_initial`
- `python reproducer.py --mode graph --variant tensor_amax_keepdims`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_amax_initial`
- `python reproducer.py --mode graph --variant ops_amax_initial`
- `python reproducer.py --mode graph --variant ops_amin_where_initial`
- `python reproducer.py --mode graph --variant tensor_amax_plain`

复验说明：六个 Tensor 关键字目标均失败，对应 ops 控制和无关键字 Tensor 控制均成功。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.amax` and `Tensor.amin` keyword arguments,
  including `initial`, `where`, and `keepdims`.
- Transformation (`m`): compare Tensor methods against direct `ops.amax` and
  `ops.amin` controls with the same keyword arguments.
- Oracle (`o`): Tensor methods should accept the same documented forwarding
  keywords in Graph mode when PyNative succeeds and the direct ops controls
  compile on the same input.
- Localized boundary: Graph `standard_method` signatures for Tensor `amax` and
  `amin`.

Input:

```text
x =
[[3 1 2]
 [6 5 4]]
```

Expected results:

```text
x.amax(axis=1, initial=10) -> [10 10]
x.amin(axis=1, initial=0)  -> [0 0]
```

## Observed behavior

PyNative Tensor methods succeed and match the direct ops controls:

```bash
python findings/MS-017/reproducer.py --mode pynative --variant tensor_amax_initial
python findings/MS-017/reproducer.py --mode pynative --variant tensor_amin_initial
```

Graph direct ops controls also succeed:

```bash
python findings/MS-017/reproducer.py --mode graph --variant ops_amax_initial
python findings/MS-017/reproducer.py --mode graph --variant ops_amin_initial
```

Graph Tensor methods fail:

```bash
python findings/MS-017/reproducer.py --mode graph --variant tensor_amax_initial
python findings/MS-017/reproducer.py --mode graph --variant tensor_amin_initial
```

Failure:

```text
RuntimeError: Got an unexpected keyword argument 'initial'
mindspore/core/ir/func_graph_extends.cc:161 GenerateKwParams
```

Plain Graph Tensor reductions without `initial` still work:

```bash
python findings/MS-017/reproducer.py --mode graph --variant tensor_amax_plain
python findings/MS-017/reproducer.py --mode graph --variant tensor_amin_plain
```

The same root cause also affects the valid `where` plus `initial` forms:

```bash
python findings/MS-017/reproducer.py --mode pynative --variant tensor_amax_where_initial
python findings/MS-017/reproducer.py --mode pynative --variant tensor_amin_where_initial
python findings/MS-017/reproducer.py --mode graph --variant ops_amax_where_initial
python findings/MS-017/reproducer.py --mode graph --variant ops_amin_where_initial
python findings/MS-017/reproducer.py --mode graph --variant tensor_amax_where_initial
python findings/MS-017/reproducer.py --mode graph --variant tensor_amin_where_initial
```

The same Graph wrapper signature issue also affects `keepdims`:

```bash
python findings/MS-017/reproducer.py --mode pynative --variant tensor_amax_keepdims
python findings/MS-017/reproducer.py --mode pynative --variant tensor_amin_keepdims
python findings/MS-017/reproducer.py --mode graph --variant ops_amax_keepdims
python findings/MS-017/reproducer.py --mode graph --variant ops_amin_keepdims
python findings/MS-017/reproducer.py --mode graph --variant tensor_amax_keepdims
python findings/MS-017/reproducer.py --mode graph --variant tensor_amin_keepdims
```

Graph Tensor failure:

```text
RuntimeError: Got an unexpected keyword argument 'keepdims'
```

## Localized source evidence

In MindSpore 2.9.0 local source, `Tensor.amin` and `Tensor.amax` expose
`initial` and `where`, then forward them:

```text
common/tensor.py:1463
def amin(self, axis=None, keepdims=False, *, initial=None, where=None):
    ...
    return tensor_operator_registry.get('amin')(self, axis, keepdims,
                                                initial=initial, where=where)

common/tensor.py:1488
def amax(self, axis=None, keepdims=False, *, initial=None, where=None):
    ...
    return tensor_operator_registry.get('amax')(self, axis, keepdims,
                                                initial=initial, where=where)
```

But the Graph wrappers omit those keywords and use `keep_dims` instead of the
Tensor method's `keepdims` name:

```text
graph/_parse/standard_method.py:4528
def amax(input, axis=None, keep_dims=False):
    return F.amax(input, axis, keep_dims)

graph/_parse/standard_method.py:4565
def amin(input, axis=None, keep_dims=False):
    return F.amin(input, axis, keep_dims)
```

## Why this is a real issue

This is not a precision issue. The reproducer uses integer tensors and exact
integer reductions.

This is not a missing CPU kernel. `ops.amax(x, axis=1, initial=10)` and
`ops.amin(x, axis=1, initial=0)` compile and run in Graph mode on the same CPU
input.

This is not a duplicate of the plain `amax/amin` path. Plain Graph Tensor
reductions work; the failure is specifically triggered by valid forwarded
keywords that are absent from the Graph Tensor wrapper signature.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260801T050439Z-p632142`
- Follow-up run: `artifacts/behavioral/20260801T052517Z-p657085`
- Triggering cases: `TREDX-008`, `TREDX-009`, `TREDX-010`, `TREDX-011`
- Minimal script: `findings/MS-017/reproducer.py`
- Repetition summary: `findings/MS-017/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
