# MS-016 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`4`
- 退出码 0：`3`；非零：`1`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant tensor_eigvals`

代表性控制命令：

- `python reproducer.py --mode pynative --variant ops_eigvals`
- `python reproducer.py --mode graph --variant tensor_eigvals`

复验说明：CPU eigvals 内核通过功能式和 Graph 控制验证，失败局限于 PyNative Tensor wrapper 调用。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.eigvals()`.
- Transformation (`m`): compare `Tensor.eigvals()` with `ops.eigvals(x)`
  under PyNative and Graph mode.
- Oracle (`o`): the Tensor method should dispatch to the same eigvals
  implementation as `ops.eigvals(x)` and should not fail only in PyNative mode.
- Localized boundary: PyNative Tensor wrapper call for `eigvals`.

Input:

```text
[[2. 1.]
 [1. 2.]]
```

## Observed behavior

PyNative `Tensor.eigvals()` fails:

```bash
python findings/MS-016/reproducer.py --mode pynative --variant tensor_eigvals
```

Failure:

```text
TypeError: eigvals() missing 1 required positional argument: 'A'
```

PyNative `ops.eigvals(x)` succeeds:

```bash
python findings/MS-016/reproducer.py --mode pynative --variant ops_eigvals
```

Graph `Tensor.eigvals()` also succeeds:

```bash
python findings/MS-016/reproducer.py --mode graph --variant tensor_eigvals
```

Graph `ops.eigvals(x)` succeeds as well.

## Localized source evidence

MindSpore 2.9.0 local Tensor source:

```text
common/tensor.py:3119
def eigvals(self):
    return tensor_operator_registry.get("eigvals")()(self)
```

That expression calls the registered `eigvals` function with no tensor argument
first, which explains the PyNative error that the required argument `A` is
missing.

The local Graph standard method does not have this extra call:

```text
standard_method.py:3803
def eigvals(x):
    return F.eigvals(x)
```

This matches the observed Graph success.

## Why this is a real issue

This is not a precision issue. The failing Tensor method raises before any
eigenvalues are computed.

This is not an unsupported CPU kernel. `ops.eigvals(x)` works in PyNative and
Graph mode, and `Tensor.eigvals()` works in Graph mode on the same input.

The failure is localized to the PyNative Tensor wrapper call shape:
`tensor_operator_registry.get("eigvals")()(self)` should not call the registered
function with no input before passing `self`.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T235845Z-p497579`
- Triggering case: `TLIN-012`
- Minimal script: `findings/MS-016/reproducer.py`
- Repetition summary: `findings/MS-016/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
