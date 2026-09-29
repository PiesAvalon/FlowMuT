# MS-020 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`12`
- 退出码 0：`12`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant next_iter`
- `python reproducer.py --mode graph --variant next_enumerate`
- `python reproducer.py --mode graph --variant next_enumerate_value`

代表性控制命令：

- `python reproducer.py --mode pynative --variant next_iter`
- `python reproducer.py --mode pynative --variant next_enumerate`
- `python reproducer.py --mode graph --variant for_iter`
- `python reproducer.py --mode graph --variant list_enumerate_value`

复验说明：这是退出码为 0 的静默结构错误；issue 预期按文档明确拒绝，不主张 Graph 必须支持 next。
## FlowMuT candidate

- Mutation site (`tau`): explicit Python `next()` applied to Tensor iterators
  and Tensor `enumerate()` iterators inside `construct`.
- Transformation (`m`): compare explicit `next(...)` with materialized
  `list(enumerate(...))` and `for` loop controls.
- Oracle (`o`): Graph mode should not silently return a different
  user-visible Python structure from PyNative mode. If `next` is unsupported
  in static graph syntax, Graph mode should reject it instead of compiling a
  wrong result.
- Localized boundary: Graph built-in `next` handling returns the internal
  iterator protocol tuple `(current, rest)`.

## Observed behavior

Input:

```text
x = Tensor([[1, 4, 2], [3, 0, 5]], float32)
```

Stable alerts:

```text
next(iter(x))
PyNative -> Tensor([1, 4, 2])
Graph    -> (Tensor([1, 4, 2]), Tensor([[3, 0, 5]]))

next(enumerate(x))
PyNative -> (0, Tensor([1, 4, 2]))
Graph    -> ((0, Tensor([1, 4, 2])), ((1, Tensor([3, 0, 5])),))

next(enumerate(x))[1]
PyNative -> Tensor([1, 4, 2])
Graph    -> ((1, Tensor([3, 0, 5])),)
```

Controls over the same input are stable:

```text
list(enumerate(x))[0][1]              -> Tensor([1, 4, 2]) in both modes
for row in x: return row              -> Tensor([1, 4, 2]) in both modes
for _, row in enumerate(x): return row -> Tensor([1, 4, 2]) in both modes
```

## Localized source evidence

MindSpore 2.9.0 local Graph parser helper:

```text
graph/_parse/standard_method.py

def enumerate_(x, start=0):
    ...
    if check_is_tensor(x_type):
        for i in range(x.shape[0]):
            ret += ((start + i, x[i]),)

def ms_next(xs):
    """Get next element and res elements"""
    return xs[0], xs[1:]
```

The observed Graph result is exactly the helper's internal `(xs[0], xs[1:])`
shape. That internal pair appears to escape as the return value of user code
instead of only driving loop lowering.

## Documentation basis

MindSpore 2.9 static graph syntax documentation lists `enumerate` among
supported built-in functions. The same page lists `next` among unsupported
built-in functions. Therefore the conservative expected behavior for explicit
`next(...)` in Graph mode is an unsupported-syntax error, not a successfully
compiled result with the wrong Python structure.

Python's builtin `next(iterator)` returns the next item from an iterator.
Python's builtin `enumerate(iterable)` yields `(index, value)` pairs.

## Why this is a real issue

This is not a precision issue. The mismatch is a structural return mismatch:
Graph returns an extra "rest of iterator" object.

This is not a missing CPU kernel issue. The failing cases compile and execute,
and the control cases over the same Tensor input pass on CPU.

This is not a general Tensor iteration issue. `list(enumerate(x))`, `for row in
x`, and `for _, row in enumerate(x)` all match PyNative mode. The problem is
the explicit `next(...)` path.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260801T084449Z-p854778`
- Repetition runs:
  - `artifacts/behavioral/20260801T084655Z-p856323`
  - `artifacts/behavioral/20260801T084655Z-p856329`
- Triggering cases: `TITER-001` through `TITER-003`
- Passing controls: `TITER-004` through `TITER-006`
- Minimal script: `findings/MS-020/reproducer.py`
- Repetition summary: `findings/MS-020/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
