# MS-026 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`6`
- 退出码 0：`6`；非零：`0`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --window hann`
- `python reproducer.py --mode graph --window kaiser`

代表性控制命令：

- `python reproducer.py --mode pynative --window hann`
- `python reproducer.py --mode graph --window hamming`

复验说明：这是退出码为 0 的返回类型错误，且被定位到 Graph 中特定 Python 组合窗口输出的 shape 推断。
## FlowMuT candidate

- Mutation site (`tau`): `ops.shape(y)` where `y` is created by window
  factory functions.
- Transformation (`m`): replace the primitive-backed `ops.hamming_window`
  control with the Python-composed `ops.hann_window` or `ops.kaiser_window`.
- Oracle (`o`): `mindspore.ops.shape` documents a `tuple[int]` return value
  and should agree with the tensor's `.shape` metadata across PyNative and
  Graph mode.
- Localized boundary: Graph-mode shape inference for tensors returned by
  `ops.hann_window` and `ops.kaiser_window`.

## Observed behavior

Input is fully static:

```text
y = ops.<window>(5, dtype=ms.float32)
return ops.shape(y), y.shape, y
```

Passing control:

```text
ops.hamming_window(5, dtype=ms.float32)
PyNative -> ops.shape(y) == (5,), y.shape == (5,), y is Tensor(shape=[5], dtype=Float32)
Graph    -> ops.shape(y) == (5,), y.shape == (5,), y is Tensor(shape=[5], dtype=Float32)
```

Failing cases:

```text
ops.hann_window(5, dtype=ms.float32)
PyNative -> ops.shape(y) == (5,), y.shape == (5,)
Graph    -> ops.shape(y) is Tensor(5, dtype=int64), y.shape == (5,)

ops.kaiser_window(5, dtype=ms.float32)
PyNative -> ops.shape(y) == (5,), y.shape == (5,)
Graph    -> ops.shape(y) is Tensor(5, dtype=int64), y.shape == (5,)
```

The tensor itself is produced successfully in all cases. The mismatch is the
type/structure returned by `ops.shape`.

## Why this is a real issue

This is not a numerical precision issue. The reproducer does not compare
floating-point window values.

This is not a general CPU unsupported issue. All three window functions run on
CPU in both modes, and the `hamming_window` control returns the documented
shape metadata.

This is not an invalid input issue. `window_length=5` is a static Python int,
and the returned tensor has `shape == (5,)` in both PyNative and Graph mode.

`ops.shape` documents `tuple[int]`. Returning a scalar Tensor in Graph mode can
break code that branches on tuple shape metadata or expects `ops.shape(y)` to
match `y.shape`.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T171915Z-p1262997`
- Repetition runs:
  - `artifacts/behavioral/20260801T172112Z-p1264636`
  - `artifacts/behavioral/20260801T172208Z-p1265426`
- Triggering cases: `TWINSHAPE-002`, `TWINSHAPE-003`
- Passing control: `TWINSHAPE-001`
- Minimal script: `findings/MS-026/reproducer.py`
- Repetition summary: `findings/MS-026/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-02 using the queries
listed in `metadata.json`.
