# MS-005 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`6`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --order default_then_int32`
- `python reproducer.py --mode pynative --order int32_then_default`

代表性控制命令：

- `python reproducer.py --mode pynative --order default`
- `python reproducer.py --mode pynative --order int32`
- `python reproducer.py --mode graph --order default_then_int32`

复验说明：失败只出现在 PyNative 的混合物化组合；单调用和 Graph 组合均通过。
## FlowMuT candidate

- Mutation site (`tau`): a CPU `ops.searchsorted` call whose output dtype is
  controlled by `out_int32`.
- Transformation (`m`): execute two valid `ops.searchsorted` calls in the same
  `nn.Cell`, one with the default `out_int32=False` and one with
  `out_int32=True`.
- Oracle (`o`): each call is valid independently, and Graph mode can materialize
  both outputs in the same return tuple, so PyNative mode should also be able to
  materialize both outputs.
- Localized boundary: PyNative CPU kernel execution / output dtype handling in
  `SearchSorted`.

MindSpore 2.9.0 documents `ops.searchsorted` as CPU supported with keyword
argument `out_int32` defaulting to `False`.  In isolated calls, both
`out_int32=False` and `out_int32=True` work correctly on CPU:

- `ops.searchsorted(sorted_sequence, values)` returns `int64`.
- `ops.searchsorted(sorted_sequence, values, out_int32=True)` returns `int32`.

However, in PyNative mode, returning both calls from the same `nn.Cell` and
materializing the tuple fails on the second tensor.  The failure is independent
of order:

- `default_then_int32` fails with `v 24` versus `output 24`.
- `int32_then_default` fails with `v 24` versus `output 48`.

Graph mode succeeds for both mixed-output orders and returns the expected
`(int64, int32)` or `(int32, int64)` tuple.  This makes the issue non-numerical
and mode-specific.  The failing frame is:

```text
mindspore/ops/kernel/cpu/native/searchsorted_cpu_kernel.cc:104 CheckParam
```

The same root cause is reachable through `mindspore.mint.searchsorted`, which
wraps the same operator.  It is not counted as a separate issue.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260731T194237Z-p249261`
- Triggering cases: `TSCH-003` and `TSCH-004`
- Minimal script: `findings/MS-005/reproducer.py`

## Duplicate search

No exact matching open or closed GitHub issue was found on 2026-08-01 using the
queries in `metadata.json`.  A related Gitee issue discusses `searchsorted`
side validation and overflow behavior, but not this same-process mixed
`out_int32` CPU kernel failure.
