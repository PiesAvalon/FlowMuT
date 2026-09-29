# MS-023 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`8`
- 退出码 0：`2`；非零：`6`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant ops_positive_decimals`
- `python reproducer.py --mode graph --variant ops_negative_decimals`
- `python reproducer.py --mode graph --variant tensor_positive_decimals`

代表性控制命令：

- `python reproducer.py --mode pynative --variant ops_default_control`
- `python reproducer.py --mode graph --variant ops_default_control`

复验说明：默认 round 在 CPU 正常，失败只由文档允许的非零 decimals 触发。
## FlowMuT candidate

- Mutation site (`tau`): `ops.round` and `Tensor.round` on CPU.
- Transformation (`m`): pass a nonzero documented `decimals` value while
  keeping the same input tensor, execution modes, and default-decimals control.
- Oracle (`o`): the public `decimals` keyword should work on CPU because the
  official `ops.round` documentation lists CPU support and documents positive
  and negative `decimals` examples.
- Localized boundary: CPU `Round` kernel parameter support.

## Observed behavior

Input:

```text
x = Tensor([0.1234567, 1.25, -2.35, 1200.1234], float32)
```

Passing control:

```text
ops.round(x)
PyNative -> Tensor([0, 1, -2, 1200], float32)
Graph    -> Tensor([0, 1, -2, 1200], float32)
```

Failing cases:

```text
ops.round(x, decimals=1)
ops.round(x, decimals=-1)
x.round(decimals=1)
```

Both PyNative and Graph mode fail on CPU with:

```text
RuntimeError: For Round only support decimals equal 0, but got decimals equal 1
RuntimeError: For Round only support decimals equal 0, but got decimals equal -1

mindspore/ops/kernel/cpu/native/round_cpu_kernel.cc:50 LaunchKernel
```

## Why this is a real issue

This is not a numerical precision issue; no rounded value is produced.

This is not a Graph-only issue; the same CPU kernel error appears in PyNative
and Graph mode.

This is not a general lack of CPU support for `round`; `decimals=0` succeeds in
both modes. The failure is specific to the documented nonzero `decimals`
keyword.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T111118Z-p1029612`
- Repetition runs:
  - `artifacts/behavioral/20260801T111411Z-p1031799`
  - `artifacts/behavioral/20260801T111411Z-p1031805`
- Triggering cases: `TROUND-001`, `TROUND-002`, `TROUND-003`
- Passing control: `TREF-006`
- Minimal script: `findings/MS-023/reproducer.py`
- Repetition summary: `findings/MS-023/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
