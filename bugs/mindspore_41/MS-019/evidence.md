# MS-019 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`24`
- 退出码 0：`18`；非零：`6`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant left_keyword`
- `python reproducer.py --mode graph --variant renorm_keyword`
- `python reproducer.py --mode graph --variant multiply_keyword`
- `python reproducer.py --mode graph --variant equal_keyword`

代表性控制命令：

- `python reproducer.py --mode graph --variant left_positional`
- `python reproducer.py --mode graph --variant renorm_positional`
- `python reproducer.py --mode pynative --variant multiply_keyword`
- `python reproducer.py --mode graph --variant equal_positional`

复验说明：六个方法共享同一根因形态；对应位置参数路径全部成功，因此合并为一个签名绑定 issue。
## FlowMuT candidate

- Mutation site (`tau`): Tensor method calls using documented/public keyword
  parameter names.
- Transformation (`m`): replace positional Tensor method calls with equivalent
  keyword calls.
- Oracle (`o`): PyNative and Graph modes should both accept the public Tensor
  method signature. A keyword call should match the positional control.
- Localized boundary: Graph `standard_method` wrappers use different parameter
  names from the public `Tensor` methods.

Affected calls:

```text
x.bitwise_left_shift(other=shift)
x.bitwise_right_shift(other=shift)
x.renorm(2, axis=0, maxnorm=5.0)
x.float_power(other=2)
x.multiply(value=other)
x.equal(other=other)
```

## Observed behavior

Each keyword call succeeds in PyNative mode and matches the positional control.
Each corresponding positional call also succeeds in Graph mode. The Graph
keyword calls fail before execution:

```text
TKEY-001 Graph x.bitwise_left_shift(other=...)
-> RuntimeError: Got an unexpected keyword argument 'other'

TKEY-002 Graph x.bitwise_right_shift(other=...)
-> RuntimeError: Got an unexpected keyword argument 'other'

TKEY-003 Graph x.renorm(2, axis=0, maxnorm=5.0)
-> RuntimeError: Got an unexpected keyword argument 'axis'

TKEY-004 Graph x.float_power(other=2)
-> RuntimeError: Got an unexpected keyword argument 'other'

TKEY-005 Graph x.multiply(value=...)
-> RuntimeError: Got an unexpected keyword argument 'value'

TKEY-006 Graph x.equal(other=...)
-> RuntimeError: Got an unexpected keyword argument 'other'
```

The errors point to Graph keyword binding:

```text
mindspore/core/ir/func_graph_extends.cc:161 GenerateKwParams
```

## Localized source evidence

MindSpore 2.9.0 local Tensor methods expose these keyword names:

```text
common/tensor.py
def bitwise_left_shift(self, other)
def bitwise_right_shift(self, other)
def renorm(self, p, axis, maxnorm)
def float_power(self, other)
def multiply(self, value)
def equal(self, other)
```

The matching Graph wrappers use different names:

```text
graph/_parse/standard_method.py
def bitwise_left_shift(x, y)
def bitwise_right_shift(x, y)
def renorm(input, p, dim, maxnorm)
def float_power(input, exponent)
def multiply(input, other)
def equal(x, y)
```

Graph mode therefore rejects the public Tensor keyword even though the same
call is valid in PyNative mode.

## Why this is a real issue

This is not a precision issue. The failures occur during Graph argument
binding before any numerical comparison.

This is not a missing CPU kernel. For every affected method, the positional
Graph control succeeds on CPU with the same input.

This is grouped as one issue rather than six separate findings because the
root cause is the same signature mismatch pattern in Graph Tensor wrappers.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260801T062000Z-p709112`
- Repetition runs:
  - `artifacts/behavioral/20260801T062348Z-p711607`
  - `artifacts/behavioral/20260801T062529Z-p712861`
  - `artifacts/behavioral/20260801T062706Z-p714130`
- Triggering cases: `TKEY-001` through `TKEY-006`
- Minimal script: `findings/MS-019/reproducer.py`
- Repetition summary: `findings/MS-019/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
