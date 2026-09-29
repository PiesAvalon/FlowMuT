# MS-041 Evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`10`
- 退出码 0：`7`；非零：`3`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant none_length_keyword`
- `python reproducer.py --mode pynative --variant none_length_positional`
- `python reproducer.py --mode pynative --variant none_tensor_init`

代表性控制命令：

- `python reproducer.py --mode graph --variant none_length_keyword`
- `python reproducer.py --mode pynative --variant list_none_control`
- `python reproducer.py --mode pynative --variant tuple_tensor_control`

复验说明：Graph 与非 None 控制证明 loop/result 有效，错误明确来自 PyNative `isinstance` 参数验证。
## Source alert

Run:

```text
artifacts/behavioral/20260802T055350Z-p1708822
```

Case:

```text
TCFLOW-008 Scan primitive None xs with length succeeds
```

Observed:

```text
PyNative: TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
Graph:    (3, [1, 2, 3])
```

The failing call is:

```python
def inc(res, el):
    res = res + 1
    return res, res

ops.Scan()(inc, 0, None, 3)
```

## Confirmations

Run:

```text
artifacts/behavioral/20260802T055714Z-p1711407
```

Additional `xs=None` variants fail the same way in PyNative and pass in Graph:

```text
TCFLOW-013 ops.Scan()(inc, 0, None, length=3)
PyNative: TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
Graph:    (3, [1, 2, 3])

TCFLOW-014 ops.Scan()(inc, Tensor(0), None, 3)
PyNative: TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
Graph:    (Tensor(3), [Tensor(1), Tensor(2), Tensor(3)])
```

## Controls

In the same control-flow campaign:

```text
TCFLOW-006 Scan with list xs passes in PyNative and Graph.
TCFLOW-007 Scan with tuple Tensor xs passes in PyNative and Graph.
```

This localizes the failure to the documented `xs=None` path, not to `Scan`
itself, the loop function, CPU execution, or the output structure.

## Local source observation

The installed MindSpore 2.9 source at:

```text
.envs/ms29/lib/python3.11/site-packages/mindspore/ops/operations/manually_defined/ops_def.py
```

documents:

```text
if xs is None:
    xs = [None] * length
```

and declares `xs` as `Union[tuple, list, None]`.

The PyNative implementation calls:

```python
validator.check_value_type("xs", xs, [list, tuple, None], "Scan")
```

Passing `None` instead of `type(None)` causes Python's `isinstance` to raise
`TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union`.
