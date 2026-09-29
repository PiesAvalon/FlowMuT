# MS-042 Evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`18`
- 退出码 0：`15`；非零：`3`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant tensor_cond_bound`
- `python reproducer.py --mode graph --variant tensor_loop_step`
- `python reproducer.py --mode graph --variant tensor_attrs`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_cond_bound`
- `python reproducer.py --mode graph --variant python_constants_control`
- `python reproducer.py --mode graph --variant global_tensor_constant_control`
- `python reproducer.py --mode graph --variant nested_tensor_constant_control`
- `python reproducer.py --mode graph --variant fori_tensor_constant_control`
- `python reproducer.py --mode graph --variant scan_tensor_constant_control`

复验说明：已从原草稿的过宽描述收窄到“局部 WhileLoop 回调 + Tensor 常量”的组合，模块级回调是新增通过控制。
## 2026-08-03 重新验证

本次复验环境：MindSpore `2.9.0`、Python `3.11.15`、CPU、Ubuntu
`24.04.4 LTS (WSL2)`。所有用例均在独立 Python 进程中执行。

执行完整的 PyNative/Graph 对照矩阵后得到：

| 变体 | PyNative | Graph |
| --- | --- | --- |
| `tensor_cond_bound` | 成功，返回 `int32(5)` | 失败，`ValueError` / `EvalCNode` |
| `tensor_loop_step` | 成功，返回 `int32(5)` | 失败，`ValueError` / `EvalCNode` |
| `tensor_attrs` | 成功，返回 `int32(5)` | 失败，`ValueError` / `EvalCNode` |
| `python_constants_control` | 成功，返回 `int32(5)` | 成功，返回 `int32(5)` |
| `global_tensor_constant_control` | 成功，返回 `int32(5)` | 成功，返回 `int32(5)` |
| `tuple_tensor_carry_control` | 成功 | 成功 |
| `nested_tensor_constant_control` | 成功 | 成功 |
| `fori_tensor_constant_control` | 成功 | 成功 |
| `scan_tensor_constant_control` | 成功 | 成功 |

另用三个新的独立进程重复执行 Graph 最小失败用例
`tensor_cond_bound`，`3/3` 次均出现：

```text
Can not cast to a AbstractFunction from AbstractProblem(
Value: DeadNode, Node: ValueNode<FuncGraph> while_loop_7).

ValueError: The object is not callable. Please check code.
mindspore/ccsrc/frontend/jit/ps/static_analysis/static_analysis.cc:1245 EvalCNode
```

复验结论：问题稳定存在，且被进一步定位到 `WhileLoop` 的局部回调函数中捕获
或构造 Tensor 常量这一组合条件；同样的 Tensor 常量在模块级回调中可以运行。
该问题不是数值精度问题、CPU 算子缺失、Tensor carry 非法、一般的 Graph
局部函数限制或 Graph 对 Tensor 常量的普遍限制。

2026-08-03 使用精确错误信息及 `WhileLoop`/`Tensor`/`Graph` 组合关键词重新搜索
Gitee 与 GitHub，未发现相同问题。

## Source alert

Run:

```text
artifacts/behavioral/20260802T055350Z-p1708822
```

Case:

```text
TCFLOW-005 WhileLoop primitive Tensor scalar carry succeeds
```

Observed:

```text
PyNative: Tensor scalar result
Graph:    ValueError: The object is not callable. Please check code.
```

## Confirmations

Runs:

```text
artifacts/behavioral/20260802T055714Z-p1711407
artifacts/behavioral/20260802T060201Z-p1714890
```

All of these variants define the supplied callbacks locally inside
`construct`; they pass in PyNative mode and fail in Graph mode with the same
`ValueError`:

```text
TCFLOW-011: Tensor constants in both cond_func and loop_func
TCFLOW-015: Tensor constant only in cond_func
TCFLOW-016: Tensor constant only in loop_func
TCFLOW-017: Tensor constants stored as Cell attributes
```

The common failure is:

```text
ValueError: The object is not callable. Please check code.
mindspore/ccsrc/frontend/jit/ps/static_analysis/static_analysis.cc:1245 EvalCNode
```

For one source case, the MindSpore analyzer also logs:

```text
Can not cast to a AbstractFunction from AbstractProblem(Value: DeadNode, Node: ValueNode<FuncGraph> while_loop_7).
```

## Controls

Control cases that pass:

```text
TCFLOW-004 WhileLoop documented Python-int example
TCFLOW-010 WhileLoop Tensor carry with Python int constants
TCFLOW-012 WhileLoop tuple Tensor carry with Python int constants
TCFLOW-018 Plain nested function with Tensor constant
TCFLOW-019 ForiLoop loop_func with Tensor constant
TCFLOW-020 Scan loop_func with Tensor constant
global_tensor_constant_control: module-level WhileLoop callbacks with a Tensor constant
```

These controls show that the failure is not caused by CPU execution, Tensor
scalar carry values, local functions, or Tensor constants in Graph mode in
general. A module-level `WhileLoop` callback with the same Tensor constant also
passes. The failure is specific to `WhileLoop` when a locally defined supplied
callback uses a Tensor constant.

## Minimal failing pattern

```python
class Net(nn.Cell):
    def construct(self):
        def cond(value):
            return value < ms.Tensor(5, ms.int32)

        def loop(value):
            return value + 1

        return ops.WhileLoop()(cond, loop, ms.Tensor(0, ms.int32))
```

This passes in PyNative and fails in Graph. The analogous version with
`return value < 5` passes in both modes, as does the Tensor-constant version
when `cond` and `loop` are defined at module level.
