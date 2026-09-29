# MS-018 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`6`
- 退出码 0：`4`；非零：`2`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode pynative --variant tensor_default`
- `python reproducer.py --mode graph --variant tensor_default`

代表性控制命令：

- `python reproducer.py --mode pynative --variant tensor_explicit`
- `python reproducer.py --mode graph --variant tensor_explicit`
- `python reproducer.py --mode graph --variant ops_explicit`

复验说明：问题同时包含不合法的公开默认值和 Graph 默认参数绑定差异；显式合法参数控制排除了 CPU 内核问题。
## FlowMuT candidate

- Mutation site (`tau`): `Tensor.reverse_sequence(seq_lengths)` using the
  Tensor method defaults.
- Transformation (`m`): compare the default Tensor call with explicit
  `seq_dim=1` Tensor and `ops.reverse_sequence` controls.
- Oracle (`o`): a public Tensor method default should be usable, or at least
  fail consistently with a documented validation error. The explicit controls
  demonstrate the CPU kernel and input are valid.
- Localized boundary: Tensor default arguments and Graph `standard_method`
  signature for `reverse_sequence`.

Input:

```text
x = [[1 2 3]
     [4 5 6]]
seq_lengths = [2 3]
```

Expected explicit result:

```text
[[2 1 3]
 [6 5 4]]
```

## Observed behavior

Explicit Tensor and ops controls succeed:

```bash
python findings/MS-018/reproducer.py --mode pynative --variant tensor_explicit
python findings/MS-018/reproducer.py --mode graph --variant tensor_explicit
python findings/MS-018/reproducer.py --mode graph --variant ops_explicit
```

The default Tensor call fails in PyNative mode:

```bash
python findings/MS-018/reproducer.py --mode pynative --variant tensor_default
```

Failure:

```text
ValueError: For 'ReverseSequence', the 'batch_dim' should be != seq_dim: 0, but got 0.
```

The same default Tensor call fails differently in Graph mode:

```bash
python findings/MS-018/reproducer.py --mode graph --variant tensor_default
```

Failure:

```text
RuntimeError: Miss argument input for parameter:seq_dim
mindspore/core/ir/func_graph_extends.cc:249 GenerateDefaultValue
```

The Graph error also labels itself as a framework unexpected exception and asks
the user to create an issue.

## Localized source evidence

MindSpore 2.9.0 local `Tensor.reverse_sequence` exposes `seq_dim=0` and
`batch_dim=0`, then forwards both:

```text
common/tensor.py:1512
def reverse_sequence(self, seq_lengths, seq_dim=0, batch_dim=0):
    return tensor_operator_registry.get("reverse_sequence")(self, seq_lengths,
                                                            seq_dim, batch_dim)
```

But the Graph wrapper makes `seq_dim` required:

```text
graph/_parse/standard_method.py:705
def reverse_sequence(x, seq_lengths, seq_dim, batch_dim=0):
    return F.reverse_sequence(x, seq_lengths, seq_dim, batch_dim)
```

The PyNative path reaches the underlying op and shows the Tensor defaults are
internally invalid, because `seq_dim` must not equal `batch_dim`.

## Why this is a real issue

This is not a precision issue. The reproducer uses integer sequence reversal.

This is not a missing CPU kernel. The explicit Tensor method and direct ops
forms work in Graph mode on CPU with the same input.

This is not counted as a duplicate of any previous finding. `MS-009` covers a
missing default for `Tensor.diagonal_scatter`; this is a separate Tensor method
and a separate invalid default combination.

## Reproduction artifacts

- Discovery run: `artifacts/behavioral/20260801T051823Z-p648854`
- Triggering cases: `TSEQ-008`, `TSEQ-009`
- Minimal script: `findings/MS-018/reproducer.py`
- Repetition summary: `findings/MS-018/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
