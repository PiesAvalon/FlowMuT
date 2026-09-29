# MS-022 evidence

## 2026-08-03 全量复验

- 判定：`confirmed / issue-ready`
- 全部声明参数组合：`20`
- 退出码 0：`15`；非零：`5`；超时：`0`
- 原始结果：`artifacts/revalidation/20260803T124653Z/results.json`

代表性目标命令：

- `python reproducer.py --mode graph --variant filter_tensor_predicate`
- `python reproducer.py --mode graph --variant list_comprehension_predicate`
- `python reproducer.py --mode graph --variant generator_predicate`

代表性控制命令：

- `python reproducer.py --mode pynative --variant filter_tensor_predicate`
- `python reproducer.py --mode graph --variant filter_tensor_true_control`
- `python reproducer.py --mode graph --variant list_comprehension_no_filter_control`
- `python reproducer.py --mode graph --variant if_tensor_predicate_control`

复验说明：失败集中在序列过滤与 Tensor bool 谓词的组合，内部 null-pointer 信息本身也构成编译器错误。
## FlowMuT candidate

- Mutation site (`tau`): Python builtin `filter`, list comprehensions and
  generator expressions in `construct` where the predicate returns a scalar
  Tensor bool.
- Transformation (`m`): compare filtered sequence forms using
  `row.sum() > 0` with unfiltered comprehensions, `filter(lambda row: True,
  ...)`, and an equivalent `if row.sum() > 0` control.
- Oracle (`o`): Graph mode should not crash with an internal null pointer for a
  predicate form whose scalar Tensor bool condition works in an `if`.
- Localized boundary: Graph filtered-sequence lowering/transform handles Python
  bool predicates and unfiltered comprehensions but fails when the filter
  predicate returns a scalar Tensor bool.

## Observed behavior

Input:

```text
x = Tensor([[1, 4, 2], [3, 0, 5]], float32)
```

Failing cases:

```text
tuple(filter(lambda row: row.sum() > 0, x))
tuple(filter(lambda row: row.sum() > 0, (x[0], x[1])))
tuple([row for row in x if row.sum() > 0])
tuple(row for row in x if row.sum() > 0)
tuple([row for row in (x[0], x[1]) if row.sum() > 0])

PyNative -> (Tensor([1, 4, 2]), Tensor([3, 0, 5]))
Graph    -> RuntimeError: The pointer[seq_abs] is null.
```

Passing controls:

```text
tuple(filter(lambda row: True, x))
tuple(filter(lambda row: True, (x[0], x[1])))
tuple([row + 1 for row in x])
tuple(row + 1 for row in x)

for row in x:
    if row.sum() > 0:
        return row
```

The controls match PyNative and Graph mode.

## Localized source evidence

MindSpore 2.9.0 maps Python builtin `filter` to
`graph/_parse/standard_method.py::filter_`:

```text
def filter_(fun, iter_):
    result = []
    for elem in iter_:
        if fun(elem):
            result.append(elem)
    return result
```

The observed error message references a null `seq_abs` pointer while lowering
sequence length. The traceback points to MindSpore's Graph `len` helper:

```text
graph/_parse/standard_method.py

def ms_len(input_data):
    """Implementation of `len`."""
    return input_data.__len__()
```

The C++ call stack reports:

```text
mindspore/ops/infer//sequence_len.cc:46 SequenceLenInferInner
RuntimeError: The pointer[seq_abs] is null.
```

The null-pointer error appears only for filtered sequence forms with
Tensor-bool predicates, not for Python-bool predicates, unfiltered
comprehensions, or equivalent `if` control flow.

## Why this is a real issue

This is not a precision issue. It is an internal Graph compiler error.

This is not a missing CPU kernel. The error occurs before backend execution,
and the controls run on CPU.

This is not a general inability to use Tensor scalar bools in Graph mode. The
equivalent `if row.sum() > 0` control passes.

## Reproduction artifacts

- Source run: `artifacts/behavioral/20260801T090152Z-p873310`
- Repetition runs:
  - `artifacts/behavioral/20260801T090152Z-p873316`
  - `artifacts/behavioral/20260801T090400Z-p876265`
  - `artifacts/behavioral/20260801T090856Z-p879832`
  - `artifacts/behavioral/20260801T092219Z-p889780`
  - `artifacts/behavioral/20260801T092408Z-p891195`
- Triggering cases: `TPYB-006`, `TPYB-008`, `TCOMP-003`, `TCOMP-004`,
  `TCOMP-005`
- Passing controls: `TPYB-009`, `TPYB-010`, `TPYB-011`, `TCOMP-001`,
  `TCOMP-002`
- Minimal script: `findings/MS-022/reproducer.py`
- Repetition summary: `findings/MS-022/repetitions/summary.json`

## Duplicate search

No exact matching GitHub/Gitee issue was found on 2026-08-01 using the queries
listed in `metadata.json`.
