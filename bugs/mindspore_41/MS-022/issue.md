# [Bug]: Graph 过滤序列时，Tensor bool 谓词触发内部空指针错误

## 🐞 问题详细描述

filter、带条件的 list/tuple comprehension 和 generator 在谓词返回标量 Tensor bool 时，Graph 触发 `pointer[seq_abs] is null` 类内部错误；Python-bool 谓词、无过滤 comprehension 和等价 if Tensor 谓词均成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return tuple(filter(lambda row: row.sum() > 0, x))

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
print(Net()(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[ERROR] ANALYZER(105069,7922d537a740,python3.11):2026-08-03-22:53:58.380.051 [mindspore/ccsrc/frontend/jit/ps/static_analysis/evaluator.cc:764] Run] Primitive: <StandardPrimEvaluator_sequence_len> infer failed, failed info: The pointer[seq_abs] is null.

----------------------------------------------------
- Framework Unexpected Exception Raised:
----------------------------------------------------
This exception is caused by framework's unexpected error. Please create an issue at https://gitee.com/mindspore/mindspore/issues to get help.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/infer//sequence_len.cc:46 SequenceLenInferInner

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file <python_env>/lib/python3.11/site-packages/mindspore/graph/_parse/standard_method.py:2756, 11~31
    return input_data.__len__()
           ^~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
Traceback (most recent call last):
  File "reproducer.py", line 11, in <module>
    print(Net()(x))
          ^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1409, in __call__
    out = self.compile_and_run(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1800, in compile_and_run
    self.compile(*args, **kwargs)
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1782, in compile
    _cell_graph_executor.compile(self, *compile_args, phase=self.phase,
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 2354, in compile
    result = self._graph_executor.compile(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: The pointer[seq_abs] is null.

----------------------------------------------------
- Framework Unexpected Exception Raised:
----------------------------------------------------
This exception is caused by framework's unexpected error. Please create an issue at https://gitee.com/mindspore/mindspore/issues to get help.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/infer//sequence_len.cc:46 SequenceLenInferInner

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file <python_env>/lib/python3.11/site-packages/mindspore/graph/_parse/standard_method.py:2756, 11~31
    return input_data.__len__()
           ^~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

受支持的过滤语法应正确解释标量 Tensor bool；即使该组合不支持，也应给出明确的用户级语法错误，不能暴露编译器空指针异常。

## 详细的环境信息描述

- MindSpore：`2.10.0`，PyPI official CPU binary wheel
- Python：`3.11.15`
- 操作系统：`Ubuntu 24.04.4 LTS`，WSL2
- Linux kernel：`6.6.87.2-microsoft-standard-WSL2`
- CPU：`Intel(R) Core(TM) i7-14700K`，x86_64
- Device target：`CPU`
- 执行模式：`Graph`
- 系统 GCC：`13.3.0`；MindSpore 使用预编译二进制包，未在本机编译
- GPU/Ascend 驱动：不适用
- 安装方式：Python 虚拟环境内安装 MindSpore 2.10.0 wheel

## 补充说明

- 问题定位：失败集中在序列过滤与 Tensor bool 谓词的组合，内部 null-pointer 信息本身也构成编译器错误。

参考文档：

- https://www.mindspore.cn/tutorials/en/r2.9.0/compile/static_graph.html
- https://docs.python.org/3/library/functions.html#filter

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
