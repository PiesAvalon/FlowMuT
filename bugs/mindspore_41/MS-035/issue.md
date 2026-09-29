# [Bug]: Graph 模式 `ops.repeat_interleave` 无法处理 0-D Tensor `repeats`

## 🐞 问题详细描述

标量 Tensor repeats 在 PyNative 成功，在 Graph 对 axis 0/1 均触发 `Convert data failed` 内部异常；Python int、单元素 1-D Tensor 和向量 Tensor repeats 在 Graph 正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)

x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
print(Net()(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[ERROR] ANALYZER(105910,756d99caf740,python3.11):2026-08-03-22:54:28.409.867 [mindspore/ccsrc/frontend/jit/ps/static_analysis/evaluator.cc:764] Run] Primitive: <StandardPrimEvaluator_TensorToList> infer failed, failed info: Convert data failed

----------------------------------------------------
- Framework Unexpected Exception Raised:
----------------------------------------------------
This exception is caused by framework's unexpected error. Please create an issue at https://gitee.com/mindspore/mindspore/issues to get help.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/frontend/jit/ps/static_analysis/prim.cc:729 RunPyInferValue

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:9, 15~71
        return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)
               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 In file reproducer.py:9, 15~36
        return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)
               ^~~~~~~~~~~~~~~~~~~~~
# 2 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py:6499~6500, 4~41
    if isinstance(repeats, Tensor):
# 3 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py:6500, 18~41
        repeats = TensorToList()(repeats)
                  ^~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
Traceback (most recent call last):
  File "reproducer.py", line 12, in <module>
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
RuntimeError: Convert data failed

----------------------------------------------------
- Framework Unexpected Exception Raised:
----------------------------------------------------
This exception is caused by framework's unexpected error. Please create an issue at https://gitee.com/mindspore/mindspore/issues to get help.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/frontend/jit/ps/static_analysis/prim.cc:729 RunPyInferValue

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:9, 15~71
        return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)
               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 In file reproducer.py:9, 15~36
        return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)
               ^~~~~~~~~~~~~~~~~~~~~
# 2 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py:6499~6500, 4~41
    if isinstance(repeats, Tensor):
# 3 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py:6500, 18~41
        repeats = TensorToList()(repeats)
                  ^~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

文档将 repeats 声明为可接受 Tensor，0-D Tensor 应与等价 Python int 一样在 Graph 返回正确重复结果。

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

- 问题定位：失败被定位到 0-D Tensor repeats，而不是 CPU、Tensor repeats 整体或特定 axis。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.repeat_interleave.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
