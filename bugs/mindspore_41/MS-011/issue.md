# [Bug]: `Tensor.diag()` 在 Graph 模式因 deprecated Tensor method 未注册而失败

## 🐞 问题详细描述

`Tensor.diag()` 在 PyNative 成功，但 Graph 触发框架 unexpected error；相同输入的 `ops.diag` 在两模式都成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.diag()

print(Net()(ms.Tensor([1, 2, 3], ms.int32)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 10, in <module>
    print(Net()(ms.Tensor([1, 2, 3], ms.int32)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
RuntimeError: As a deprecated Tensor method, 'diag' should be registered in graph/_parse/deprecated/deprecated_tensor_method.py::deprecated_tensor_method_map

----------------------------------------------------
- Framework Unexpected Exception Raised:
----------------------------------------------------
This exception is caused by framework's unexpected error. Please create an issue at https://gitee.com/mindspore/mindspore/issues to get help.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/frontend/operator/composite/functional_overload.cc:512 GenerateFuncGraph

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:8, 15~23
        return x.diag()
               ^~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

Graph 应将仍公开的 Tensor 方法正确路由到 `ops.diag`，或给出明确的弃用提示，不能因内部注册缺失失败。

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

- 问题定位：仅 Graph Tensor 方法入口失败，功能式算子与 PyNative Tensor 方法均返回正确对角矩阵。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
