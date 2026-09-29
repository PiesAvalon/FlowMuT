# [Bug]: `Tensor.diagonal_scatter(src)` 在 Graph 模式省略默认 `offset` 时失败

## 🐞 问题详细描述

公开签名将 `offset` 默认设为 0。PyNative 的默认调用成功；Graph 只有省略 `offset` 的 Tensor 方法失败，显式位置/关键字 `offset=0` 以及功能式调用均成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, src):
        return x.diagonal_scatter(src)

x = ms.Tensor([[1, 4, 2], [3, 0, 5]], ms.float32)
src = ms.Tensor([9, 8], ms.float32)
print(Net()(x, src))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 12, in <module>
    print(Net()(x, src))
          ^^^^^^^^^^^^^
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
RuntimeError: Miss argument input for parameter:offset

----------------------------------------------------
- Framework Unexpected Exception Raised:
----------------------------------------------------
This exception is caused by framework's unexpected error. Please create an issue at https://gitee.com/mindspore/mindspore/issues to get help.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/ir/func_graph_extends.cc:249 GenerateDefaultValue

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:8, 15~38
        return x.diagonal_scatter(src)
               ^~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

Graph Tensor 方法应应用公开签名中的 `offset=0` 默认值，并返回与显式传参和 PyNative 相同的结果。

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

- 问题定位：失败仅由 Graph Tensor 方法省略默认参数触发，显式 `offset=0` 和功能式入口均通过。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
