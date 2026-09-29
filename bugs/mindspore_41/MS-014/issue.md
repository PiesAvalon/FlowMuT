# [Bug]: `Tensor.expand` 在 Graph 模式无法解析公开的 size 参数形式

## 🐞 问题详细描述

PyNative 的 varargs 和 tuple 两种 `Tensor.expand` 调用均成功；Graph 中 varargs 报参数数量错误，tuple 报 `GetShapeVector()` 未实现，而相同目标形状的 `broadcast_to` 成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.expand(2, 3)

print(Net()(ms.Tensor([[1, 2, 3]], ms.float32)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 10, in <module>
    print(Net()(ms.Tensor([[1, 2, 3]], ms.float32)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
TypeError: The parameters number of the function is 2, but the number of provided arguments is 3.
FunctionGraph : expand_2
NodeInfo: In file <python_env>/lib/python3.11/site-packages/mindspore/graph/_parse/standard_method.py:4207~4212
def expand(input, size):

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/frontend/jit/ps/static_analysis/evaluator.cc:457 Eval

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:8, 15~29
        return x.expand(2, 3)
               ^~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

Graph 应支持 Tensor 公开签名 `expand(*size)` 接受的合法尺寸形式，并与 PyNative 返回相同广播结果。

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

- 问题定位：相同数据和目标形状可由 PyNative expand 与 Graph broadcast_to 正确处理，失败局限于 Graph Tensor.expand 参数降低。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
