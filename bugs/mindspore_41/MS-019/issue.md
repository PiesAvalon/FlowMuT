# [Bug]: 多个 Tensor 方法的公开关键字在 Graph 模式因内部参数名不一致而被拒绝

## 🐞 问题详细描述

`bitwise_left_shift/right_shift(other)`、`renorm(axis=...)`、`float_power(other)`、`multiply(value)`、`equal(other)` 在 PyNative 和 Graph 位置参数形式正常，但 Graph 关键字形式分别报 `other`、`axis` 或 `value` 为意外参数。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, shift):
        return x.bitwise_left_shift(other=shift)

x = ms.Tensor([[3, 2, 1], [6, 5, 4]], ms.int32)
print(Net()(x, ms.Tensor(1, ms.int32)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 11, in <module>
    print(Net()(x, ms.Tensor(1, ms.int32)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
RuntimeError: Got an unexpected keyword argument 'other'

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/ir/func_graph_extends.cc:161 GenerateKwParams

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:8, 15~48
        return x.bitwise_left_shift(other=shift)
               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

Graph 的 Tensor method wrapper 参数名应与公开 Python 签名一致，接受文档中的关键字并返回与位置参数相同的结果。

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

- 问题定位：六个方法表现为相同的关键字参数绑定错误；对应的位置参数调用均成功。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/mindspore.Tensor.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.renorm.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.float_power.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.multiply.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.equal.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
