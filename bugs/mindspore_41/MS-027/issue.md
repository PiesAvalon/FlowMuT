# [Bug]: Graph 模式拒绝 `Tensor.col2im` 的公开关键字参数

## 🐞 问题详细描述

Tensor 方法的位置参数形式在 Graph 成功，关键字和混合形式失败；相同关键字的 `ops.col2im` 在 Graph 成功，PyNative Tensor 关键字也成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.col2im(output_size=ms.Tensor([4, 4], ms.int32),
                        kernel_size=[2, 2], dilation=[1, 1],
                        padding_value=[0, 0], stride=[1, 1])

x = ms.Tensor(np.arange(36, dtype=np.float32).reshape(1, 4, 9))
print(Net()(x.expand_dims(1)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 14, in <module>
    print(Net()(x.expand_dims(1)))
          ^^^^^^^^^^^^^^^^^^^^^^^
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
RuntimeError: Got an unexpected keyword argument 'output_size'

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/ir/func_graph_extends.cc:161 GenerateKwParams

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:9~11, 15~60
        return x.col2im(output_size=ms.Tensor([4, 4], ms.int32),
               ^
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

Graph Tensor wrapper 应接受公开签名中的 `output_size`、`kernel_size`、`dilation`、`padding_value` 和 `stride` 关键字。

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

- 问题定位：底层算子、输入和位置参数均由控制验证，失败局限于 Graph Tensor 方法关键字绑定。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.col2im.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.col2im.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
