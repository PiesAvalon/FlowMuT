# [Bug]: `Tensor.max/min(dim)` 的位置参数重载在 PyNative 与 Graph 模式返回结构不一致

## 🐞 问题详细描述

同一个 Cell 调用 `Tensor.max(0)` 和 `Tensor.min(0)` 时，PyNative 返回 `(values, indices)`，Graph 却只返回 values Tensor，静默丢失 indices 并改变外层返回结构。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.max(0), x.min(0)

x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.float32)
result = Net()(x)
print("outer_type:", type(result).__name__)
print("items:", len(result))
print(result)
```

### 实际结果

进程退出码：`0`。实际输出：

```text
outer_type: tuple
items: 2
(Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  5.00000000e+00,  4.00000000e+00]), Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  1.00000000e+00,  2.00000000e+00]))
```

### 预期结果

Graph 应与公开 API 约定及 PyNative 一致，为每次按维度归约返回 `(values, indices)`，并保留 indices 的整数 dtype。

## 详细的环境信息描述

- MindSpore：`2.10.0`，PyPI official CPU binary wheel
- Python：`3.11.15`
- 操作系统：`Ubuntu 24.04.4 LTS`，WSL2
- Linux kernel：`6.6.87.2-microsoft-standard-WSL2`
- CPU：`Intel(R) Core(TM) i7-14700K`，x86_64
- Device target：`CPU`
- 执行模式：`PyNative、Graph`
- 系统 GCC：`13.3.0`；MindSpore 使用预编译二进制包，未在本机编译
- GPU/Ascend 驱动：不适用
- 安装方式：Python 虚拟环境内安装 MindSpore 2.10.0 wheel

## 补充说明

- 问题定位：两种模式均成功执行，但当前输出直接显示返回嵌套结构与 indices 是否存在的差异。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
