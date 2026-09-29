# [Bug]: Graph 中 `ops.shape` 对 `hann_window/kaiser_window` 输出返回标量 Tensor

## 🐞 问题详细描述

窗口结果本身都是 shape `(5,)`。PyNative 的 `ops.shape` 返回 tuple；Graph 对 hann/kaiser 返回值却产生 int64 标量 Tensor `5`，而 hamming 控制仍返回 tuple，Tensor `.shape` 也始终正确。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self):
        window = ops.hann_window(5, dtype=ms.float32)
        return ops.shape(window), window.shape

ops_shape, property_shape = Net()()
print("ops.shape:", type(ops_shape).__name__, ops_shape)
print("Tensor.shape:", type(property_shape).__name__, property_shape)
```

### 实际结果

进程退出码：`0`。实际输出：

```text
ops.shape: Tensor 5
Tensor.shape: tuple (5,)
```

### 预期结果

`ops.shape` 应按文档在所有入口和模式返回 `tuple[int]`，这里应为 `(5,)`。

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

- 问题定位：这是退出码为 0 的返回类型错误，且被定位到 Graph 中特定 Python 组合窗口输出的 shape 推断。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.shape.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.hamming_window.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.hann_window.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.kaiser_window.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
