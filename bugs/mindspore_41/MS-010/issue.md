# [Bug]: Graph 模式中无参数 `Tensor.argmax()` 错误采用最后一维语义

## 🐞 问题详细描述

PyNative 的 `x.argmax()` 返回全 Tensor 的标量索引 `3`；Graph 的同一调用返回按最后一维计算的向量 `[0, 0]`。Graph 中显式 `dim=None` 与 `ops.argmax(x)` 均返回标量 `3`。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.argmax()

x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
result = Net()(x)
print("shape:", result.shape)
print("value:", result)
```

### 实际结果

进程退出码：`0`。实际输出：

```text
shape: (2,)
value: [0 0]
```

### 预期结果

根据 Tensor API 文档，无参数调用等价于 `axis/dim=None`，Graph 应返回全 Tensor 的标量索引 `3`。

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

- 问题定位：这是静默的值与形状错误，进程退出码为 0；显式 None 和功能式控制给出正确标量。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
