# [Bug]: `ops.log_softmax` 在 CPU 上静默接受 float64，但按较低精度计算

## 🐞 问题详细描述

在 CPU PyNative 模式下，`ops.log_softmax` 接受 float64 输入并返回 float64 Tensor，但与 NumPy float64 稳定公式相比，最大绝对误差为 `2.2237522419032985e-06`，结果呈现明显的较低精度计算特征。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x_np = np.array([[20.0, 19.0, 18.0], [-20.0, -21.0, -22.0]], np.float64)
actual = ops.log_softmax(ms.Tensor(x_np), axis=1).asnumpy()
shifted = x_np - x_np.max(axis=1, keepdims=True)
expected = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
print("dtype:", actual.dtype)
print("max_abs_error:", np.max(np.abs(actual - expected)))
```

### 实际结果

进程退出码：`0`。实际输出：

```text
dtype: float64
max_abs_error: 2.2237522419032985e-06
```

### 预期结果

对于文档未支持的 float64 输入，应按文档抛出 `TypeError`；如果选择接受该输入，则计算精度应与返回的 float64 类型一致，不应以 float64 元数据包装较低精度结果。

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

- 问题定位：复现器同时计算 NumPy float64 参考值；两种模式得到完全相同的异常误差，且返回 dtype 均为 float64。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
