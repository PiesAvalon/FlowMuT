# [Bug]: float64 输入下 `ops.leaky_relu` 的 `alpha` 被先按 float32 舍入

## 🐞 问题详细描述

float64 输入配合 Python 浮点数 `alpha=0.2` 时，PyNative 和 Graph 的输出均与直接 float64 公式相差 `3.725290298461914e-09`，误差恰好对应先把 `alpha` 转成 float32 再扩展到 float64。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x_np = np.linspace(-1.25, 1.1, 6, dtype=np.float64).reshape(2, 3)
actual = ops.leaky_relu(ms.Tensor(x_np), 0.2).asnumpy()
expected = np.where(x_np >= 0, x_np, np.float64(0.2) * x_np)
print("dtype:", actual.dtype)
print("max_abs_error:", np.max(np.abs(actual - expected)))
```

### 实际结果

进程退出码：`0`。实际输出：

```text
dtype: float64
max_abs_error: 3.725290298461914e-09
```

### 预期结果

对 float64 输入，Python `alpha` 应直接按 float64 转换并参与计算，不能先经 float32 舍入。

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

- 问题定位：两种模式误差完全一致，且与已定位的 `scalar_to_tensor(alpha)` 默认 float32 转换路径吻合。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
