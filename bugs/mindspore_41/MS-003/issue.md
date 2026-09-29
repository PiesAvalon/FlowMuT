# [Bug]: CPU `softplus` 和 `mish` 声明支持 float64，但返回值呈现较低精度

## 🐞 问题详细描述

在 CPU PyNative 模式下，`ops.softplus` 和 `ops.mish` 都返回 float64 Tensor，但与独立的 NumPy float64 公式相比，最大绝对误差分别为 `1.7685137105871718e-06` 和 `4.392856105539522e-07`。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x_np = np.array([-8.25, -1.75, 0.25, 3.5, 8.75], np.float64)
x = ms.Tensor(x_np)
softplus = ops.softplus(x).asnumpy()
mish = ops.mish(x).asnumpy()
softplus_ref = np.logaddexp(0.0, x_np)
mish_ref = x_np * np.tanh(softplus_ref)
print("softplus dtype/error:", softplus.dtype, np.max(np.abs(softplus-softplus_ref)))
print("mish dtype/error:", mish.dtype, np.max(np.abs(mish-mish_ref)))
```

### 实际结果

进程退出码：`0`。实际输出：

```text
softplus dtype/error: float64 1.7685137105871718e-06
mish dtype/error: float64 4.392856105539522e-07
```

### 预期结果

既然 CPU 文档声明支持 float64，计算精度应与 float64 契约一致，或明确拒绝该 dtype，而不是返回 float64 元数据但保留较低精度误差。

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

- 问题定位：复现器使用稳定的独立 float64 公式作为参考，并在两个执行模式得到相同误差。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
