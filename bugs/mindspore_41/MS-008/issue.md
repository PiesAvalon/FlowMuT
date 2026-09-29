# [Bug]: Graph 模式静默忽略语句形式的 `Tensor.masked_fill_`

## 🐞 问题详细描述

CPU 文档未宣称支持原地操作。PyNative 在物化结果时明确报未注册内核；Graph 的返回值形式却得到填充值，而语句形式 `y.masked_fill_(...); return y` 静默返回未修改的原 Tensor。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, mask):
        y = x + 0
        y.masked_fill_(mask, ms.Tensor(9.0, ms.float32))
        return y

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
mask = ms.Tensor([[True, False, True], [False, True, False]])
print(Net()(x, mask))
```

### 实际结果

进程退出码：`0`。实际输出：

```text
[[1. 2. 3.]
 [4. 5. 6.]]
```

### 预期结果

Graph 不应静默忽略原地语句：要么与返回值形式一致地产生填充结果，要么像 PyNative 一样明确拒绝不支持的 CPU 原地操作。

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

- 问题定位：Graph 语句形式的原地操作静默无效，未返回明确的不支持错误。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
