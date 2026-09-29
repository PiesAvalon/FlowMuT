# [Bug]: CPU 上 `Tensor.add_` 的语句形式与返回值形式语义不一致

## 🐞 问题详细描述

该原地 API 在 CPU 未宣称支持，但调用并未一致拒绝：PyNative 语句形式静默不修改，返回值形式产生加法结果；Graph 语句与返回形式都产生结果，paired 场景还显示别名/重复更新差异。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, delta):
        statement = x + 0
        statement.add_(delta)
        returned = x + 0
        return statement, returned.add_(delta)

x = ms.Tensor([[1, 2], [3, 4]], ms.float32)
statement, returned = Net()(x, ms.Tensor(2, ms.float32))
print("statement form:", statement)
print("return-value form:", returned)
```

### 实际结果

进程退出码：`0`。实际输出：

```text
statement form: [[1. 2.]
 [3. 4.]]
return-value form: [[3. 4.]
 [5. 6.]]
```

### 预期结果

不支持的 CPU 原地 API 应被明确且一致地拒绝；若允许执行，语句形式、返回值形式及两种模式应保持一致的修改和别名语义。

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

- 问题定位：成功调用后的行为与文档描述的语义不一致。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
