# [Bug]: PyNative CPU 中同一 Cell 混合物化不同 `out_int32` 的 `ops.searchsorted` 调用会失败

## 🐞 问题详细描述

单独使用默认输出或 `out_int32=True` 均成功；但在同一 Cell 中按任一顺序物化两种返回 dtype 时，PyNative 抛出类型/内核相关异常。相同组合在 Graph 模式正常返回 int64 与 int32 结果。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, sorted_sequence, values):
        return (ops.searchsorted(sorted_sequence, values),
                ops.searchsorted(sorted_sequence, values, out_int32=True))

sorted_sequence = ms.Tensor([[1, 3, 5], [2, 4, 6]], ms.int32)
values = ms.Tensor([[0, 3, 7], [1, 5, 9]], ms.int32)
print(tuple(item.asnumpy() for item in Net()(sorted_sequence, values)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 14, in <module>
    print(tuple(item.asnumpy() for item in Net()(sorted_sequence, values)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "reproducer.py", line 14, in <genexpr>
    print(tuple(item.asnumpy() for item in Net()(sorted_sequence, values)))
                ^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/common/tensor.py", line 1090, in asnumpy
    return TensorPy_.asnumpy(self)
           ^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: For 'SearchSorted', the dimension of `v` and output must be equal, but got the dimension of `v` 24 and the dimension of output 24

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/kernel/cpu/native/searchsorted_cpu_kernel.cc:107 CheckParam
```

### 预期结果

每次调用的 `out_int32` 应独立决定输出 dtype；同一 Cell 连续物化不同配置不应污染另一调用或导致失败。

## 详细的环境信息描述

- MindSpore：`2.10.0`，PyPI official CPU binary wheel
- Python：`3.11.15`
- 操作系统：`Ubuntu 24.04.4 LTS`，WSL2
- Linux kernel：`6.6.87.2-microsoft-standard-WSL2`
- CPU：`Intel(R) Core(TM) i7-14700K`，x86_64
- Device target：`CPU`
- 执行模式：`PyNative`
- 系统 GCC：`13.3.0`；MindSpore 使用预编译二进制包，未在本机编译
- GPU/Ascend 驱动：不适用
- 安装方式：Python 虚拟环境内安装 MindSpore 2.10.0 wheel

## 补充说明

- 问题定位：失败只出现在 PyNative 的混合物化组合；单调用和 Graph 组合均通过。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
