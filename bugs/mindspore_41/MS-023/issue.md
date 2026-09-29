# [Bug]: CPU 上 `ops.round/Tensor.round` 拒绝文档支持的非零 `decimals`

## 🐞 问题详细描述

文档描述并示例正负 `decimals`，但 PyNative 与 Graph 的 `decimals=1` 和 `decimals=-1` 均报 CPU Round 只支持 0；默认 `decimals=0` 成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([1.25, 2.75, -3.15], ms.float32)
print(ops.round(x, decimals=1))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 6, in <module>
    print(ops.round(x, decimals=1))
  File "<python_env>/lib/python3.11/site-packages/mindspore/common/tensor.py", line 528, in __str__
    if not self._data_ptr():
           ^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/common/tensor.py", line 4114, in _data_ptr
    return TensorPy_._data_ptr(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: For Round only support decimals equal 0, but got decimals equal 1

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/kernel/cpu/native/round_cpu_kernel.cc:50 LaunchKernel
```

### 预期结果

CPU 应按文档实现正负 decimals，或在文档和签名中明确限制，而不是接受参数后由内核拒绝所有非零值。

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

- 问题定位：默认 round 在 CPU 正常，失败只由文档允许的非零 decimals 触发。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.round.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
