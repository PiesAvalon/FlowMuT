# [Bug]: Graph 模式中 CPU `CSRMV` 无法处理合法 dense 向量

## 🐞 问题详细描述

shape `(2,3)` 的 CSR 与 shape `(3,1)` dense 向量在 PyNative 返回 `(2,1)` Tensor，在 Graph 的 Tensor method 和 ops 入口均报 `Parsed metadata of op[CSRMV] failed`；CSR tuple/abs/add 控制成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, vector):
        return x.to_csr().mv(vector)

x = ms.Tensor([[0, 2, 0], [3, 0, 4]], ms.float32)
vector = ms.Tensor([[1], [2], [3]], ms.float32)
print(Net()(x, vector))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[WARNING] ME(105594:134821646948160,MainProcess):2026-08-03-22:54:18.711.000 [mindspore/common/_decorator.py:69] 'DenseToCSRSparseMatrix' is deprecated from version 2.8.0 and will be removed in a future version.
[WARNING] SESSION(105594,7a9e9c5fd740,python3.11):2026-08-03-22:54:18.819.101 [mindspore/ccsrc/backend/common/kernel_graph/anf_runtime_algorithm.cc:2955] ParseMetadata] The size of inputs in OpIOInfo should be great than real input. Inputs size in OpIOInfo:4, real input num: 6, node: Default/CSRMV-op0
Traceback (most recent call last):
  File "reproducer.py", line 12, in <module>
    print(Net()(x, vector))
          ^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1409, in __call__
    out = self.compile_and_run(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1800, in compile_and_run
    self.compile(*args, **kwargs)
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1782, in compile
    _cell_graph_executor.compile(self, *compile_args, phase=self.phase,
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 2354, in compile
    result = self._graph_executor.compile(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Parsed metadata of op[CSRMV] failed.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/cpu/kernel_executor/kernel_select/kernel_select_cpu.cc:584 UpdateCustomKernelBuildInfo
```

### 预期结果

文档列出 CPU 支持，Graph 应接受维度兼容的 dense 向量并返回与 PyNative 一致的 dense Tensor。

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

- 问题定位：维度由 PyNative 结果验证，CSR 基础控制通过，失败定位到 CSRMV CPU kernel metadata。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/mindspore.CSRTensor.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore.ops.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
