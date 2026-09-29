# [Bug]: CPU `CSRGather` 在 Graph 模式失败，而 PyNative 可返回 gathered values

## 🐞 问题详细描述

由有效 dense Tensor 转换得到的 CSR 元数据在 PyNative `csr_gather` 返回 shape `(3,)` 的非零值，在 Graph 报 CSRGather metadata 解析失败；CSR tuple 与 COO 控制成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        csr = x.to_csr()
        return ops.csr_gather(csr.indptr, csr.indices, x, x.shape)

x = ms.Tensor([[0, 2, 0], [3, 0, 4]], ms.float32)
print(Net()(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[WARNING] ME(105682:139968758847296,MainProcess):2026-08-03-22:54:21.393.000 [mindspore/common/_decorator.py:69] 'DenseToCSRSparseMatrix' is deprecated from version 2.8.0 and will be removed in a future version.
[WARNING] SESSION(105682,7f4d04269740,python3.11):2026-08-03-22:54:21.481.490 [mindspore/ccsrc/backend/common/kernel_graph/anf_runtime_algorithm.cc:2955] ParseMetadata] The size of inputs in OpIOInfo should be great than real input. Inputs size in OpIOInfo:3, real input num: 5, node: Default/CSRGather-op0
Traceback (most recent call last):
  File "reproducer.py", line 13, in <module>
    print(Net()(x))
          ^^^^^^^^
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
RuntimeError: Parsed metadata of op[CSRGather] failed.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/cpu/kernel_executor/kernel_select/kernel_select_cpu.cc:584 UpdateCustomKernelBuildInfo
```

### 预期结果

文档列出 CPU 支持，Graph 应使用相同合法 CSR 元数据返回与 PyNative 一致的 gathered Tensor。

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

- 问题定位：输入由 PyNative 和 Graph CSR tuple 控制验证，失败局限于 Graph CSRGather CPU metadata。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore.ops.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
