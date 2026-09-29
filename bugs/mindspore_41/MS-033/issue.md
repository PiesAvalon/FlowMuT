# [Bug]: Graph 模式 `ops.csr_softmax` 因 Closure/free-variable 内部限制失败

## 🐞 问题详细描述

相同合法 CSR 输入在 PyNative 返回 CSRTensor，在 Graph 报 `Map operator don't support Closure with free variable yet`；CSR tuple 与 COO 控制正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return ops.csr_softmax(x.to_csr(), ms.float32).to_tuple()

x = ms.Tensor([[0, 2, 0], [3, 0, 4]], ms.float32)
print(Net()(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[WARNING] ME(105758:139649231009600,MainProcess):2026-08-03-22:54:23.640.00 [mindspore/common/_decorator.py:69] 'DenseToCSRSparseMatrix' is deprecated from version 2.8.0 and will be removed in a future version.
[WARNING] ME(105758:139649231009600,MainProcess):2026-08-03-22:54:23.107.000 [mindspore/common/_decorator.py:69] 'SparseMatrixSoftmax' is deprecated from version 2.8.0 and will be removed in a future version.
Traceback (most recent call last):
  File "reproducer.py", line 12, in <module>
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
RuntimeError: The Map operator don't support Closure with free variable yet.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/frontend/operator/composite/map.cc:304 NormalizeArgs

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:9, 15~54
        return ops.csr_softmax(x.to_csr(), ms.float32).to_tuple()
               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/sparse_func.py:810~811, 4~108
    if not isinstance(logits, CSRTensor):
# 2 In file reproducer.py:9, 15~30
        return ops.csr_softmax(x.to_csr(), ms.float32).to_tuple()
               ^~~~~~~~~~~~~~~
# 3 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/sparse_func.py:813, 28~82
    logits_batch_pointers = make_tensor([0, logits.values.shape[0]], mstype.int32)
                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:189~190, 4~50
    if data_shape:
# 5 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:192~193, 4~101
    if not isinstance(a, (list, tuple, int, float, bool)):
# 6 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:198~199, 4~43
    if isinstance(a, int):
# 7 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:201~210, 4~90
    if isinstance(a, (list, tuple)):
# 8 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:202~203, 8~35
        if not a:
# 9 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:205, 12~35
        a = _deep_list(a, dim_size)
            ^~~~~~~~~~~~~~~~~~~~~~~
# 10 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:122~123, 4~71
    if isinstance(array_like, (list, tuple)):
# 11 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_constexpr_utils.py:123, 20~70
        return list(map(lambda x: _deep_list(x, dim_size), array_like))
                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

文档列出 CPU 支持，公开 `csr_softmax` 应在 Graph 返回与 PyNative 一致的 CSRTensor，内部实现不应暴露 closure 限制。

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

- 问题定位：PyNative 与 Graph 稀疏基础控制证明输入有效，错误指向 csr_softmax Python 实现中的 free variable 路径。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore.ops.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
