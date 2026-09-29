# [Bug]: Graph 模式 `CSRDiv` 返回 CSR 结果时触发 TupleUnfold 内部错误

## 🐞 问题详细描述

`ops.csr_div(csr, dense)` 和 `csr / dense` 在 PyNative 可执行，在 Graph 均因 `Tuple to TupleUnfold pattern...` 内部返回处理错误失败；CSR dtype/unary 控制成功。

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
        return ops.csr_div(csr, ops.ones_like(x))

x = ms.Tensor([[0, 2, 0], [3, 0, 4]], ms.float32)
print(Net()(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[WARNING] ME(105543:133419055585088,MainProcess):2026-08-03-22:54:15.980.00 [mindspore/common/_decorator.py:69] 'DenseToCSRSparseMatrix' is deprecated from version 2.8.0 and will be removed in a future version.
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
RuntimeError: Tuple to TupleUnfold pattern should have TupleGetItem as user node, but got Default/CSRDiv-op0, @kernel_graph0:res_values{[0]: ValueNode<Primitive> CSRDiv, [1]: indptr, [2]: indices, [3]: values, [4]: ValueNode<ValueTuple> (2, 3), [5]: ValueNode<Tensor> Tensor(shape=[2, 3], dtype=Float32, value=
[[ 1.00000000e+00  1.00000000e+00  1.00000000e+00]
 [ 1.00000000e+00  1.00000000e+00  1.00000000e+00]])}
----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------

# In file reproducer.py:8~10, 4~49
    def construct(self, x):

# In file reproducer.py:10, 15~26
        return ops.csr_div(csr, ops.ones_like(x))
               ^~~~~~~~~~~

# In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/sparse_func.py:193~221
def csr_div(x: CSRTensor, y: Tensor) -> Tensor:

# In file <python_env>/lib/python3.11/site-packages/mindspore/ops/function/sparse_func.py:220, 17~77
    res_values = _csr_ops.CSRDiv()(x.indptr, x.indices, x_values, x.shape, y)
                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/backend/common/pass/insert_type_transform_op.cc:722 ProcessTupleToTupleUnfold
```

### 预期结果

Graph 应按实际公开路径返回与 PyNative 一致的 CSRTensor，或根据明确文档契约返回 values Tensor，不能触发内部 TupleUnfold 错误。

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

- 问题定位：PyNative 可执行两条入口，Graph 的其他 CSR 控制正常，失败局限于 CSRDiv 结果展开。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/mindspore.CSRTensor.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore.ops.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
