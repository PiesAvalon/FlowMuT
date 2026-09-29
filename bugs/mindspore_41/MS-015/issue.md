# [Bug]: `Tensor.scatter_div` 在 Graph 模式因 `standard_method` 拼写错误而失败

## 🐞 问题详细描述

PyNative Tensor 方法正常；Graph 中 `Tensor.scatter_div` 报 `standard_method` 不存在 `tensor_scatter_div`，并提示只有拼错的 `tensor_sactter_div`。相同输入的 `ops.tensor_scatter_div` 及其他 scatter Tensor 方法正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, indices, updates):
        return x.scatter_div(indices, updates)

x = ms.Tensor([[1, 4, 2], [3, 2, 5]], ms.float32)
indices = ms.Tensor([[0, 1], [1, 2]], ms.int32)
updates = ms.Tensor([2, 4], ms.float32)
print(Net()(x, indices, updates))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[ERROR] ANALYZER(104681,736b750cf740,python3.11):2026-08-03-22:53:44.769.370 [mindspore/ccsrc/frontend/jit/ps/static_analysis/async_eval_result.cc:74] HandleException] Exception happened, check the information as below.
AttributeError: module 'mindspore.graph._parse.standard_method' has no attribute 'tensor_scatter_div'

# 0 In file reproducer.py:8, 15~28
        return x.scatter_div(indices, updates)
               ^~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
Traceback (most recent call last):
  File "reproducer.py", line 13, in <module>
    print(Net()(x, indices, updates))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
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
AttributeError: module 'mindspore.graph._parse.standard_method' has no attribute 'tensor_scatter_div'. Did you mean: 'tensor_sactter_div'?
```

### 预期结果

Graph 应通过正确注册名调用 `tensor_scatter_div`，返回与 PyNative 和功能式入口一致的结果。

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

- 问题定位：异常直接暴露注册名拼写错误，且同一底层算子的功能式入口成功。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
