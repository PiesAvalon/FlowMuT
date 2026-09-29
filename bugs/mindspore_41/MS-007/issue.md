# [Bug]: 布尔掩码 Tensor 赋值在 PyNative 与 Graph 使用不同的 RHS 形状规则

## 🐞 问题详细描述

对同一 `(2,3)` Tensor 和布尔掩码，选中元素形状 RHS 在 PyNative 成功、Graph 广播失败；完整 `(2,3)` RHS 则在 Graph 成功、PyNative 失败。标量 RHS 在两模式都成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, mask, selected_values):
        y = x + 0
        y[mask] = selected_values
        return y

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
mask = ms.Tensor([[True, False, True], [False, True, False]])
selected_values = ms.Tensor([10, 20, 30], ms.float32)
print(Net()(x, mask, selected_values))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
[ERROR] ANALYZER(104170,7f50d8cba740,python3.11):2026-08-03-22:53:32.667.736 [mindspore/ccsrc/frontend/jit/ps/static_analysis/evaluator.cc:764] Run] Primitive: <PrimitiveFunctionEvaluator_PrimitiveFunction_BroadcastTo> infer failed, failed info: For 'BroadcastTo', in order to broadcast, each dimension pair must be equal or input dimension is 1 or target dimension is -1. But got x_shape: [const vector]{3, 1}, target shape: [const vector]{2, 3}.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/infer/ops_func_impl//broadcast_to.cc:67 CheckShapeValid

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:9, 8~33
        y[mask] = selected_values
        ^~~~~~~~~~~~~~~~~~~~~~~~~
# 1 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py:1355~1356, 4~83
    if tensor_dtype == const_utils.INT_:
# 2 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py:1358, 11~80
    return _tensor_setitem_by_bool_tensor_with_tensor(data, index, value_tensor)
           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py:1346, 12~45
    value = F.broadcast_to(value, data.shape)
            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:1757, 11~42
    return broadcast_to_impl(input, shape)
           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
Traceback (most recent call last):
  File "reproducer.py", line 15, in <module>
    print(Net()(x, mask, selected_values))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
ValueError: For 'BroadcastTo', in order to broadcast, each dimension pair must be equal or input dimension is 1 or target dimension is -1. But got x_shape: [const vector]{3, 1}, target shape: [const vector]{2, 3}.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/infer/ops_func_impl//broadcast_to.cc:67 CheckShapeValid

----------------------------------------------------
- The Traceback of Net Construct Code:
----------------------------------------------------
# 0 In file reproducer.py:9, 8~33
        y[mask] = selected_values
        ^~~~~~~~~~~~~~~~~~~~~~~~~
# 1 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py:1355~1356, 4~83
    if tensor_dtype == const_utils.INT_:
# 2 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py:1358, 11~80
    return _tensor_setitem_by_bool_tensor_with_tensor(data, index, value_tensor)
           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py:1346, 12~45
    value = F.broadcast_to(value, data.shape)
            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4 In file <python_env>/lib/python3.11/site-packages/mindspore/ops/auto_generate/gen_ops_def.py:1757, 11~42
    return broadcast_to_impl(input, shape)
           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 (See file '<temporary_workdir>/rank_0/om/analyze_fail.ir' for more details. Get instructions about `analyze_fail.ir` at https://www.mindspore.cn/search?inputValue=analyze_fail.ir)
```

### 预期结果

布尔掩码赋值的 RHS 形状语义应在两种执行模式保持一致；若某种形状不支持，应给出一致且可理解的校验错误。

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

- 问题定位：同一索引操作在两模式呈现互相相反的 RHS 接受规则，标量控制排除了赋值路径整体不可用。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
