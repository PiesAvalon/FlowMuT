# [Bug]: `Tensor.reverse_sequence` 的默认参数不可用，且 Graph 默认参数绑定与 PyNative 不一致

## 🐞 问题详细描述

公开 Tensor 签名把 `seq_dim` 和 `batch_dim` 都默认为 0，默认调用在 PyNative 因二者相等而报错；Graph 默认调用触发内部 unexpected error。显式合法 `seq_dim=1` 的 Tensor 与 ops 路径在两模式均成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
seq_lengths = ms.Tensor([2, 3], ms.int32)
print(x.reverse_sequence(seq_lengths))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 6, in <module>
    print(x.reverse_sequence(seq_lengths))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/common/tensor.py", line 1595, in reverse_sequence
    return tensor_operator_registry.get("reverse_sequence")(self, seq_lengths, seq_dim, batch_dim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py", line 1695, in reverse_sequence
    return _get_cache_prim(P.ReverseSequence)(seq_dim=seq_dim, batch_dim=batch_dim)(x, seq_lengths)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 397, in __call__
    return _run_op(self, self.name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 1003, in _run_op
    res = _pynative_executor.run_op_async(obj, op_name, args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 1805, in run_op_async
    return self._executor.run_op_async(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: For 'ReverseSequence', the 'batch_dim' should be != seq_dim: 0, but got 0.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/infer//reverse_sequence.cc:91 ReverseSequenceInferShape
```

### 预期结果

Tensor API 不应提供必然违反算子约束的默认参数；Graph 与 PyNative 还应使用一致的签名绑定和清晰校验错误。

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

- 问题定位：问题同时包含不合法的公开默认值和 Graph 默认参数绑定差异；显式合法参数控制排除了 CPU 内核问题。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.reverse_sequence.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.reverse_sequence.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
