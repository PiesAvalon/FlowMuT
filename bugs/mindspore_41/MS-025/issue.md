# [Bug]: CPU 分段归约对文档支持的 Tensor `num_segments` 处理不一致

## 🐞 问题详细描述

`unsorted_segment_min/max` 在 PyNative 与 Graph 均拒绝 0-D int32/int64 Tensor `num_segments`；`prod` 在 PyNative 拒绝、Graph 成功；同族 `sum` 接受 Tensor，三者的 Python int 形式也成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[1, 2], [3, 4], [5, 6]], ms.float32)
segment_ids = ms.Tensor([0, 1, 1], ms.int32)
print(ops.unsorted_segment_min(x, segment_ids, ms.Tensor(2, ms.int32)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 7, in <module>
    print(ops.unsorted_segment_min(x, segment_ids, ms.Tensor(2, ms.int32)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py", line 3958, in unsorted_segment_min
    return unsorted_segment_min_(x, segment_ids, num_segments)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 397, in __call__
    return _run_op(self, self.name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 1003, in _run_op
    res = _pynative_executor.run_op_async(obj, op_name, args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 1805, in run_op_async
    return self._executor.run_op_async(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: For 'UnsortedSegmentMin', num_segments value must be greater than 0, but got: -1.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ops/infer//unsorted_segment_arithmetic.cc:174 UnsortedSegmentArithmeticInferShape
```

### 预期结果

`min/max/prod` 应按文档一致接受正的 0-D Tensor `num_segments`，并与 int 形式及 `sum` 的族内契约一致。

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

- 问题定位：`prod` 在 Graph 模式下成功；`sum` 与 Python int 对照排除一般 CPU 或输入问题。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.unsorted_segment_min.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.unsorted_segment_max.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.unsorted_segment_prod.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.unsorted_segment_sum.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
