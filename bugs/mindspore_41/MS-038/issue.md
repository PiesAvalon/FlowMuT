# [Bug]: `NonMaxSuppressionWithOverlaps` 拒绝文档允许的标量参数

## 🐞 问题详细描述

两模式下 Python int `max_output_size` 报必须为 Tensor，Python float 阈值触发 `GetShapeVector()` 未实现；三个 0-D Tensor 参数的控制正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
overlaps = ms.Tensor([[1.0, 0.6, 0.1], [0.6, 1.0, 0.2],
                      [0.1, 0.2, 1.0]], ms.float32)
scores = ms.Tensor([0.9, 0.8, 0.7], ms.float32)
print(ops.NonMaxSuppressionWithOverlaps()(
    overlaps, scores, 2, ms.Tensor(0.5), ms.Tensor(0.0)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 8, in <module>
    print(ops.NonMaxSuppressionWithOverlaps()(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 397, in __call__
    return _run_op(self, self.name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 1003, in _run_op
    res = _pynative_executor.run_op_async(obj, op_name, args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 1805, in run_op_async
    return self._executor.run_op_async(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: The primitive[NonMaxSuppressionWithOverlaps]'s input arguments[max_output_size] must be all tensor and those type must be same. But got input argument[max_output_size]:Int64
Valid type list: {Tensor[Int32]}.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/utils/check_convert_utils.cc:787 CheckTensorTypeSame
```

### 预期结果

primitive 应按文档接受 `Number.int/Number.float`，并与等价 0-D Tensor 参数产生相同 selected indices。

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

- 问题定位：CPU 算子由 all-Tensor 控制验证，失败集中在公开声明的 Python 标量参数。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.NonMaxSuppressionWithOverlaps.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
