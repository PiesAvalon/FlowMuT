# [Bug]: CPU 上 `ops.matrix_diag_part` 拒绝文档允许的 Python int `k`

## 🐞 问题详细描述

文档将 `k` 声明为 `Union[int, Tensor]`，但 PyNative 与 Graph 的 Python int `k` 均报 MatrixDiagPartV3 输入必须为 Tensor；Tensor `k` 控制正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
print(ops.matrix_diag_part(x, 0, ms.Tensor(0, ms.int32)))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 6, in <module>
    print(ops.matrix_diag_part(x, 0, ms.Tensor(0, ms.int32)))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/function/array_func.py", line 3638, in matrix_diag_part
    return matrix_diag_part_v3(x, k, padding_value)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 397, in __call__
    return _run_op(self, self.name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 1003, in _run_op
    res = _pynative_executor.run_op_async(obj, op_name, args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 1805, in run_op_async
    return self._executor.run_op_async(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: For primitive[MatrixDiagPartV3], the input[1] should be a Tensor, but got Int64.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/utils/check_convert_utils.cc:1548 CheckArgsType
```

### 预期结果

功能式 API 应接受文档声明的 Python int `k`，或在进入 primitive 前转换为标量 Tensor。

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

- 问题定位：相同矩阵与参数只有 Python int 形式失败，Tensor 形式在两模式都成功。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.matrix_diag_part.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
