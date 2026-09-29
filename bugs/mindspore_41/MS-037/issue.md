# [Bug]: `ops.MatrixDiagV3` primitive 拒绝文档允许的 Python int 对角参数

## 🐞 问题详细描述

直接 primitive 在两模式对主对角和偏移对角的 Python int `k` 均报输入必须为 Tensor；Tensor 参数控制和高层 `ops.matrix_diag` 的同一 int 参数均成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
diagonal = ms.Tensor([1, 2, 3], ms.float32)
print(ops.MatrixDiagV3()(diagonal, 0, 3, 3, 0))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 6, in <module>
    print(ops.MatrixDiagV3()(diagonal, 0, 3, 3, 0))
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
TypeError: For primitive[MatrixDiagV3], the input[1] should be a Tensor, but got Int64.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/utils/check_convert_utils.cc:1548 CheckArgsType
```

### 预期结果

公开 primitive 应接受文档声明的 Python int 参数，或在调用边界转换为标量 Tensor。

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

- 问题定位：Tensor primitive 和功能式 int 控制均成功，失败局限于直接 MatrixDiagV3 的 Python 标量处理。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.MatrixDiagV3.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.matrix_diag.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
