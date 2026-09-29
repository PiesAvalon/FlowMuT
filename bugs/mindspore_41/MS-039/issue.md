# [Bug]: `ops.CountNonZero(dims=...)` 的构造参数未接入实际 primitive 调用

## 🐞 问题详细描述

本地公开签名与示例使用 `CountNonZero(dims=[1])(x)`，但 PyNative 报期望 2 个输入而只得到 1 个；Graph 将 dim 降低为 None 后报 CPU 不支持。功能式 `ops.count_nonzero(x, axis=[1])` 在两模式成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[0, 1, 2], [3, 0, 4]], ms.int64)
print(ops.CountNonZero(dims=[1])(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 6, in <module>
    print(ops.CountNonZero(dims=[1])(x))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 397, in __call__
    return _run_op(self, self.name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 1003, in _run_op
    res = _pynative_executor.run_op_async(obj, op_name, args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 1805, in run_op_async
    return self._executor.run_op_async(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: For Operator[CountNonZero], the inputs number should be 2 but got 1.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/pynative/utils/pynative_utils.cc:860 ParseOpInputByOpDef
```

### 预期结果

constructor 的 dims 属性应按公开示例生效，或统一为明确的 call-time dim 签名；当前不能存在两套冲突实现并导致有效示例失败。

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

- 问题定位：相同 axis 由功能式 CPU 路径成功验证，失败源于 constructor attribute 与生成 primitive call-time input 的冲突。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.count_nonzero.html
- https://atomgit.com/mindspore/mindspore/blob/v2.9.0/mindspore/python/mindspore/ops/function/math_func.py

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
