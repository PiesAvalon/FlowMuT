# [Bug]: `ops.IndexFill` primitive 拒绝文档允许的标量 `value`

## 🐞 问题详细描述

直接 `IndexFill` primitive 在两模式拒绝 Python int/float value，称必须为 Tensor；Tensor value 控制正常，功能式 `ops.index_fill` 和 Tensor 方法对同一标量也正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
indices = ms.Tensor([0, 2], ms.int32)
print(ops.IndexFill()(x, 1, indices, 9))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 7, in <module>
    print(ops.IndexFill()(x, 1, indices, 9))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 397, in __call__
    return _run_op(self, self.name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/primitive.py", line 1003, in _run_op
    res = _pynative_executor.run_op_async(obj, op_name, args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 1805, in run_op_async
    return self._executor.run_op_async(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: For Primitive[IndexFill], the type of input argument[value] must be Tensor but got Int64.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/utils/check_convert_utils.cc:881 CheckTensorTypeValid
```

### 预期结果

primitive 公开文档列出 `bool/int/float/Tensor`，应接受标量 value 或由 wrapper 统一转换，不能与文档及其他入口矛盾。

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

- 问题定位：CPU 与底层 IndexFill 可由 Tensor value 控制运行，失败仅是直接 primitive 的标量参数契约。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.IndexFill.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.index_fill.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
