# [Bug]: `ops.polar` 和 `Polar` 拒绝文档允许的 Python float 参数

## 🐞 问题详细描述

功能式和直接 primitive 在两模式均拒绝 Python float abs/angle，并声称有效签名只接受 Tensor；同形状 Tensor 和成对 0-D Tensor 控制正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
absolute = ms.Tensor([1, 2, 3], ms.float32)
print(ops.polar(absolute, 2.0))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 6, in <module>
    print(ops.polar(absolute, 2.0))
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/function/math_func.py", line 1881, in polar
    return polar_(abs, angle)
           ^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/auto_generate/gen_ops_prim.py", line 20633, in __call__
    res = pyboost_polar(self, [abs, angle])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Failed calling Polar with "Polar()(abs=Tensor, angle=float)".
The valid calling should be:
"Polar()(abs=<Tensor>, angle=<Tensor>)".

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/pynative/utils/pynative_utils.cc:821 PrintTypeCastErrorForPyObject
```

### 预期结果

文档将 abs/angle 声明为 `(Tensor, float)`，Python float 应被接受并广播/转换为合适 Tensor，结果应与等价 Tensor 参数一致。

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

- 问题定位：CPU complex 结果与 Polar primitive 由 Tensor 控制验证，失败仅是 Python float 处理。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.polar.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
