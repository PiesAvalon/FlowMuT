# [Bug]: PyNative 中 `Tensor.eigvals()` 未向算子传入当前 Tensor

## 🐞 问题详细描述

`Tensor.eigvals()` 在 PyNative 抛出 `eigvals() missing 1 required positional argument: 'A'`；`ops.eigvals(x)` 在 PyNative 成功，Tensor 方法在 Graph 也成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[2, 1], [1, 2]], ms.float32)
print(x.eigvals())
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 5, in <module>
    print(x.eigvals())
          ^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/common/tensor.py", line 3202, in eigvals
    return tensor_operator_registry.get("eigvals")()(self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: eigvals() missing 1 required positional argument: 'A'
```

### 预期结果

PyNative Tensor wrapper 应把 `self` 传给 `ops.eigvals`，并返回与功能式及 Graph 路径相同的特征值。

## 详细的环境信息描述

- MindSpore：`2.10.0`，PyPI official CPU binary wheel
- Python：`3.11.15`
- 操作系统：`Ubuntu 24.04.4 LTS`，WSL2
- Linux kernel：`6.6.87.2-microsoft-standard-WSL2`
- CPU：`Intel(R) Core(TM) i7-14700K`，x86_64
- Device target：`CPU`
- 执行模式：`PyNative`
- 系统 GCC：`13.3.0`；MindSpore 使用预编译二进制包，未在本机编译
- GPU/Ascend 驱动：不适用
- 安装方式：Python 虚拟环境内安装 MindSpore 2.10.0 wheel

## 补充说明

- 问题定位：CPU eigvals 内核通过功能式和 Graph 控制验证，失败局限于 PyNative Tensor wrapper 调用。

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
