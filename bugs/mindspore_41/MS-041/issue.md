# [Bug]: PyNative 中 `ops.Scan` 拒绝文档允许的 `xs=None`

## 🐞 问题详细描述

文档允许 `xs=None` 配合 length。Graph 的位置/关键字 length 和 Tensor init 均成功；PyNative 均因 validator 把 None 当作 `isinstance` 的类型参数而抛出内部 TypeError。list/tuple xs 控制正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def step(carry, _):
    return carry + 1, carry

print(ops.Scan()(step, ms.Tensor(0, ms.int32), None, length=3))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 9, in <module>
    print(ops.Scan()(step, ms.Tensor(0, ms.int32), None, length=3))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/ops/operations/manually_defined/ops_def.py", line 2535, in __call__
    validator.check_value_type("xs", xs, [list, tuple, None], "Scan")
  File "<python_env>/lib/python3.11/site-packages/mindspore/_checkparam.py", line 742, in check_value_type
    cond = not isinstance(arg_value, tuple(valid_types))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```

### 预期结果

PyNative 应把 `None` 校验为 `type(None)` 并按 length 构造扫描序列，返回与 Graph 相同的结果。

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

- 问题定位：Graph 与非 None 控制证明 loop/result 有效，错误明确来自 PyNative `isinstance` 参数验证。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.Scan.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
