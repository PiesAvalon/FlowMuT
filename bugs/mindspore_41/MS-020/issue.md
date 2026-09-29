# [Bug]: Graph 模式对不支持的 `next()` 静默返回迭代器内部状态

## 🐞 问题详细描述

静态图文档将 `next` 列为不支持，但 `next(iter(Tensor))` 和 `next(enumerate(Tensor))` 没有被拒绝，反而把 `(current, rest)` 等内部协议结构作为用户结果返回；PyNative 和 Graph 的 for/list 控制返回正常首行。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return next(iter(x))

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
print(Net()(x))
```

### 实际结果

进程退出码：`0`。实际输出：

```text
(Tensor(shape=[3], dtype=Float32, value= [ 1.00000000e+00,  2.00000000e+00,  3.00000000e+00]), Tensor(shape=[1, 3], dtype=Float32, value=
[[ 4.00000000e+00,  5.00000000e+00,  6.00000000e+00]]))
```

### 预期结果

Graph 应明确拒绝文档不支持的 `next`，不能成功编译后返回与 Python 语义不符的内部迭代状态。

## 详细的环境信息描述

- MindSpore：`2.10.0`，PyPI official CPU binary wheel
- Python：`3.11.15`
- 操作系统：`Ubuntu 24.04.4 LTS`，WSL2
- Linux kernel：`6.6.87.2-microsoft-standard-WSL2`
- CPU：`Intel(R) Core(TM) i7-14700K`，x86_64
- Device target：`CPU`
- 执行模式：`Graph`
- 系统 GCC：`13.3.0`；MindSpore 使用预编译二进制包，未在本机编译
- GPU/Ascend 驱动：不适用
- 安装方式：Python 虚拟环境内安装 MindSpore 2.10.0 wheel

## 补充说明

- 问题定位：调用正常退出却返回错误结构；对于不支持的 `next` 调用，应按文档明确拒绝。

参考文档：

- https://www.mindspore.cn/tutorials/en/r2.9.0/compile/static_graph.html
- https://docs.python.org/3/library/functions.html#next
- https://docs.python.org/3/library/functions.html#enumerate

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
