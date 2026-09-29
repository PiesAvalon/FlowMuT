# [Bug]: Graph 中 `ops.shape` 对 `Tensor.svd` 输出返回标量 Tensor

## 🐞 问题详细描述

PyNative 中 `Tensor.svd` 输出传给 `ops.shape` 得到 tuple；Graph 中相同调用对 s/u/v 分别返回标量 Tensor `2/3/2`。Tensor `.shape` 属性和功能式 `ops.svd` 的 `ops.shape` 在 Graph 正常。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        s, u, v = x.svd(full_matrices=True, compute_uv=True)
        return ops.shape(s), ops.shape(u), ops.shape(v)

x = ms.Tensor([[1, 2], [3, 4], [5, 7]], ms.float32)
for value in Net()(x):
    print(type(value).__name__, value)
```

### 实际结果

进程退出码：`0`。实际输出：

```text
Tensor 2
Tensor 3
Tensor 2
```

### 预期结果

无论 SVD 入口为何，`ops.shape` 都应返回完整的 `tuple[int]`，例如 `(2,)`、`(3,3)`、`(2,2)`。

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

- 问题定位：这是退出码为 0 的元数据结构错误，定位到 Graph Tensor.svd wrapper 输出与 ops.shape 的组合。

参考文档：

- https://www.mindspore.cn/docs/en/r2.9.0/api_python/mindspore/Tensor/mindspore.Tensor.svd.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.svd.html
- https://www.mindspore.cn/docs/en/r2.9.0/api_python/ops/mindspore.ops.shape.html

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
