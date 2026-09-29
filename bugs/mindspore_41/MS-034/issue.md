# [Bug]: Graph 无法编译混合 Python bool 与标量 Tensor bool 的 `while` 条件

## 🐞 问题详细描述

`python_bool and tensor_bool` 及反向顺序在 PyNative 成功，在 Graph 编译失败；纯 Python while、纯 Tensor while 和相同混合条件的 if 在 Graph 均成功。

### 最小可复现示例

将以下代码保存为 `reproducer.py`，直接执行 `python reproducer.py`。示例完全独立，不需要下载任何数据或模型。

```python
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        i = 0
        out = x[0]
        while i < 1 and out.sum() > 0:
            out = out + 1
            i += 1
        return out

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
print(Net()(x))
```

### 实际结果

进程退出码：`1`。实际输出：

```text
Traceback (most recent call last):
  File "reproducer.py", line 16, in <module>
    print(Net()(x))
          ^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1409, in __call__
    out = self.compile_and_run(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1800, in compile_and_run
    self.compile(*args, **kwargs)
  File "<python_env>/lib/python3.11/site-packages/mindspore/nn/cell.py", line 1782, in compile
    _cell_graph_executor.compile(self, *compile_args, phase=self.phase,
  File "<python_env>/lib/python3.11/site-packages/mindspore/graph/api.py", line 2354, in compile
    result = self._graph_executor.compile(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Cannot join the return values of different branches, perhaps you need to make them equal.
Type Join Failed: Abstract type AbstractTensor cannot join with AbstractScalar.
For more details, please refer to https://www.mindspore.cn/search?inputValue=Type%20Join%20Failed

----------------------------------------------------
- Framework Error Message: (For framework developers)
----------------------------------------------------
The abstract type of the return value of the current branch is:
AbstractScalar(Type: Bool, Value: ValueAny, Shape: NoShape),
 and that of the previous branch is:
AbstractTensor(shape: (), element: AbstractScalar(Type: Bool, Value: ValueAny, Shape: NoShape), value_ptr: 0x581e75af45f0, value: ValueAny).
The node is @3_↵__main___Net_construct_47:CNode_51{[0]: @3_↵__main___Net_construct_47:CNode_52{[0]: ValueNode<Primitive> Switch, [1]: CNode_55, [2]: ValueNode<FuncGraph> 9_↰↵__main___Net_construct_49, [3]: ValueNode<FuncGraph> 19_↱↵__main___Net_construct_50}}, true branch: 9_↰↵__main___Net_construct_49
In file reproducer.py:10, 14~37
        while i < 1 and out.sum() > 0:
              ^~~~~~~~~~~~~~~~~~~~~~~

, false branch: 19_↱↵__main___Net_construct_50
In file reproducer.py:10, 24~37
        while i < 1 and out.sum() > 0:
                        ^~~~~~~~~~~~~


----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/core/abstract/abstract_value.cc:598 ThrowException
```

### 预期结果

Graph 应在 while 中正确处理有效的混合布尔表达式，与相邻 if 和纯类型控制保持一致。

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

- 问题定位：失败只出现在 mixed-bool while 组合，不是 Tensor bool、Python bool 或一般控制流不支持。

参考文档：

- https://www.mindspore.cn/tutorials/en/r2.9.0/compile/static_graph.html
- https://docs.python.org/3/reference/compound_stmts.html#the-while-statement
- https://docs.python.org/3/reference/expressions.html#boolean-operations

## 版本信息

- 可复现版本：`MindSpore 2.9.0`、`MindSpore 2.10.0`
