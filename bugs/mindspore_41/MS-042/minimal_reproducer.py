import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self):
        def cond(value):
            return value < ms.Tensor(5, ms.int32)

        def body(value):
            return value + 1

        return ops.WhileLoop()(cond, body, ms.Tensor(0, ms.int32))

print(Net()())
