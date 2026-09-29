import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)

x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
print(Net()(x))
