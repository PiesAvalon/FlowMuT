import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, shift):
        return x.bitwise_left_shift(other=shift)

x = ms.Tensor([[3, 2, 1], [6, 5, 4]], ms.int32)
print(Net()(x, ms.Tensor(1, ms.int32)))
