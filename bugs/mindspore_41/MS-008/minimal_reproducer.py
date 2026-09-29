import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, mask):
        y = x + 0
        y.masked_fill_(mask, ms.Tensor(9.0, ms.float32))
        return y

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
mask = ms.Tensor([[True, False, True], [False, True, False]])
print(Net()(x, mask))
