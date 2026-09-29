import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, src):
        return x.diagonal_scatter(src)

x = ms.Tensor([[1, 4, 2], [3, 0, 5]], ms.float32)
src = ms.Tensor([9, 8], ms.float32)
print(Net()(x, src))
