import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, indices, y):
        return x.index_add(indices, y, 1)

x = ms.Tensor([[1, 4, 2], [3, 0, 5]], ms.float32)
indices = ms.Tensor([0, 2], ms.int32)
y = ms.Tensor([[10, 20], [30, 40]], ms.float32)
print(Net()(x, indices, y))
