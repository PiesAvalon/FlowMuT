import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, indices, updates):
        return x.scatter_div(indices, updates)

x = ms.Tensor([[1, 4, 2], [3, 2, 5]], ms.float32)
indices = ms.Tensor([[0, 1], [1, 2]], ms.int32)
updates = ms.Tensor([2, 4], ms.float32)
print(Net()(x, indices, updates))
