import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return next(iter(x))

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
print(Net()(x))
