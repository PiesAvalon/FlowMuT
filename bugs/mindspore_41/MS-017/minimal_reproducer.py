import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.amax(axis=1, initial=10)

print(Net()(ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)))
