import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.argmax()

x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
result = Net()(x)
print("shape:", result.shape)
print("value:", result)
