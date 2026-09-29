import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.max(0), x.min(0)

x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.float32)
result = Net()(x)
print("outer_type:", type(result).__name__)
print("items:", len(result))
print(result)
