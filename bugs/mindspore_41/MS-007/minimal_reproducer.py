import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, mask, selected_values):
        y = x + 0
        y[mask] = selected_values
        return y

x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
mask = ms.Tensor([[True, False, True], [False, True, False]])
selected_values = ms.Tensor([10, 20, 30], ms.float32)
print(Net()(x, mask, selected_values))
