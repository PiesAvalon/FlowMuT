import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, vector):
        return x.to_csr().mv(vector)

x = ms.Tensor([[0, 2, 0], [3, 0, 4]], ms.float32)
vector = ms.Tensor([[1], [2], [3]], ms.float32)
print(Net()(x, vector))
