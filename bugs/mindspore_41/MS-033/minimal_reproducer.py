import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return ops.csr_softmax(x.to_csr(), ms.float32).to_tuple()

x = ms.Tensor([[0, 2, 0], [3, 0, 4]], ms.float32)
print(Net()(x))
