import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self):
        window = ops.hann_window(5, dtype=ms.float32)
        return ops.shape(window), window.shape

ops_shape, property_shape = Net()()
print("ops.shape:", type(ops_shape).__name__, ops_shape)
print("Tensor.shape:", type(property_shape).__name__, property_shape)
