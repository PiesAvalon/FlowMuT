import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        s, u, v = x.svd(full_matrices=True, compute_uv=True)
        return ops.shape(s), ops.shape(u), ops.shape(v)

x = ms.Tensor([[1, 2], [3, 4], [5, 7]], ms.float32)
for value in Net()(x):
    print(type(value).__name__, value)
