import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, sorted_sequence, values):
        return (ops.searchsorted(sorted_sequence, values),
                ops.searchsorted(sorted_sequence, values, out_int32=True))

sorted_sequence = ms.Tensor([[1, 3, 5], [2, 4, 6]], ms.int32)
values = ms.Tensor([[0, 3, 7], [1, 5, 9]], ms.int32)
print(tuple(item.asnumpy() for item in Net()(sorted_sequence, values)))
