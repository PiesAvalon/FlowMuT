import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x, delta):
        statement = x + 0
        statement.add_(delta)
        returned = x + 0
        return statement, returned.add_(delta)

x = ms.Tensor([[1, 2], [3, 4]], ms.float32)
statement, returned = Net()(x, ms.Tensor(2, ms.float32))
print("statement form:", statement)
print("return-value form:", returned)
