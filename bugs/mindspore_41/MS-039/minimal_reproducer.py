import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[0, 1, 2], [3, 0, 4]], ms.int64)
print(ops.CountNonZero(dims=[1])(x))
