import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[3, 1, 2], [6, 5, 4]], ms.int32)
print(ops.matrix_diag_part(x, 0, ms.Tensor(0, ms.int32)))
