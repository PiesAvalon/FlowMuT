import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([1.25, 2.75, -3.15], ms.float32)
print(ops.round(x, decimals=1))
