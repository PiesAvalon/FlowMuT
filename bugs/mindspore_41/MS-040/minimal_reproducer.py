import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
absolute = ms.Tensor([1, 2, 3], ms.float32)
print(ops.polar(absolute, 2.0))
