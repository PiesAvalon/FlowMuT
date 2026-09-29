import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

def step(carry, _):
    return carry + 1, carry

print(ops.Scan()(step, ms.Tensor(0, ms.int32), None, length=3))
