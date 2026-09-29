import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x_np = np.array([[20.0, 19.0, 18.0], [-20.0, -21.0, -22.0]], np.float64)
actual = ops.log_softmax(ms.Tensor(x_np), axis=1).asnumpy()
shifted = x_np - x_np.max(axis=1, keepdims=True)
expected = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
print("dtype:", actual.dtype)
print("max_abs_error:", np.max(np.abs(actual - expected)))
