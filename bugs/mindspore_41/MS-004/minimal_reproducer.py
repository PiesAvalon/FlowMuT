import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x_np = np.linspace(-1.25, 1.1, 6, dtype=np.float64).reshape(2, 3)
actual = ops.leaky_relu(ms.Tensor(x_np), 0.2).asnumpy()
expected = np.where(x_np >= 0, x_np, np.float64(0.2) * x_np)
print("dtype:", actual.dtype)
print("max_abs_error:", np.max(np.abs(actual - expected)))
