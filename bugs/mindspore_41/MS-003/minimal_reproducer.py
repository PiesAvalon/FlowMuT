import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x_np = np.array([-8.25, -1.75, 0.25, 3.5, 8.75], np.float64)
x = ms.Tensor(x_np)
softplus = ops.softplus(x).asnumpy()
mish = ops.mish(x).asnumpy()
softplus_ref = np.logaddexp(0.0, x_np)
mish_ref = x_np * np.tanh(softplus_ref)
print("softplus dtype/error:", softplus.dtype, np.max(np.abs(softplus-softplus_ref)))
print("mish dtype/error:", mish.dtype, np.max(np.abs(mish-mish_ref)))
