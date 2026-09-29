"""Reproduce lower-than-float64 precision in CPU Softplus and Mish."""

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class Net(nn.Cell):
    def construct(self, x):
        return ops.softplus(x), ops.mish(x)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
args = parser.parse_args()
ms.set_context(mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE, device_target="CPU")

x_np = np.linspace(-1.25, 1.1, 6, dtype=np.float64).reshape(2, 3)
softplus, mish = (item.asnumpy() for item in Net()(ms.Tensor(x_np)))
softplus_ref = np.logaddexp(0.0, x_np)
mish_ref = x_np * np.tanh(softplus_ref)
result = {
    "mode": args.mode,
    "mindspore": ms.__version__,
    "softplus_dtype": str(softplus.dtype),
    "softplus_max_abs_error": float(np.max(np.abs(softplus - softplus_ref))),
    "mish_dtype": str(mish.dtype),
    "mish_max_abs_error": float(np.max(np.abs(mish - mish_ref))),
}
print(json.dumps(result, indent=2))
assert softplus.dtype == mish.dtype == np.float64
assert result["softplus_max_abs_error"] > 1e-7
assert result["mish_max_abs_error"] > 1e-8
