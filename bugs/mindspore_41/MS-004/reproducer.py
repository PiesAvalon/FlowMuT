"""Reproduce float32 rounding of LeakyReLU alpha with float64 input."""

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class Net(nn.Cell):
    def construct(self, x):
        return ops.leaky_relu(x, 0.2)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
args = parser.parse_args()
ms.set_context(mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE, device_target="CPU")
x_np = np.linspace(-1.25, 1.1, 6, dtype=np.float64).reshape(2, 3)
actual = Net()(ms.Tensor(x_np)).asnumpy()
reference = np.where(x_np >= 0.0, x_np, np.float64(0.2) * x_np)
error = float(np.max(np.abs(actual - reference)))
print(json.dumps({"mode": args.mode, "dtype": str(actual.dtype), "max_abs_error": error}, indent=2))
assert actual.dtype == np.float64
assert error > 1e-10
