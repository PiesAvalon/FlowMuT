"""Minimal reproducer for the documented log_softmax dtype contract.

The documented contract accepts only float16/float32 and says other dtypes
raise TypeError.  CPU execution instead accepts float64, returns a float64
tensor, and performs the central LogSoftmax calculation at roughly float32
precision.
"""

from __future__ import annotations

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class LogSoftmaxNet(nn.Cell):
    def construct(self, x):
        return ops.log_softmax(x, -1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), default="pynative")
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    input_np = np.linspace(-1.25, 1.1, 6, dtype=np.float64).reshape(2, 3)
    actual = LogSoftmaxNet()(ms.Tensor(input_np)).asnumpy()
    maximum = input_np.max(axis=-1, keepdims=True)
    reference = input_np - (maximum + np.log(np.exp(input_np - maximum).sum(axis=-1, keepdims=True)))
    max_abs_error = float(np.max(np.abs(actual - reference)))

    result = {
        "mode": args.mode,
        "mindspore_version": ms.__version__,
        "input_dtype": str(input_np.dtype),
        "output_dtype": str(actual.dtype),
        "actual": actual.tolist(),
        "numpy_float64_reference": reference.tolist(),
        "max_abs_error": max_abs_error,
    }
    print(json.dumps(result, indent=2))

    # The bug is reproduced when an unsupported float64 input is silently
    # accepted, labeled float64 on output, but its error is characteristic of
    # a lower-precision computation.
    assert actual.dtype == np.float64
    assert max_abs_error > 1e-7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
