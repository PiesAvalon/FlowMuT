"""Minimal reproducer for Tensor.diag Graph registration failure."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class DiagNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "tensor_method":
            return x.diag()
        if self.variant == "functional":
            return ops.diag(x)
        raise ValueError("unknown variant: " + self.variant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=("tensor_method", "functional"), required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([1, 2, 3], np.int32))
    result = DiagNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
