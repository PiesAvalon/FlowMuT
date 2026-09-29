"""Minimal reproducer for Tensor.eigvals failing in PyNative mode."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class EigvalsNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "tensor_eigvals":
            return x.eigvals()
        if self.variant == "ops_eigvals":
            return ops.eigvals(x)
        raise ValueError("unknown variant: " + self.variant)


def print_result(result) -> None:
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=("tensor_eigvals", "ops_eigvals"), required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[2.0, 1.0], [1.0, 2.0]], np.float32))
    result = EigvalsNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
