"""Minimal reproducer for Tensor.expand failing in Graph mode."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import numpy as np


class ExpandNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "expand_varargs":
            return x.expand(2, 3)
        if self.variant == "expand_tuple":
            return x.expand((2, 3))
        if self.variant == "broadcast_to":
            return x.broadcast_to((2, 3))
        raise ValueError("unknown variant: " + self.variant)


def print_result(result) -> None:
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument(
        "--variant",
        choices=("expand_varargs", "expand_tuple", "broadcast_to"),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 2.0, 3.0]], np.float32))
    result = ExpandNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
