"""Minimal reproducer for Tensor.add_ CPU in-place semantics."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import numpy as np


class AddInplaceNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.delta = ms.Tensor(2.0, ms.float32)

    def construct(self, x):
        if self.variant == "statement":
            y = x + 0
            y.add_(self.delta)
            return y
        if self.variant == "return_value":
            y = x + 0
            return y.add_(self.delta)
        if self.variant == "paired":
            y = x + 0
            y.add_(self.delta)
            z = x + 0
            r = z.add_(self.delta)
            return y, r
        if self.variant == "functional":
            y = x + 0
            return y.add(self.delta)
        raise ValueError("unknown variant: " + self.variant)


def print_result(result) -> None:
    if isinstance(result, (tuple, list)):
        for index, item in enumerate(result):
            print(f"result[{index}] shape:", item.shape)
            print(f"result[{index}] dtype:", item.dtype)
            print(f"result[{index}]:", item.asnumpy())
        return
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument(
        "--variant",
        choices=("statement", "return_value", "paired", "functional"),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = AddInplaceNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
