"""Minimal reproducer for Tensor keyword arguments rejected in Graph mode."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import numpy as np


VARIANTS = (
    "left_keyword",
    "left_positional",
    "right_keyword",
    "right_positional",
    "renorm_keyword",
    "renorm_positional",
    "float_power_keyword",
    "float_power_positional",
    "multiply_keyword",
    "multiply_positional",
    "equal_keyword",
    "equal_positional",
)


class KeywordNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.shift = ms.Tensor(np.array([[1, 0, 2], [2, 1, 0]], np.int32))
        self.other_int = ms.Tensor(np.array([[3, 1, 2], [6, 4, 4]], np.int32))

    def construct(self, x):
        if self.variant == "left_keyword":
            return x.bitwise_left_shift(other=self.shift)
        if self.variant == "left_positional":
            return x.bitwise_left_shift(self.shift)
        if self.variant == "right_keyword":
            return x.bitwise_right_shift(other=self.shift)
        if self.variant == "right_positional":
            return x.bitwise_right_shift(self.shift)
        if self.variant == "renorm_keyword":
            return x.renorm(2, axis=0, maxnorm=5.0)
        if self.variant == "renorm_positional":
            return x.renorm(2, 0, 5.0)
        if self.variant == "float_power_keyword":
            return x.float_power(other=2)
        if self.variant == "float_power_positional":
            return x.float_power(2)
        if self.variant == "multiply_keyword":
            return x.multiply(value=self.other_int)
        if self.variant == "multiply_positional":
            return x.multiply(self.other_int)
        if self.variant == "equal_keyword":
            return x.equal(other=self.other_int)
        if self.variant == "equal_positional":
            return x.equal(self.other_int)
        raise ValueError("unknown variant: " + self.variant)


def input_for_variant(variant: str) -> ms.Tensor:
    if variant.startswith("renorm"):
        return ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], np.float32))
    if variant.startswith("float_power"):
        return ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]], np.float32))
    return ms.Tensor(np.array([[3, 2, 1], [6, 5, 4]], np.int32))


def print_result(result) -> None:
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    result = KeywordNet(args.variant)(input_for_variant(args.variant))
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
