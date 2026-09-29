"""Minimal reproducer for Tensor.diagonal_scatter Graph default argument failure."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class DiagonalScatterNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.src = ms.Tensor(np.array([9.0, 8.0], np.float32))

    def construct(self, x):
        if self.variant == "method_default":
            return x.diagonal_scatter(self.src)
        if self.variant == "method_positional_offset":
            return x.diagonal_scatter(self.src, 0)
        if self.variant == "method_keyword_offset":
            return x.diagonal_scatter(self.src, offset=0)
        if self.variant == "functional_default":
            return ops.diagonal_scatter(x, self.src)
        if self.variant == "functional_positional_offset":
            return ops.diagonal_scatter(x, self.src, 0)
        raise ValueError("unknown variant: " + self.variant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "method_default",
            "method_positional_offset",
            "method_keyword_offset",
            "functional_default",
            "functional_positional_offset",
        ),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = DiagonalScatterNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
