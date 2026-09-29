"""Minimal reproducer for Tensor.masked_fill_ PyNative/Graph behavior."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import numpy as np


class MaskedFillInplaceNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.mask = ms.Tensor(
            np.array([[True, False, True], [False, True, False]], np.bool_)
        )

    def construct(self, x):
        y = x + 0
        value = ms.Tensor(9.0, ms.float32)
        if self.variant == "statement":
            y.masked_fill_(self.mask, value)
            return y
        if self.variant == "return_value":
            return y.masked_fill_(self.mask, value)
        if self.variant == "functional":
            return y.masked_fill(self.mask, value)
        raise ValueError("unknown variant: " + self.variant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument(
        "--variant",
        choices=("statement", "return_value", "functional"),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = MaskedFillInplaceNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
