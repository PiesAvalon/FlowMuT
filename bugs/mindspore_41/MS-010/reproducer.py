"""Minimal reproducer for Tensor.argmax default-axis Graph/PyNative mismatch."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class ArgmaxDefaultNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "tensor_default":
            return x.argmax()
        if self.variant == "tensor_dim_none":
            return x.argmax(dim=None)
        if self.variant == "tensor_axis_minus1":
            return x.argmax(axis=-1)
        if self.variant == "ops_default":
            return ops.argmax(x)
        raise ValueError("unknown variant: " + self.variant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "tensor_default",
            "tensor_dim_none",
            "tensor_axis_minus1",
            "ops_default",
        ),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[3, 1, 2], [6, 5, 4]], np.int32))
    result = ArgmaxDefaultNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
