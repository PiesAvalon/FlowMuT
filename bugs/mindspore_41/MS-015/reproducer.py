"""Minimal reproducer for Tensor.scatter_div failing in Graph mode."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class ScatterDivNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.indices = ms.Tensor(np.array([[0, 1], [1, 2]], np.int32))
        self.updates = ms.Tensor(np.array([2.0, 4.0], np.float32))

    def construct(self, x):
        if self.variant == "tensor_scatter_div":
            return x.scatter_div(self.indices, self.updates)
        if self.variant == "ops_tensor_scatter_div":
            return ops.tensor_scatter_div(x, self.indices, self.updates)
        if self.variant == "tensor_scatter_sub":
            return x.scatter_sub(self.indices, self.updates)
        if self.variant == "tensor_scatter_mul":
            return x.scatter_mul(self.indices, self.updates)
        if self.variant == "tensor_scatter_max":
            return x.scatter_max(self.indices, self.updates)
        if self.variant == "tensor_scatter_min":
            return x.scatter_min(self.indices, self.updates)
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
        choices=(
            "tensor_scatter_div",
            "ops_tensor_scatter_div",
            "tensor_scatter_sub",
            "tensor_scatter_mul",
            "tensor_scatter_max",
            "tensor_scatter_min",
        ),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 2.0, 5.0]], np.float32))
    result = ScatterDivNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
