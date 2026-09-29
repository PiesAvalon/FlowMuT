"""Minimal reproducer for Tensor.amax/amin initial keyword failing in Graph mode."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class ReduceInitialNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.mask = ms.Tensor(np.array([[True, False, True], [False, True, True]], np.bool_))

    def construct(self, x):
        if self.variant == "tensor_amax_initial":
            return x.amax(axis=1, initial=10)
        if self.variant == "tensor_amin_initial":
            return x.amin(axis=1, initial=0)
        if self.variant == "tensor_amax_where_initial":
            return x.amax(axis=1, initial=0, where=self.mask)
        if self.variant == "tensor_amin_where_initial":
            return x.amin(axis=1, initial=10, where=self.mask)
        if self.variant == "tensor_amax_keepdims":
            return x.amax(axis=1, keepdims=True)
        if self.variant == "tensor_amin_keepdims":
            return x.amin(axis=1, keepdims=True)
        if self.variant == "ops_amax_initial":
            return ops.amax(x, axis=1, initial=10)
        if self.variant == "ops_amin_initial":
            return ops.amin(x, axis=1, initial=0)
        if self.variant == "ops_amax_where_initial":
            return ops.amax(x, axis=1, initial=0, where=self.mask)
        if self.variant == "ops_amin_where_initial":
            return ops.amin(x, axis=1, initial=10, where=self.mask)
        if self.variant == "ops_amax_keepdims":
            return ops.amax(x, axis=1, keepdims=True)
        if self.variant == "ops_amin_keepdims":
            return ops.amin(x, axis=1, keepdims=True)
        if self.variant == "tensor_amax_plain":
            return x.amax(axis=1)
        if self.variant == "tensor_amin_plain":
            return x.amin(axis=1)
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
            "tensor_amax_initial",
            "tensor_amin_initial",
            "tensor_amax_where_initial",
            "tensor_amin_where_initial",
            "tensor_amax_keepdims",
            "tensor_amin_keepdims",
            "ops_amax_initial",
            "ops_amin_initial",
            "ops_amax_where_initial",
            "ops_amin_where_initial",
            "ops_amax_keepdims",
            "ops_amin_keepdims",
            "tensor_amax_plain",
            "tensor_amin_plain",
        ),
        required=True,
    )
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[3, 1, 2], [6, 5, 4]], np.int32))
    result = ReduceInitialNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
