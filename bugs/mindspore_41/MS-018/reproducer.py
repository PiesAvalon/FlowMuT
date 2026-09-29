"""Minimal reproducer for Tensor.reverse_sequence unusable default arguments."""
from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class ReverseSequenceNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.seq_lengths = ms.Tensor(np.array([2, 3], np.int32))

    def construct(self, x):
        if self.variant == "tensor_default":
            return x.reverse_sequence(self.seq_lengths)
        if self.variant == "tensor_explicit":
            return x.reverse_sequence(self.seq_lengths, seq_dim=1)
        if self.variant == "ops_explicit":
            return ops.reverse_sequence(x, self.seq_lengths, seq_dim=1)
        raise ValueError("unknown variant: " + self.variant)


def print_result(result) -> None:
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=("tensor_default", "tensor_explicit", "ops_explicit"), required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]], np.int32))
    result = ReverseSequenceNet(args.variant)(x)
    print("mode:", args.mode)
    print("variant:", args.variant)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
