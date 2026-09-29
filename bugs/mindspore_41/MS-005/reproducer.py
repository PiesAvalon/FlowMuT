"""Minimal reproducer for PyNative CPU searchsorted mixed output dtype failure."""

from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class SearchSortedNet(nn.Cell):
    def __init__(self, order: str) -> None:
        super().__init__()
        self.order = order

    def construct(self, sorted_sequence, values):
        if self.order == "default":
            return ops.searchsorted(sorted_sequence, values)
        if self.order == "int32":
            return ops.searchsorted(sorted_sequence, values, out_int32=True)
        if self.order == "default_then_int32":
            return (
                ops.searchsorted(sorted_sequence, values),
                ops.searchsorted(sorted_sequence, values, out_int32=True),
            )
        if self.order == "int32_then_default":
            return (
                ops.searchsorted(sorted_sequence, values, out_int32=True),
                ops.searchsorted(sorted_sequence, values),
            )
        raise ValueError("unknown order: " + self.order)


def materialize(value):
    if isinstance(value, tuple):
        return tuple(materialize(item) for item in value)
    array = value.asnumpy()
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "values": array.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), default="pynative")
    parser.add_argument(
        "--order",
        choices=("default", "int32", "default_then_int32", "int32_then_default"),
        default="default_then_int32",
    )
    args = parser.parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE,
        device_target="CPU",
    )
    sorted_sequence = ms.Tensor(np.array([[1, 3, 5], [2, 4, 6]], np.int32))
    values = ms.Tensor(np.array([[0, 3, 7], [1, 5, 9]], np.int32))

    result = SearchSortedNet(args.order)(sorted_sequence, values)
    print("mode:", args.mode)
    print("order:", args.order)
    print("materialized:", materialize(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
