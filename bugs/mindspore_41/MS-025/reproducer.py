"""Minimal reproducer for Tensor num_segments in unsorted_segment ops."""
from __future__ import annotations

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


OPS = ("sum", "min", "max", "prod")
VARIANTS = ("int", "tensor_i32", "tensor_i64")


def num_segments_for(variant: str):
    if variant == "int":
        return 2
    if variant == "tensor_i32":
        return ms.Tensor(2, ms.int32)
    if variant == "tensor_i64":
        return ms.Tensor(2, ms.int64)
    raise ValueError("unknown variant: " + variant)


class SegmentSumNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.segment_ids = ms.Tensor(np.array([0, 1, 1], np.int32))

    def construct(self, x):
        return ops.unsorted_segment_sum(x, self.segment_ids, num_segments_for(self.variant))


class SegmentMinNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.segment_ids = ms.Tensor(np.array([0, 1, 1], np.int32))

    def construct(self, x):
        return ops.unsorted_segment_min(x, self.segment_ids, num_segments_for(self.variant))


class SegmentMaxNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.segment_ids = ms.Tensor(np.array([0, 1, 1], np.int32))

    def construct(self, x):
        return ops.unsorted_segment_max(x, self.segment_ids, num_segments_for(self.variant))


class SegmentProdNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.segment_ids = ms.Tensor(np.array([0, 1, 0], np.int32))

    def construct(self, x):
        return ops.unsorted_segment_prod(x, self.segment_ids, num_segments_for(self.variant))


NETS = {
    "sum": SegmentSumNet,
    "min": SegmentMinNet,
    "max": SegmentMaxNet,
    "prod": SegmentProdNet,
}


def summarize(value):
    if isinstance(value, ms.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value": value.asnumpy().tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--op", choices=OPS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]], np.int32))
    result = NETS[args.op](args.variant)(x)
    print(json.dumps({
        "mode": args.mode,
        "op": args.op,
        "variant": args.variant,
        "result": summarize(result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
