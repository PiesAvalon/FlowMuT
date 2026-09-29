"""Minimal reproducer for Graph map() rejecting Tensor iterators."""
from __future__ import annotations

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import numpy as np


VARIANTS = (
    "map_tensor",
    "map_tuple_control",
    "zip_tensor_control",
    "sum_tensor_control",
)


class MapTensorNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "map_tensor":
            return tuple(map(lambda row: row + 1, x))
        if self.variant == "map_tuple_control":
            return tuple(map(lambda row: row + 1, (x[0], x[1])))
        if self.variant == "zip_tensor_control":
            return tuple(zip(x, x + 1))
        if self.variant == "sum_tensor_control":
            return sum(x)
        raise ValueError("unknown variant: " + self.variant)


def summarize(value):
    if isinstance(value, ms.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value": value.asnumpy().tolist(),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [summarize(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [summarize(item) for item in value]}
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = MapTensorNet(args.variant)(x)
    print(json.dumps({
        "mode": args.mode,
        "variant": args.variant,
        "result": summarize(result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
