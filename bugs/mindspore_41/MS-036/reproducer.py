"""Minimal reproducer for IndexFill primitive scalar value handling."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "primitive_int_value",
    "primitive_float_value",
    "primitive_tensor_value_control",
    "functional_int_value_control",
    "tensor_method_int_value_control",
    "tensor_dim_control",
)


class IndexFillNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.index_fill = ops.IndexFill()

    def construct(self, x):
        index = ms.Tensor([0, 2], ms.int32)
        if self.variant == "primitive_int_value":
            return self.index_fill(x, 1, index, 9)
        if self.variant == "primitive_float_value":
            return self.index_fill(x, 1, index, 9.0)
        if self.variant == "primitive_tensor_value_control":
            value = ms.Tensor(9, ms.int32) if x.dtype == ms.int32 else ms.Tensor(9.0, ms.float32)
            return self.index_fill(x, 1, index, value)
        if self.variant == "functional_int_value_control":
            return ops.index_fill(x, 1, index, 9)
        if self.variant == "tensor_method_int_value_control":
            return x.index_fill(1, index, 9)
        if self.variant == "tensor_dim_control":
            return self.index_fill(x, ms.Tensor(1, ms.int32), index, ms.Tensor(9, ms.int32))
        raise ValueError("unknown variant: " + self.variant)


def input_tensor(variant: str) -> ms.Tensor:
    if variant == "primitive_float_value":
        return ms.Tensor(np.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]], np.float32))
    return ms.Tensor(np.array([[3, 1, 2], [6, 5, 4]], np.int32))


def summarize(value: Any) -> Any:
    if isinstance(value, ms.Tensor):
        array = np.asarray(value.asnumpy())
        return {
            "kind": "Tensor",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "value": array.tolist(),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [summarize(item) for item in value]}
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

    result = IndexFillNet(args.variant)(input_tensor(args.variant))
    print(json.dumps(
        {
            "mode": args.mode,
            "variant": args.variant,
            "result": summarize(result),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
