"""Minimal reproducer for Tensor.svd output shape metadata in Graph mode."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "tensor_ops_shape_full",
    "tensor_ops_shape_reduced",
    "tensor_shape_property_full",
    "ops_ops_shape_full",
)


class SvdShapeNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "tensor_ops_shape_full":
            s, u, v = x.svd(full_matrices=True, compute_uv=True)
            return ops.shape(s), ops.shape(u), ops.shape(v)
        if self.variant == "tensor_ops_shape_reduced":
            s, u, v = x.svd(full_matrices=False, compute_uv=True)
            return ops.shape(s), ops.shape(u), ops.shape(v)
        if self.variant == "tensor_shape_property_full":
            s, u, v = x.svd(full_matrices=True, compute_uv=True)
            return s.shape, u.shape, v.shape
        if self.variant == "ops_ops_shape_full":
            s, u, v = ops.svd(x, full_matrices=True, compute_uv=True)
            return ops.shape(s), ops.shape(u), ops.shape(v)
        raise ValueError("unknown variant: " + self.variant)


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
    if isinstance(value, list):
        return {"kind": "list", "items": [summarize(item) for item in value]}
    if isinstance(value, np.generic):
        return value.item()
    return {"kind": type(value).__name__, "value": value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], np.float32))
    result = SvdShapeNet(args.variant)(x)
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
