"""Minimal reproducer for Graph repeat_interleave with scalar Tensor repeats."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "scalar_tensor_axis0",
    "scalar_tensor_axis1",
    "single_element_tensor_control",
    "vector_tensor_control",
    "int_control",
)


class RepeatInterleaveNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "scalar_tensor_axis0":
            return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=0)
        if self.variant == "scalar_tensor_axis1":
            return ops.repeat_interleave(x, ms.Tensor(2, ms.int32), axis=1)
        if self.variant == "single_element_tensor_control":
            return ops.repeat_interleave(x, ms.Tensor([2], ms.int32), axis=0)
        if self.variant == "vector_tensor_control":
            return ops.repeat_interleave(x, ms.Tensor([1, 2, 1], ms.int32), axis=1)
        if self.variant == "int_control":
            return ops.repeat_interleave(x, 2, axis=0)
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

    x = ms.Tensor(np.array([[3, 1, 2], [6, 5, 4]], np.int32))
    result = RepeatInterleaveNet(args.variant)(x)
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
