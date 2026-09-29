"""Minimal reproducer for Graph-mode mixed Python/Tensor bool while failure."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import numpy as np


VARIANTS = (
    "python_then_tensor_while",
    "tensor_then_python_while",
    "pure_python_while_control",
    "pure_tensor_while_control",
    "mixed_if_control",
)


class MixedBoolWhileNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "python_then_tensor_while":
            i = 0
            out = x[0]
            while i < 1 and out.sum() > 0:
                out = out + 1
                i = i + 1
            return out
        if self.variant == "tensor_then_python_while":
            i = 0
            out = x[0]
            while out.sum() > 0 and i < 1:
                out = out + 1
                i = i + 1
            return out
        if self.variant == "pure_python_while_control":
            i = 0
            out = x[0]
            while i < 1:
                out = out + 1
                i = i + 1
            return out
        if self.variant == "pure_tensor_while_control":
            i = ms.Tensor(0, ms.int32)
            one = ms.Tensor(1, ms.int32)
            limit = ms.Tensor(1, ms.int32)
            out = x[0]
            while out.sum() > 0 and i < limit:
                out = out + 1
                i = i + one
            return out
        if self.variant == "mixed_if_control":
            if 0 < 1 and x[0].sum() > 0:
                return x[0]
            return x[1]
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "value": str(value)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = MixedBoolWhileNet(args.variant)(x)
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
