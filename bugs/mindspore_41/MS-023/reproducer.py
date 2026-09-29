"""Minimal reproducer for ops.round/Tensor.round decimals on CPU."""
from __future__ import annotations

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "ops_default_control",
    "ops_positive_decimals",
    "ops_negative_decimals",
    "tensor_positive_decimals",
)


class RoundNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "ops_default_control":
            return ops.round(x)
        if self.variant == "ops_positive_decimals":
            return ops.round(x, decimals=1)
        if self.variant == "ops_negative_decimals":
            return ops.round(x, decimals=-1)
        if self.variant == "tensor_positive_decimals":
            return x.round(decimals=1)
        raise ValueError("unknown variant: " + self.variant)


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
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([0.1234567, 1.25, -2.35, 1200.1234], np.float32))
    result = RoundNet(args.variant)(x)
    print(json.dumps({
        "mode": args.mode,
        "variant": args.variant,
        "result": summarize(result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
