"""Minimal reproducer for ops.matrix_diag_part rejecting an int k."""
from __future__ import annotations

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "int_k",
    "tensor_k_control",
)


class MatrixDiagPartNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        padding = ms.Tensor(0, ms.int32)
        if self.variant == "int_k":
            return ops.matrix_diag_part(x, 0, padding)
        if self.variant == "tensor_k_control":
            return ops.matrix_diag_part(x, ms.Tensor(0, ms.int32), padding)
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

    x = ms.Tensor(np.array([[3, 1, 2], [6, 5, 4]], np.int32))
    result = MatrixDiagPartNet(args.variant)(x)
    print(json.dumps({
        "mode": args.mode,
        "variant": args.variant,
        "result": summarize(result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
