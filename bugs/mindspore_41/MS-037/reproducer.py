"""Minimal reproducer for MatrixDiagV3 primitive Python int arguments."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "primitive_int_main_diag",
    "primitive_int_offset_diag",
    "primitive_tensor_main_control",
    "primitive_tensor_offset_control",
    "functional_int_main_control",
    "functional_int_offset_control",
)


class MatrixDiagV3Net(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.matrix_diag_v3 = ops.MatrixDiagV3()

    def construct(self, x):
        if self.variant == "primitive_int_main_diag":
            return self.matrix_diag_v3(x, 0, 3, 3, 0)
        if self.variant == "primitive_int_offset_diag":
            return self.matrix_diag_v3(x, 1, 3, 4, 0)
        if self.variant == "primitive_tensor_main_control":
            return self.matrix_diag_v3(
                x,
                ms.Tensor(0, ms.int32),
                ms.Tensor(3, ms.int32),
                ms.Tensor(3, ms.int32),
                ms.Tensor(0, ms.int32),
            )
        if self.variant == "primitive_tensor_offset_control":
            return self.matrix_diag_v3(
                x,
                ms.Tensor(1, ms.int32),
                ms.Tensor(3, ms.int32),
                ms.Tensor(4, ms.int32),
                ms.Tensor(0, ms.int32),
            )
        if self.variant == "functional_int_main_control":
            return ops.matrix_diag(x, k=0, num_rows=3, num_cols=3, padding_value=0)
        if self.variant == "functional_int_offset_control":
            return ops.matrix_diag(x, k=1, num_rows=3, num_cols=4, padding_value=0)
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

    x = ms.Tensor(np.array([1, 2, 3], np.int32))
    result = MatrixDiagV3Net(args.variant)(x)
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
