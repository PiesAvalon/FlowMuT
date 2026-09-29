"""Minimal reproducer for CountNonZero primitive constructor dims."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "primitive_dims_list",
    "primitive_dims_tuple",
    "primitive_default_dims",
    "functional_axis_list_control",
    "functional_default_control",
)


class CountNonZeroNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        if variant == "primitive_dims_list":
            self.primitive = ops.CountNonZero(dims=[1])
        elif variant == "primitive_dims_tuple":
            self.primitive = ops.CountNonZero(dims=(1,))
        elif variant == "primitive_default_dims":
            self.primitive = ops.CountNonZero()
        else:
            self.primitive = None

    def construct(self, x):
        if self.variant in ("primitive_dims_list", "primitive_dims_tuple", "primitive_default_dims"):
            return self.primitive(x)
        if self.variant == "functional_axis_list_control":
            return ops.count_nonzero(x, axis=[1])
        if self.variant == "functional_default_control":
            return ops.count_nonzero(x)
        raise ValueError("unknown variant: " + self.variant)


def input_tensor() -> ms.Tensor:
    return ms.Tensor(np.array([[0, 0, 1], [1, 1, 2], [0, 0, 1]], np.int64))


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

    try:
        result = CountNonZeroNet(args.variant)(input_tensor())
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "mode": args.mode,
                    "variant": args.variant,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "mode": args.mode,
                "variant": args.variant,
                "result": summarize(result),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
