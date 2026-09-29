"""Minimal reproducer for Polar Python float arguments."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "functional_scalar_angle",
    "functional_scalar_abs",
    "functional_both_scalar",
    "primitive_scalar_angle",
    "primitive_scalar_abs",
    "functional_tensor_angle_control",
    "functional_tensor_abs_control",
    "functional_scalar_tensor_pair_control",
    "primitive_tensor_angle_control",
)


class PolarNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.polar = ops.Polar()

    def construct(self, x):
        if self.variant == "functional_scalar_angle":
            return ops.polar(x, 2.0)
        if self.variant == "functional_scalar_abs":
            return ops.polar(2.0, x)
        if self.variant == "functional_both_scalar":
            return ops.polar(2.0, 2.0)
        if self.variant == "primitive_scalar_angle":
            return self.polar(x, 2.0)
        if self.variant == "primitive_scalar_abs":
            return self.polar(2.0, x)
        if self.variant == "functional_tensor_angle_control":
            return ops.polar(x, ops.ones_like(x) * 2.0)
        if self.variant == "functional_tensor_abs_control":
            return ops.polar(ops.ones_like(x) * 2.0, x)
        if self.variant == "functional_scalar_tensor_pair_control":
            return ops.polar(ms.Tensor(2.0, ms.float32), ms.Tensor(2.0, ms.float32))
        if self.variant == "primitive_tensor_angle_control":
            return self.polar(x, ops.ones_like(x) * 2.0)
        raise ValueError("unknown variant: " + self.variant)


def input_tensor() -> ms.Tensor:
    return ms.Tensor(np.array([1.0, 2.0, 3.0], np.float32))


def summarize(value: Any) -> Any:
    if isinstance(value, ms.Tensor):
        array = np.asarray(value.asnumpy())
        return {
            "kind": "Tensor",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "value_real": np.real(array).tolist(),
            "value_imag": np.imag(array).tolist(),
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
        result = PolarNet(args.variant)(input_tensor())
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
