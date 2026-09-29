"""Minimal reproducer for WhileLoop Tensor constants in Graph mode."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "tensor_cond_bound",
    "tensor_loop_step",
    "tensor_attrs",
    "python_constants_control",
    "global_tensor_constant_control",
    "tuple_tensor_carry_control",
    "nested_tensor_constant_control",
    "fori_tensor_constant_control",
    "scan_tensor_constant_control",
)


def scan_inc(res, el):
    res = res + ms.Tensor(1, ms.int32)
    return res, res


def global_tensor_cond(value):
    return value < ms.Tensor(5, ms.int32)


def global_loop(value):
    return value + 1


class WhileLoopNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.limit = ms.Tensor(5, ms.int32)
        self.step = ms.Tensor(1, ms.int32)
        self.while_loop = ops.WhileLoop()
        self.fori_loop = ops.ForiLoop()
        self.scan = ops.Scan()

    def construct(self):
        if self.variant == "tensor_cond_bound":
            def cond(value):
                return value < ms.Tensor(5, ms.int32)

            def loop(value):
                return value + 1

            return self.while_loop(cond, loop, ms.Tensor(0, ms.int32))

        if self.variant == "tensor_loop_step":
            def cond(value):
                return value < 5

            def loop(value):
                return value + ms.Tensor(1, ms.int32)

            return self.while_loop(cond, loop, ms.Tensor(0, ms.int32))

        if self.variant == "tensor_attrs":
            def cond(value):
                return value < self.limit

            def loop(value):
                return value + self.step

            return self.while_loop(cond, loop, ms.Tensor(0, ms.int32))

        if self.variant == "python_constants_control":
            def cond(value):
                return value < 5

            def loop(value):
                return value + 1

            return self.while_loop(cond, loop, ms.Tensor(0, ms.int32))

        if self.variant == "global_tensor_constant_control":
            return self.while_loop(
                global_tensor_cond, global_loop, ms.Tensor(0, ms.int32)
            )

        if self.variant == "tuple_tensor_carry_control":
            def cond(state):
                value, count = state
                return value < 3

            def loop(state):
                value, count = state
                return value + 1, count + 1

            return self.while_loop(cond, loop, (ms.Tensor(0, ms.int32), ms.Tensor(0, ms.int32)))

        if self.variant == "nested_tensor_constant_control":
            def loop(value):
                return value + ms.Tensor(1, ms.int32)

            return loop(ms.Tensor(0, ms.int32))

        if self.variant == "fori_tensor_constant_control":
            def step(index, value):
                return value + ms.Tensor(1, ms.int32)

            return self.fori_loop(0, 3, step, ms.Tensor(0, ms.int32))

        if self.variant == "scan_tensor_constant_control":
            return self.scan(scan_inc, ms.Tensor(0, ms.int32), [None, None, None])

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
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    try:
        result = WhileLoopNet(args.variant)()
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
