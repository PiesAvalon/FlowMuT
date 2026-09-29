"""Minimal reproducer for Scan xs=None in PyNative mode."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "none_length_positional",
    "none_length_keyword",
    "none_tensor_init",
    "list_none_control",
    "tuple_tensor_control",
)


def inc(res, el):
    res = res + 1
    return res, res


def cumsum(res, el):
    res = res + el
    return res, res


class ScanNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.scan = ops.Scan()

    def construct(self):
        if self.variant == "none_length_positional":
            return self.scan(inc, 0, None, 3)
        if self.variant == "none_length_keyword":
            return self.scan(inc, 0, None, length=3)
        if self.variant == "none_tensor_init":
            return self.scan(inc, ms.Tensor(0, ms.int32), None, 3)
        if self.variant == "list_none_control":
            return self.scan(inc, 0, [None, None, None])
        if self.variant == "tuple_tensor_control":
            return self.scan(cumsum, ms.Tensor(0, ms.int32), (ms.Tensor(1, ms.int32), ms.Tensor(2, ms.int32)))
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
        result = ScanNet(args.variant)()
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
