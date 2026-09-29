"""Minimal reproducer for ops.shape return type on window tensors."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


WINDOWS = ("hamming", "hann", "kaiser")


class WindowShapeNet(nn.Cell):
    def __init__(self, window: str) -> None:
        super().__init__()
        self.window = window

    def construct(self):
        if self.window == "hamming":
            y = ops.hamming_window(5, dtype=ms.float32)
        elif self.window == "hann":
            y = ops.hann_window(5, dtype=ms.float32)
        elif self.window == "kaiser":
            y = ops.kaiser_window(5, dtype=ms.float32)
        else:
            raise ValueError("unknown window: " + self.window)
        return ops.shape(y), y.shape, y


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
    parser.add_argument("--window", choices=WINDOWS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    ops_shape, tensor_shape, tensor = WindowShapeNet(args.window)()
    print(json.dumps(
        {
            "mode": args.mode,
            "window": args.window,
            "ops_shape": summarize(ops_shape),
            "tensor_shape": summarize(tensor_shape),
            "tensor_summary": {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            },
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
