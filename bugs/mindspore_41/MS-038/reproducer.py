"""Minimal reproducer for NonMaxSuppressionWithOverlaps scalar arguments."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = (
    "scalar_max_output_size",
    "scalar_thresholds",
    "all_scalar_args",
    "tensor_args_control",
)


class NmsOverlapsNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.nms = ops.NonMaxSuppressionWithOverlaps()

    def construct(self, overlaps, scores):
        if self.variant == "scalar_max_output_size":
            return self.nms(
                overlaps,
                scores,
                2,
                ms.Tensor(0.5, ms.float32),
                ms.Tensor(0.0, ms.float32),
            )
        if self.variant == "scalar_thresholds":
            return self.nms(
                overlaps,
                scores,
                ms.Tensor(2, ms.int32),
                0.5,
                0.0,
            )
        if self.variant == "all_scalar_args":
            return self.nms(overlaps, scores, 2, 0.5, 0.0)
        if self.variant == "tensor_args_control":
            return self.nms(
                overlaps,
                scores,
                ms.Tensor(2, ms.int32),
                ms.Tensor(0.5, ms.float32),
                ms.Tensor(0.0, ms.float32),
            )
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


def inputs() -> tuple[ms.Tensor, ms.Tensor]:
    overlaps = ms.Tensor(
        np.array(
            [
                [1.0, 0.2, 0.7],
                [0.2, 1.0, 0.1],
                [0.7, 0.1, 1.0],
            ],
            np.float32,
        )
    )
    scores = ms.Tensor(np.array([0.9, 0.8, 0.7], np.float32))
    return overlaps, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    result = NmsOverlapsNet(args.variant)(*inputs())
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
