"""Minimal reproducer for Tensor.col2im keyword arguments in Graph mode."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


VARIANTS = ("tensor_positional", "tensor_keywords", "tensor_mixed", "ops_keywords")


class Col2ImNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.output_size = ms.Tensor(np.array([4, 4], np.int32))

    def construct(self, x):
        x4d = x.expand_dims(1)
        if self.variant == "tensor_positional":
            return x4d.col2im(self.output_size, [2, 2], [1, 1], [0, 0], [1, 1])
        if self.variant == "tensor_keywords":
            return x4d.col2im(
                output_size=self.output_size,
                kernel_size=[2, 2],
                dilation=[1, 1],
                padding_value=[0, 0],
                stride=[1, 1],
            )
        if self.variant == "tensor_mixed":
            return x4d.col2im(
                self.output_size,
                kernel_size=[2, 2],
                dilation=[1, 1],
                padding_value=[0, 0],
                stride=[1, 1],
            )
        if self.variant == "ops_keywords":
            return ops.col2im(
                input_x=x4d,
                output_size=self.output_size,
                kernel_size=[2, 2],
                dilation=[1, 1],
                padding_value=[0, 0],
                stride=[1, 1],
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

    x = ms.Tensor(
        np.array(
            [[[1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
              [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
              [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
              [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]]],
            np.float32,
        )
    )
    result = Col2ImNet(args.variant)(x)
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
