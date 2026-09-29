"""Minimal reproducer for boolean-mask Tensor setitem Graph/PyNative mismatch."""

from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import numpy as np


class MaskSetItemNet(nn.Cell):
    def __init__(self, rhs: str) -> None:
        super().__init__()
        self.rhs = rhs
        self.mask = ms.Tensor(
            np.array(
                [
                    [True, False, True, False],
                    [False, True, False, True],
                    [True, False, False, True],
                ],
                np.bool_,
            )
        )
        self.selected_values = ms.Tensor(np.ones((6,), np.float32) * 5.0)
        self.full_values = ms.Tensor(np.arange(100, 112, dtype=np.float32).reshape(3, 4))

    def construct(self, x):
        y = x + 0
        if self.rhs == "scalar":
            y[self.mask] = ms.Tensor(5.0, ms.float32)
        elif self.rhs == "selected":
            y[self.mask] = self.selected_values
        elif self.rhs == "full":
            y[self.mask] = self.full_values
        else:
            raise ValueError("unknown rhs: " + self.rhs)
        return y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--rhs", choices=("scalar", "selected", "full"), required=True)
    args = parser.parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE,
        device_target="CPU",
    )
    x = ms.Tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    result = MaskSetItemNet(args.rhs)(x)
    print("mode:", args.mode)
    print("rhs:", args.rhs)
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
