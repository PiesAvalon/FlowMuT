"""Minimal reproducer for positional Tensor.max/min cross-mode inconsistency."""

from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import numpy as np


class ReduceNet(nn.Cell):
    def construct(self, x):
        return x.max(0), x.min(0)


def structure(value):
    if isinstance(value, tuple):
        return tuple(structure(item) for item in value)
    return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    args = parser.parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE,
        device_target="CPU",
    )
    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = ReduceNet()(x)
    print("mode:", args.mode)
    print("return structure:", structure(result))
    print("result:", result)

    if args.mode == "pynative":
        assert isinstance(result[0], tuple) and isinstance(result[1], tuple)
    else:
        assert not isinstance(result[0], tuple) and not isinstance(result[1], tuple)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
