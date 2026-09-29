"""Minimal reproducer for Tensor.index_add Graph registration failure."""

from __future__ import annotations

import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class TensorIndexAddNet(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.indices = ms.Tensor(np.array([0, 2], np.int32))
        self.y = ms.Tensor(np.array([[10.0, 20.0], [30.0, 40.0]], np.float32))

    def construct(self, x):
        return x.index_add(self.indices, self.y, 1)


class OpsIndexAddNet(nn.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.indices = ms.Tensor(np.array([0, 2], np.int32))
        self.y = ms.Tensor(np.array([[10.0, 20.0], [30.0, 40.0]], np.float32))

    def construct(self, x):
        return ops.index_add(x, self.indices, self.y, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--api", choices=("tensor", "ops"), default="tensor")
    args = parser.parse_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE,
        device_target="CPU",
    )
    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    net = TensorIndexAddNet() if args.api == "tensor" else OpsIndexAddNet()
    result = net(x)
    print("mode:", args.mode)
    print("api:", args.api)
    print("shape:", result.shape)
    print("dtype:", result.dtype)
    print("result:", result.asnumpy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
