"""Minimal reproducer for Graph-mode CSRMul metadata failure."""
from __future__ import annotations

import argparse
import json
from typing import Any

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.common.sparse_tensor import CSRTensor


VARIANTS = (
    "ops_csr_mul_to_tuple",
    "tensor_mul_to_tuple",
    "csr_relu_control",
    "csr_dense_control",
)


class CSRMulNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        csr = x.to_csr()
        dense = ops.ones_like(x)
        if self.variant == "ops_csr_mul_to_tuple":
            return ops.csr_mul(csr, dense).to_tuple()
        if self.variant == "tensor_mul_to_tuple":
            return (csr * dense).to_tuple()
        if self.variant == "csr_relu_control":
            return ops.csr_relu(csr).to_tuple()
        if self.variant == "csr_dense_control":
            return csr.to_dense()
        raise ValueError("unknown variant: " + self.variant)


def summarize(value: Any) -> Any:
    if isinstance(value, CSRTensor):
        return {
            "kind": "CSRTensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "indptr": summarize(value.indptr),
            "indices": summarize(value.indices),
            "values": summarize(value.values),
        }
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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "value": str(value)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pynative", "graph"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()

    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target="CPU")

    x = ms.Tensor(np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], np.float32))
    result = CSRMulNet(args.variant)(x)
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
