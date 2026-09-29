"""Minimal reproducer for Graph filter() with Tensor bool predicates."""
from __future__ import annotations

import argparse
import json

import mindspore as ms
import mindspore.nn as nn
import numpy as np


VARIANTS = (
    "filter_tensor_predicate",
    "filter_tuple_predicate",
    "list_comprehension_predicate",
    "generator_predicate",
    "tuple_comprehension_predicate",
    "list_comprehension_no_filter_control",
    "generator_no_filter_control",
    "filter_tensor_true_control",
    "filter_tuple_true_control",
    "if_tensor_predicate_control",
)


class FilterTensorNet(nn.Cell):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

    def construct(self, x):
        if self.variant == "filter_tensor_predicate":
            return tuple(filter(lambda row: row.sum() > 0, x))
        if self.variant == "filter_tuple_predicate":
            return tuple(filter(lambda row: row.sum() > 0, (x[0], x[1])))
        if self.variant == "list_comprehension_predicate":
            return tuple([row for row in x if row.sum() > 0])
        if self.variant == "generator_predicate":
            return tuple(row for row in x if row.sum() > 0)
        if self.variant == "tuple_comprehension_predicate":
            return tuple([row for row in (x[0], x[1]) if row.sum() > 0])
        if self.variant == "list_comprehension_no_filter_control":
            return tuple([row + 1 for row in x])
        if self.variant == "generator_no_filter_control":
            return tuple(row + 1 for row in x)
        if self.variant == "filter_tensor_true_control":
            return tuple(filter(lambda row: True, x))
        if self.variant == "filter_tuple_true_control":
            return tuple(filter(lambda row: True, (x[0], x[1])))
        if self.variant == "if_tensor_predicate_control":
            for row in x:
                if row.sum() > 0:
                    return row
            return x[0]
        raise ValueError("unknown variant: " + self.variant)


def summarize(value):
    if isinstance(value, ms.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "value": value.asnumpy().tolist(),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [summarize(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [summarize(item) for item in value]}
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

    x = ms.Tensor(np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], np.float32))
    result = FilterTensorNet(args.variant)(x)
    print(json.dumps({
        "mode": args.mode,
        "variant": args.variant,
        "result": summarize(result),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
