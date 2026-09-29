import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_018_fractional_max_pool_rng() -> dict[str, Any]:
    def fn(x):
        return F.fractional_max_pool2d(
            x, 3, output_size=(4, 4), return_indices=True
        )

    torch.manual_seed(18)
    x = torch.randn(1, 3, 10, 10)
    torch.manual_seed(0)
    eager = fn(x)
    torch.manual_seed(0)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    index_difference = eager[1].ne(compiled[1])
    return {
        "reproduced": bool(index_difference.any()),
        "eager": f"values={eager[0].flatten()[:8]!r}, indices={eager[1].flatten()[:8]!r}",
        "compiled": (
            f"values={compiled[0].flatten()[:8]!r}, "
            f"indices={compiled[1].flatten()[:8]!r}"
        ),
        "metric": {"index_mismatched": int(index_difference.sum())},
    }


if __name__ == "__main__":
    result = probe_018_fractional_max_pool_rng()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
