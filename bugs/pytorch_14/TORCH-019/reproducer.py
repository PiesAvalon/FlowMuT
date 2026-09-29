import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_019_softshrink_nan() -> dict[str, Any]:
    def fn(x):
        return F.softshrink(x)

    x = torch.tensor([float("-nan"), float("nan"), -1.0, 1.0])
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    lost_nan = torch.isnan(eager) & ~torch.isnan(compiled)
    return {
        "reproduced": bool(lost_nan.any()),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"nan_to_finite": int(lost_nan.sum())},
    }


if __name__ == "__main__":
    result = probe_019_softshrink_nan()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
