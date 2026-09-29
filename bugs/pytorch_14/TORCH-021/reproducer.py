import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_021_std_large_finite_cpu() -> dict[str, Any]:
    def fn(x):
        return torch.std(x)

    x = torch.tensor([1e19, 1e19, -1e19, -1e19], dtype=torch.float32)
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    return {
        "reproduced": bool(torch.isfinite(eager) and torch.isinf(compiled)),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"finite_to_inf": bool(torch.isinf(compiled))},
    }


if __name__ == "__main__":
    result = probe_021_std_large_finite_cpu()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
