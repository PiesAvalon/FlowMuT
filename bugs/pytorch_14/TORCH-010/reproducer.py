import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_010_acosh_large_finite() -> dict[str, Any]:
    def fn(x):
        return torch.acosh(x)

    x = torch.tensor([5e22, 9e25, 7e21, 2.0], dtype=torch.float32)
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    incorrect_infinity = torch.isfinite(eager) & torch.isinf(compiled)
    return {
        "reproduced": bool(incorrect_infinity.any()),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"finite_to_inf": int(incorrect_infinity.sum())},
    }


if __name__ == "__main__":
    result = probe_010_acosh_large_finite()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
