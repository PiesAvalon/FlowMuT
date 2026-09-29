import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_011_fp16_cast_multiply_overflow() -> dict[str, Any]:
    def fn(x):
        half = x.to(torch.float16)
        return (half * half).to(torch.float32)

    x = torch.tensor([5000.0])
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    return {
        "reproduced": bool(torch.isinf(eager).any() and torch.isfinite(compiled).any()),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"overflow_removed": bool(torch.isfinite(compiled).all())},
    }


if __name__ == "__main__":
    result = probe_011_fp16_cast_multiply_overflow()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
