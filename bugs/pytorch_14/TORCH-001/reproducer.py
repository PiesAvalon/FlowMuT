import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_001_int8_bmm_dtype() -> dict[str, Any]:
    def fn(a, b):
        return torch.matmul(a, b)

    a = torch.ones((1, 1, 2), dtype=torch.int8)
    b = torch.ones((1, 2, 2), dtype=torch.int8)
    eager = fn(a, b)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
    return {
        "reproduced": eager.dtype != compiled.dtype,
        "eager": f"{eager!r}, dtype={eager.dtype}",
        "compiled": f"{compiled!r}, dtype={compiled.dtype}",
        "metric": {"dtype_equal": eager.dtype == compiled.dtype},
    }


if __name__ == "__main__":
    result = probe_001_int8_bmm_dtype()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
