import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_009_arange_int64_overflow() -> dict[str, Any]:
    def fn(anchor):
        return torch.arange(
            9, device=anchor.device, dtype=torch.int64
        ) * torch.tensor([1_500_000_000], device=anchor.device, dtype=torch.int64)

    anchor = torch.zeros(1, device="cuda")
    eager = fn(anchor)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(anchor)
    return {
        "reproduced": not torch.equal(eager, compiled),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {
            "mismatched": int(eager.ne(compiled).sum()),
            "max_abs": int((eager - compiled).abs().max()),
        },
    }


if __name__ == "__main__":
    result = probe_009_arange_int64_overflow()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
