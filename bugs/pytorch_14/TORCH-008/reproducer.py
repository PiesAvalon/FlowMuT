import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_008_threshold_bf16_scalar_cast() -> dict[str, Any]:
    def fn(x):
        return F.threshold(x, 0.003, 0.0)

    x = torch.tensor(
        [0.00299, 0.003, 0.00301], device="cuda", dtype=torch.bfloat16
    )
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    return {
        "reproduced": not torch.equal(eager, compiled),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"mismatched": int(eager.ne(compiled).sum())},
    }


if __name__ == "__main__":
    result = probe_008_threshold_bf16_scalar_cast()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
