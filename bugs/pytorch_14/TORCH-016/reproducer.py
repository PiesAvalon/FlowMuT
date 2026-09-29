import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_016_eval_norm_chain() -> dict[str, Any]:
    batch_norm = torch.nn.BatchNorm1d(10).eval()
    elu = torch.nn.ELU()
    group_norm = torch.nn.GroupNorm(10, 10).eval()

    def fn(x):
        x = group_norm(elu(batch_norm(x)))
        return torch.log(torch.clamp(x, min=1e-6))

    x = torch.ones(6, 10, 12)
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    difference = (eager - compiled).abs()
    return {
        "reproduced": not torch.allclose(
            eager, compiled, rtol=1e-4, atol=1e-4, equal_nan=True
        ),
        "eager": repr(eager.flatten()[:6]),
        "compiled": repr(compiled.flatten()[:6]),
        "metric": {"max_abs": float(difference.max())},
    }


if __name__ == "__main__":
    result = probe_016_eval_norm_chain()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
