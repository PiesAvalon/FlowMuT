import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_013_hessian_trace_broadcast() -> dict[str, Any]:
    from torch.func import hessian

    def fn(tensor):
        eye = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
        return torch.mean(1.0 / (tensor + torch.trace(tensor) * eye))

    x = torch.tensor(
        [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60], [0.70, 0.80, 0.95]],
        dtype=torch.float64,
    )
    eager = hessian(fn)(x)
    compiled = torch.compile(
        hessian(fn), backend="inductor", fullgraph=True
    )(x)
    difference = (eager - compiled).abs()
    return {
        "reproduced": not torch.allclose(eager, compiled),
        "eager": repr(eager.reshape(9, 9)[5]),
        "compiled": repr(compiled.reshape(9, 9)[5]),
        "metric": {"max_abs": float(difference.max())},
    }


if __name__ == "__main__":
    result = probe_013_hessian_trace_broadcast()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
