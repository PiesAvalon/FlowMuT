import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_017_permute_slice_inplace_transpose() -> dict[str, Any]:
    def fn(x):
        x = x.permute(2, 1, 0)
        x[0] += 1
        x[0] = x[0].t()
        return x

    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    eager = fn(x.clone())
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x.clone())
    return {
        "reproduced": not torch.equal(eager, compiled),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"mismatched": int(eager.ne(compiled).sum())},
    }


if __name__ == "__main__":
    result = probe_017_permute_slice_inplace_transpose()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
