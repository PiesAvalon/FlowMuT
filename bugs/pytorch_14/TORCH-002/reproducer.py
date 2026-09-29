import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_002_forward_ad_tangent() -> dict[str, Any]:
    import torch.autograd.forward_ad as fw_ad

    def fn(x):
        return (x**2).sum()

    x = torch.tensor([0.1, 0.2, 0.3])
    tangent = torch.ones(3)
    with fw_ad.dual_level():
        eager = fw_ad.unpack_dual(fn(fw_ad.make_dual(x, tangent))).tangent
    with fw_ad.dual_level():
        compiled_output = torch.compile(
            fn, backend="inductor", fullgraph=True
        )(fw_ad.make_dual(x, tangent))
        compiled = fw_ad.unpack_dual(compiled_output).tangent
    return {
        "reproduced": eager is not None and compiled is None,
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"tangent_dropped": compiled is None},
    }


if __name__ == "__main__":
    result = probe_002_forward_ad_tangent()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
