import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_020_unused_output_nan_gradient() -> dict[str, Any]:
    def fn(x):
        return torch.relu(x), torch.sqrt(x)

    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    eager_output, _ = fn(x)
    eager_gradient = torch.autograd.grad(eager_output.sum(), x)[0]
    y = x.detach().clone().requires_grad_()
    compiled_output, _ = torch.compile(
        fn, backend="aot_eager", fullgraph=True
    )(y)
    compiled_gradient = torch.autograd.grad(compiled_output.sum(), y)[0]
    new_nan = ~torch.isnan(eager_gradient) & torch.isnan(compiled_gradient)
    return {
        "reproduced": bool(new_nan.any()),
        "eager": repr(eager_gradient),
        "compiled": repr(compiled_gradient),
        "metric": {"spurious_nan_gradients": int(new_nan.sum())},
    }


if __name__ == "__main__":
    result = probe_020_unused_output_nan_gradient()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
