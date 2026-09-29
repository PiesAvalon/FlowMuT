import json
import math
from typing import Any
import torch
import torch.nn.functional as F

def probe_005_invalid_bernoulli_probability() -> dict[str, Any]:
    def fn(probabilities):
        return torch.bernoulli(probabilities)

    probabilities = torch.tensor([-0.1, 1.1])
    eager_error = compiled_error = None
    compiled_output = None
    try:
        fn(probabilities)
    except Exception as error:
        eager_error = f"{type(error).__name__}: {error}"
    try:
        torch.manual_seed(0)
        compiled_output = torch.compile(
            fn, backend="inductor", fullgraph=True
        )(probabilities)
    except Exception as error:
        compiled_error = f"{type(error).__name__}: {error}"
    return {
        "reproduced": eager_error is not None and compiled_error is None,
        "eager": eager_error,
        "compiled": repr(compiled_output) if compiled_error is None else compiled_error,
        "metric": {"compiled_accepted_invalid_input": compiled_error is None},
    }


if __name__ == "__main__":
    result = probe_005_invalid_bernoulli_probability()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]
