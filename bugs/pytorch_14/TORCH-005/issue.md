# Compiled torch.bernoulli accepts probabilities outside [0, 1]

## 🐛 Describe the bug

On CPU, eager `torch.bernoulli` rejects probabilities outside `[0, 1]`, but the Inductor-compiled call accepts the same invalid input and returns a tensor.

## Minimal reproducer

Run:

```bash
python reproducer.py
```

Standalone source:

```python
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

```

## Actual behavior

Eager result:

```text
RuntimeError: Expected p_in >= 0 && p_in <= 1 to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
```

Compiled result:

```text
tensor([0., 1.])
```

Measured difference:

```json
{
  "compiled_accepted_invalid_input": true
}
```

## Expected behavior

both modes reject invalid probabilities.

## Versions

- `torch_version`: `2.12.1+cu130`
- `torch_git_version`: `7269437d655783a26cba32aa88195b741ff496aa`
- `python`: `3.13.12 | packaged by Anaconda, Inc. | (main, Feb 24 2026, 16:13:31) [GCC 14.3.0]`
- `platform`: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- `cuda_runtime`: `13.0`
- `cuda_available`: `True`
- `cudnn_version`: `92000`
- `gpu`: `{'name': 'NVIDIA GeForce RTX 4090', 'compute_capability': '8.9', 'total_memory': 25756696576}`
- `nvidia_driver`: `591.86`
- `default_dtype`: `torch.float32`
- `float32_matmul_precision`: `highest`
