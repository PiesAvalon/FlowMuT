# AOTAutograd leaks NaN from an unused output into gradients

## 🐛 Describe the bug

On CPU, AOTAutograd produces a NaN gradient for a ReLU output when another, unused output contains NaN. Eager execution returns a finite gradient for the same input.

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

```

## Actual behavior

Eager result:

```text
tensor([0., 1.])
```

AOTAutograd result:

```text
tensor([nan, 1.])
```

Measured difference:

```json
{
  "spurious_nan_gradients": 1
}
```

## Expected behavior

zero cotangent for unused sqrt output cannot contaminate ReLU gradient.

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
