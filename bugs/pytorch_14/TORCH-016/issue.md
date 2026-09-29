# Eval BatchNorm + ELU + GroupNorm chain produces wrong values

## 🐛 Describe the bug

On CPU, compiling the BatchNorm, ELU, and GroupNorm chain changes the final output values. The reproduced maximum absolute difference from eager execution is about `1.29`.

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

```

## Actual behavior

Eager result:

```text
tensor([-12.5268, -12.5268, -12.5268, -12.5268, -12.5268, -12.5268],
       grad_fn=<SliceBackward0>)
```

Compiled result:

```text
tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155, -13.8155],
       grad_fn=<SliceBackward0>)
```

Measured difference:

```json
{
  "max_abs": 1.2886991500854492
}
```

## Expected behavior

outputs agree within 1e-4.

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
