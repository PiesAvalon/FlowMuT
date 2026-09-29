# Compiled Hessian broadcasts a trace derivative across all entries

## 🐛 Describe the bug

On CPU, Inductor computes incorrect second derivatives for a trace operation. The eager Hessian has one nonzero entry, while the compiled Hessian changes that entry and adds nonzero values elsewhere.

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

```

## Actual behavior

Eager result:

```text
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0288, 0.0000, 0.0000, 0.0000],
       dtype=torch.float64)
```

Compiled result:

```text
tensor([0.1037, 0.1037, 0.1037, 0.1037, 0.1037, 1.1325, 0.1037, 0.1037, 0.1037],
       dtype=torch.float64)
```

Measured difference:

```json
{
  "max_abs": 0.1037081598277112
}
```

## Expected behavior

Hessian entries agree.

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
