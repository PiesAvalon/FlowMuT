# CPU compiled std overflows when eager returns a finite result

## 🐛 Describe the bug

On CPU, Inductor returns `inf` from `torch.std` for an input whose standard deviation is representable as a finite value. Eager execution returns approximately `1.1547e+19`.

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

def probe_021_std_large_finite_cpu() -> dict[str, Any]:
    def fn(x):
        return torch.std(x)

    x = torch.tensor([1e19, 1e19, -1e19, -1e19], dtype=torch.float32)
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    return {
        "reproduced": bool(torch.isfinite(eager) and torch.isinf(compiled)),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"finite_to_inf": bool(torch.isinf(compiled))},
    }


if __name__ == "__main__":
    result = probe_021_std_large_finite_cpu()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor(1.1547e+19)
```

Compiled result:

```text
tensor(inf)
```

Measured difference:

```json
{
  "finite_to_inf": true
}
```

## Expected behavior

representable standard deviation stays finite.

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
