# Inductor softshrink converts NaN inputs to zero

## 🐛 Describe the bug

On CPU, Inductor turns NaN inputs to `F.softshrink` into zero. Eager execution preserves NaN for those inputs.

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

def probe_019_softshrink_nan() -> dict[str, Any]:
    def fn(x):
        return F.softshrink(x)

    x = torch.tensor([float("-nan"), float("nan"), -1.0, 1.0])
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    lost_nan = torch.isnan(eager) & ~torch.isnan(compiled)
    return {
        "reproduced": bool(lost_nan.any()),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"nan_to_finite": int(lost_nan.sum())},
    }


if __name__ == "__main__":
    result = probe_019_softshrink_nan()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([    nan,     nan, -0.5000,  0.5000])
```

Compiled result:

```text
tensor([ 0.0000,  0.0000, -0.5000,  0.5000])
```

Measured difference:

```json
{
  "nan_to_finite": 2
}
```

## Expected behavior

NaN inputs remain NaN.

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
