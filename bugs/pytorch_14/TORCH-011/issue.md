# Inductor removes fp16 intermediate multiplication overflow

## 🐛 Describe the bug

On CPU, an fp16 multiplication overflows to `inf` in eager execution before being converted to fp32. Inductor instead returns a finite value, changing the result of the explicit fp16 intermediate operation.

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

def probe_011_fp16_cast_multiply_overflow() -> dict[str, Any]:
    def fn(x):
        half = x.to(torch.float16)
        return (half * half).to(torch.float32)

    x = torch.tensor([5000.0])
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    return {
        "reproduced": bool(torch.isinf(eager).any() and torch.isfinite(compiled).any()),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"overflow_removed": bool(torch.isfinite(compiled).all())},
    }


if __name__ == "__main__":
    result = probe_011_fp16_cast_multiply_overflow()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([inf])
```

Compiled result:

```text
tensor([25000000.])
```

Measured difference:

```json
{
  "overflow_removed": true
}
```

## Expected behavior

explicit fp16 intermediate semantics are preserved.

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
