# Inductor changes int8 batched matmul output dtype to int64

## 🐛 Describe the bug

On CPU, `torch.compile(..., backend="inductor")` changes the output of an int8 batched `torch.matmul` from int8 to int64. The values remain the same, but the output dtype differs from eager execution.

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

def probe_001_int8_bmm_dtype() -> dict[str, Any]:
    def fn(a, b):
        return torch.matmul(a, b)

    a = torch.ones((1, 1, 2), dtype=torch.int8)
    b = torch.ones((1, 2, 2), dtype=torch.int8)
    eager = fn(a, b)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
    return {
        "reproduced": eager.dtype != compiled.dtype,
        "eager": f"{eager!r}, dtype={eager.dtype}",
        "compiled": f"{compiled!r}, dtype={compiled.dtype}",
        "metric": {"dtype_equal": eager.dtype == compiled.dtype},
    }


if __name__ == "__main__":
    result = probe_001_int8_bmm_dtype()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([[[2, 2]]], dtype=torch.int8), dtype=torch.int8
```

Compiled result:

```text
tensor([[[2, 2]]]), dtype=torch.int64
```

Measured difference:

```json
{
  "dtype_equal": false
}
```

## Expected behavior

output dtype and values agree.

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
