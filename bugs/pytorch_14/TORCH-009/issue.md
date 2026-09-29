# CUDA int64 arange multiplication overflows as int32

## 🐛 Describe the bug

On CUDA, multiplying an int64 `torch.arange` by a large integer produces incorrect negative values under Inductor. Eager execution retains the expected int64 results.

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

def probe_009_arange_int64_overflow() -> dict[str, Any]:
    def fn(anchor):
        return torch.arange(
            9, device=anchor.device, dtype=torch.int64
        ) * torch.tensor([1_500_000_000], device=anchor.device, dtype=torch.int64)

    anchor = torch.zeros(1, device="cuda")
    eager = fn(anchor)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(anchor)
    return {
        "reproduced": not torch.equal(eager, compiled),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {
            "mismatched": int(eager.ne(compiled).sum()),
            "max_abs": int((eager - compiled).abs().max()),
        },
    }


if __name__ == "__main__":
    result = probe_009_arange_int64_overflow()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([          0,  1500000000,  3000000000,  4500000000,  6000000000,
         7500000000,  9000000000, 10500000000, 12000000000], device='cuda:0')
```

Compiled result:

```text
tensor([          0,  1500000000, -1294967296,   205032704,  1705032704,
        -1089934592,   410065408,  1910065408,  -884901888], device='cuda:0')
```

Measured difference:

```json
{
  "mismatched": 7,
  "max_abs": 12884901888
}
```

## Expected behavior

int64 arithmetic is preserved.

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
