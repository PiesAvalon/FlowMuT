# fractional_max_pool2d consumes RNG differently when compiled

## 🐛 Describe the bug

On CPU, `fractional_max_pool2d` selects different pooling indices under Inductor even when the random seed is reset to the same value before the eager and compiled calls. Ten indices differ in the reproduced output.

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

def probe_018_fractional_max_pool_rng() -> dict[str, Any]:
    def fn(x):
        return F.fractional_max_pool2d(
            x, 3, output_size=(4, 4), return_indices=True
        )

    torch.manual_seed(18)
    x = torch.randn(1, 3, 10, 10)
    torch.manual_seed(0)
    eager = fn(x)
    torch.manual_seed(0)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    index_difference = eager[1].ne(compiled[1])
    return {
        "reproduced": bool(index_difference.any()),
        "eager": f"values={eager[0].flatten()[:8]!r}, indices={eager[1].flatten()[:8]!r}",
        "compiled": (
            f"values={compiled[0].flatten()[:8]!r}, "
            f"indices={compiled[1].flatten()[:8]!r}"
        ),
        "metric": {"index_mismatched": int(index_difference.sum())},
    }


if __name__ == "__main__":
    result = probe_018_fractional_max_pool_rng()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
values=tensor([0.9802, 1.5740, 1.0297, 1.0842, 2.1582, 1.9337, 1.9337, 1.6727]), indices=tensor([ 2,  3,  6,  9, 30, 44, 44, 49])
```

Compiled result:

```text
values=tensor([0.9802, 1.5740, 1.0297, 1.0842, 2.1582, 1.9337, 1.9337, 1.6727]), indices=tensor([ 2,  3,  6,  9, 30, 44, 44, 49])
```

Measured difference:

```json
{
  "index_mismatched": 10
}
```

The excerpts above show only the first eight indices; the full outputs differ at ten positions.

## Expected behavior

same seed selects the same pooling regions.

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
