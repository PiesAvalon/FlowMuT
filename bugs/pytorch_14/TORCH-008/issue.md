# CUDA bf16 threshold compares against an uncast scalar

## 🐛 Describe the bug

On CUDA, Inductor evaluates the `F.threshold` condition differently for a bfloat16 tensor. Two output values differ from eager execution because the threshold scalar is not converted to bfloat16 before comparison.

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

def probe_008_threshold_bf16_scalar_cast() -> dict[str, Any]:
    def fn(x):
        return F.threshold(x, 0.003, 0.0)

    x = torch.tensor(
        [0.00299, 0.003, 0.00301], device="cuda", dtype=torch.bfloat16
    )
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    return {
        "reproduced": not torch.equal(eager, compiled),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"mismatched": int(eager.ne(compiled).sum())},
    }


if __name__ == "__main__":
    result = probe_008_threshold_bf16_scalar_cast()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([0., 0., 0.], device='cuda:0', dtype=torch.bfloat16)
```

Compiled result:

```text
tensor([0.0000, 0.0030, 0.0030], device='cuda:0', dtype=torch.bfloat16)
```

Measured difference:

```json
{
  "mismatched": 2
}
```

## Expected behavior

threshold uses bf16-cast scalar semantics.

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
