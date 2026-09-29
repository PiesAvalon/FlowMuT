# CPU acosh returns inf for large finite float32 inputs when compiled

## 🐛 Describe the bug

On CPU, compiling `torch.acosh` turns several finite float32 outputs into `inf` for large but finite inputs. Eager execution returns finite values for the same inputs.

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

def probe_010_acosh_large_finite() -> dict[str, Any]:
    def fn(x):
        return torch.acosh(x)

    x = torch.tensor([5e22, 9e25, 7e21, 2.0], dtype=torch.float32)
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x)
    incorrect_infinity = torch.isfinite(eager) & torch.isinf(compiled)
    return {
        "reproduced": bool(incorrect_infinity.any()),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"finite_to_inf": int(incorrect_infinity.sum())},
    }


if __name__ == "__main__":
    result = probe_010_acosh_large_finite()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([52.9595, 60.4550, 50.9933,  1.3170])
```

Compiled result:

```text
tensor([   inf,    inf,    inf, 1.3170])
```

Measured difference:

```json
{
  "finite_to_inf": 3
}
```

## Expected behavior

finite input with representable result stays finite.

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
