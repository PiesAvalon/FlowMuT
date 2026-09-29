# Inductor drops forward-mode AD tangent

## 🐛 Describe the bug

On CPU, the compiled function drops the forward-mode automatic differentiation tangent: eager execution returns a tensor, while Inductor returns `None` for the same input.

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

def probe_002_forward_ad_tangent() -> dict[str, Any]:
    import torch.autograd.forward_ad as fw_ad

    def fn(x):
        return (x**2).sum()

    x = torch.tensor([0.1, 0.2, 0.3])
    tangent = torch.ones(3)
    with fw_ad.dual_level():
        eager = fw_ad.unpack_dual(fn(fw_ad.make_dual(x, tangent))).tangent
    with fw_ad.dual_level():
        compiled_output = torch.compile(
            fn, backend="inductor", fullgraph=True
        )(fw_ad.make_dual(x, tangent))
        compiled = fw_ad.unpack_dual(compiled_output).tangent
    return {
        "reproduced": eager is not None and compiled is None,
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"tangent_dropped": compiled is None},
    }


if __name__ == "__main__":
    result = probe_002_forward_ad_tangent()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor(1.2000)
```

Compiled result:

```text
None
```

Measured difference:

```json
{
  "tangent_dropped": true
}
```

## Expected behavior

JVP tangent is preserved.

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
