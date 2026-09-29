# permute + inplace slice update + transpose mutates the wrong value

## 🐛 Describe the bug

On CPU, Inductor produces an incorrect value after a `permute`, in-place slice assignment, and `transpose` sequence. One output element differs from eager execution.

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

def probe_017_permute_slice_inplace_transpose() -> dict[str, Any]:
    def fn(x):
        x = x.permute(2, 1, 0)
        x[0] += 1
        x[0] = x[0].t()
        return x

    x = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    eager = fn(x.clone())
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)(x.clone())
    return {
        "reproduced": not torch.equal(eager, compiled),
        "eager": repr(eager),
        "compiled": repr(compiled),
        "metric": {"mismatched": int(eager.ne(compiled).sum())},
    }


if __name__ == "__main__":
    result = probe_017_permute_slice_inplace_transpose()
    print(json.dumps(result, indent=2, default=str))
    assert result["reproduced"]

```

## Actual behavior

Eager result:

```text
tensor([[[2., 6.],
         [6., 8.]],

        [[2., 6.],
         [4., 8.]]])
```

Compiled result:

```text
tensor([[[2., 4.],
         [6., 8.]],

        [[2., 6.],
         [4., 8.]]])
```

Measured difference:

```json
{
  "mismatched": 1
}
```

## Expected behavior

all updated tensor values agree.

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
