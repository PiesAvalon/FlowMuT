"""Adapter registry: ``(framework, mode)`` -> :class:`FrameworkAdapter`."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

from flowmut.adapters.base import FrameworkAdapter
from flowmut.ir.specs import MODES, parse_mode

#: ``(framework, mode)`` -> ``"module:ClassName"``.  Imported lazily so that a
#: PyTorch-only interpreter can still describe the MindSpore campaigns.
ADAPTER_SPECS: Dict[Tuple[str, str], str] = {
    ("pytorch", "eager"): "flowmut.adapters.torch_adapter:PyTorchEagerAdapter",
    ("pytorch", "compiled"): "flowmut.adapters.torch_adapter:PyTorchCompiledAdapter",
    ("mindspore", "pynative"): "flowmut.adapters.mindspore_adapter:MindSporePyNativeAdapter",
    ("mindspore", "graph"): "flowmut.adapters.mindspore_adapter:MindSporeGraphAdapter",
}

_CACHE: Dict[Tuple[str, str], Any] = {}


def adapter_class(framework: str, mode: str):
    key = (framework, mode)
    if key not in ADAPTER_SPECS:
        raise KeyError(f"no adapter registered for {framework}:{mode}")
    if key not in _CACHE:
        module_name, _, class_name = ADAPTER_SPECS[key].partition(":")
        module = importlib.import_module(module_name)
        _CACHE[key] = getattr(module, class_name)
    return _CACHE[key]


def make_adapter(framework: str, mode: str, **kwargs: Any) -> FrameworkAdapter:
    return adapter_class(framework, mode)(mode=mode, **kwargs)


def make_adapter_for(spec: str, **kwargs: Any) -> FrameworkAdapter:
    framework, mode = parse_mode(spec)
    return make_adapter(framework, mode, **kwargs)


def available_modes(probe: bool = True) -> List[Dict[str, Any]]:
    """Report every declared mode and whether this interpreter can run it."""
    out: List[Dict[str, Any]] = []
    for framework, mode in MODES:
        entry: Dict[str, Any] = {"framework": framework, "mode": mode, "key": f"{framework}:{mode}"}
        if not probe:
            out.append(entry)
            continue
        try:
            cls = adapter_class(framework, mode)
            ok, reason = cls.is_available()
        except Exception as exc:
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        entry["available"] = bool(ok)
        entry["reason"] = reason
        out.append(entry)
    return out


def require_available(framework: str, mode: str) -> FrameworkAdapter:
    cls = adapter_class(framework, mode)
    ok, reason = cls.is_available()
    if not ok:
        raise RuntimeError(f"{framework}:{mode} is not available in this interpreter: {reason}")
    return cls(mode=mode)
