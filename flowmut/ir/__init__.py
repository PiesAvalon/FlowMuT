"""IR package: the framework-neutral graph that FlowMuT profiles and mutates."""

from flowmut.ir.graph import Edge, Graph, GraphBuilder, Node  # noqa: F401
from flowmut.ir.ops import (  # noqa: F401
    OP_REGISTRY,
    OpDef,
    attach_requirements,
    canonical_op,
    get_op,
    infer_shape,
    op_names,
    requirements_for,
)
from flowmut.ir.specs import (  # noqa: F401
    DTYPES,
    FRAMEWORKS,
    MODES,
    OpRequirement,
    TensorSpec,
    ValueProfile,
    broadcast_shape,
    parse_mode,
    shape_numel,
)

__all__ = [
    "Graph", "GraphBuilder", "Node", "Edge",
    "TensorSpec", "ValueProfile", "OpRequirement",
    "OP_REGISTRY", "OpDef", "canonical_op", "get_op", "op_names",
    "requirements_for", "infer_shape", "attach_requirements",
    "DTYPES", "FRAMEWORKS", "MODES", "parse_mode",
    "broadcast_shape", "shape_numel",
]
