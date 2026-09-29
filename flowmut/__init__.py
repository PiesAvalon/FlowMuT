"""FlowMuT: data-flow-guided mutation testing for deep learning frameworks.

FlowMuT profiles a seed model, builds a *Tensor Flow Graph* (TFG) that combines
operator topology ``G``, runtime tensor annotations ``F`` and operator input
requirements ``R``, then repeatedly mutates the seed model through constrained
subgraph replacement, executes the mutant on a target framework/mode, and uses a
contextual bandit (LinUCB) to steer operator/site selection towards novel
framework behaviour.

Package layout
--------------
``flowmut.ir``          framework-neutral model graph (the IR that is mutated)
``flowmut.tfg``         Tensor Flow Graph construction and annotation
``flowmut.adapters``    framework/mode adapters (PyTorch, MindSpore)
``flowmut.seed_models`` the 14 benchmark seed models
``flowmut.operators``   the mutation operator pool
``flowmut.selection``   candidate features and bandit policies
``flowmut.oracle``      legality checking and bug oracles
``flowmut.loop``        the FlowMuT main loop
"""

__version__ = "1.0.0"

from flowmut.ir.graph import Edge, Graph, GraphBuilder, Node  # noqa: F401
from flowmut.tfg.tfg import TFG  # noqa: F401

__all__ = ["__version__", "Graph", "GraphBuilder", "Node", "Edge", "TFG"]
