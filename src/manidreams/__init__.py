"""
ManiDreams: Abstract Framework for Cage-Constrained Robotic Manipulation

Framework implementing ManiDreams paper concepts:
- Domain-Randomized State Sets (DRIS) with context support
- Task-Specific Intuitive Physics (TSIP) for state prediction
- Cage constraints as virtual bounds on DRIS to prevent divergence
- Cage-constrained action selection via parallel evaluation
- Concrete implementations for common use cases

This framework provides both abstract interfaces and concrete implementations.
Applications can use provided implementations or define their own.

Version 0.1.0 - Restructured Framework
"""

__version__ = "0.1.0"

# Core base components
from .base import DRIS, Cage, SolverBase, TSIPBase
from .env import ManiDreamsEnv

# Concrete implementations
from .cages import CircularCage
from .physics import SimulationBasedTSIP
from .solvers import GeometricOptimizer
from .solvers.samplers import SamplerBase, DiscreteSampler, GaussianSampler

__all__ = [
    # Core base components
    "DRIS",
    "ManiDreamsEnv",
    "Cage",
    "SolverBase",
    "TSIPBase",

    # Concrete cage implementations
    "CircularCage",

    # Physics implementations
    "SimulationBasedTSIP",

    # Solver implementations
    "GeometricOptimizer",

    # Sampler implementations
    "SamplerBase",
    "DiscreteSampler",
    "GaussianSampler",
]