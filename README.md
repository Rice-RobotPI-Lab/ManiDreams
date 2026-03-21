# [ManiDreams](https://rice-robotpi-lab.github.io/ManiDreams/)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://rice-robotpi-lab.github.io/ManiDreams/)
[![Documentation](https://img.shields.io/badge/Docs-Online-green)](https://rice-robotpi-lab.github.io/ManiDreams/documentation.html)
[![arXiv](https://img.shields.io/badge/arXiv-2603.18336-b31b1b)](https://arxiv.org/abs/2603.18336)

ManiDreams: An Open-Source Library for Robust Object Manipulation via Uncertainty-aware World Models

## Overview

ManiDreams implements a planning paradigm where a virtual constraint (cage) bounds object states during action selection. At each timestep, the framework generates candidate actions, predicts outcomes in parallel via a forward model (TSIP), evaluates them against cage constraints, and executes the best valid action.

**Core abstractions:**
- **DRIS** — Domain-Randomized Instance Set: universal state representation
- **TSIP** — Task-Specific Intuitive Physics: forward model (simulation or learned)
- **Cage** — Constraints with cost evaluation and validation
- **Solver** — Action selection via sampling or trajectory optimization

## Installation

```bash
git clone https://github.com/Rice-RobotPI-Lab/ManiDreams
cd ManiDreams
pip install -e .
```

Optional extras: `catching`, `diffusion`, `ppo`, `eval`, `docs`, or `all`:
```bash
pip install -e ".[all]"
```

## Quick Start

```python
import gymnasium as gym
from manidreams.env import ManiDreamsEnv
from manidreams.cages.geometric import CircularCage
from manidreams.solvers.optimizers.geometric_optimizer import GeometricOptimizer
from manidreams.physics.simulation_tsip import SimulationBasedTSIP

# 1. Create TSIP with your simulation backend
tsip = SimulationBasedTSIP(backend=my_backend, env_config={...})

# 2. Create cage constraint
cage = CircularCage(state_space=state_space, center=[0, 0], radius=0.28,
                    time_varying=True, orbit_radius=0.2, orbit_speed=0.1)

# 3. Create solver and environment
solver = GeometricOptimizer(config={'horizon': 1, 'num_trajectories': 16})
env = ManiDreamsEnv(tsip=tsip, action_space=action_space, cage=cage, solver=solver)

# 4. Plan under cage constraints
env.reset(seed=42)
trajectory, actions, cage_history = env.dream(horizon=50)
```

See `examples/tasks/` for complete runnable scripts (pushing, catching, picking).

## Documentation

Full documentation is available at the [docs site](https://rice-robotpi-lab.github.io/ManiDreams/), including:

- [Why ManiDreams?](https://rice-robotpi-lab.github.io/ManiDreams/why_manidreams.html) — Architecture and design principles
- [Core Concepts](https://rice-robotpi-lab.github.io/ManiDreams/dris.html) — DRIS, Cage, TSIP, Solvers
- [Supported Tasks](https://rice-robotpi-lab.github.io/ManiDreams/supported_tasks.html) — Object pushing, catching, picking, and more
- [API Reference](https://rice-robotpi-lab.github.io/ManiDreams/api/index.html)

## Project Structure

```
src/manidreams/          # Core package
├── base/                # Abstract interfaces (DRIS, TSIPBase, Cage, SolverBase)
├── cages/               # Cage implementations (Circular, DRIS, Plate, Pixel, etc.)
├── solvers/             # Samplers (PolicySampler) + Optimizers (MPPI, Geometric)
├── physics/             # TSIP implementations (Simulation, Learned)
├── executors/           # Action execution (Simulation, Real)
└── env.py               # ManiDreamsEnv: Gym-compatible interface with dream()

examples/                # Task-specific integration
├── physics/             # Backend wrappers (ManiSkill, Diamond diffusion)
├── samplers/            # Trained policies and checkpoints
├── tasks/               # Main scripts (pushing, catching, picking)
└── executors/           # Task-specific executors
```

## License

MIT License
