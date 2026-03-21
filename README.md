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

### Object Pushing (Simulation)

Multi-object herding with orbital cage constraints in parallel ManiSkill environments.

```bash
python examples/tasks/object_pushing/main.py
```

### Object Pushing (Diffusion World Model)

Cage-constrained planning using a learned diffusion world model. Requires downloading model weights (see [Model Weights](#model-weights)).

```bash
python examples/tasks/object_pushing/main_pixel.py
```

With iterative feedback loop between diffusion planning and real execution:

```bash
python examples/tasks/object_pushing/main_pixel_feedback.py
```

### Object Picking

MPPI-based trajectory optimization with 3D cage constraints for pick-and-place.

```bash
python examples/tasks/object_picking/main.py
```

### Object Catching

Plate-based ball catching with optional CAGE enhancement.

```bash
# Baseline (direct policy)
python examples/tasks/object_catching/main.py --num_samples 0

# CAGE mode (8 candidate actions, 16 DRIS copies)
python examples/tasks/object_catching/main.py --num_samples 8 --num_objs_tsip 16
```

### ManiSkill Default Tasks

PushCube, PickCube, and PushT with PPO policies and optional CAGE enhancement.

```bash
python examples/tasks/maniskill_defaults/main.py --task PushCube-v1
python examples/tasks/maniskill_defaults/main.py --task PickCube-v1
python examples/tasks/maniskill_defaults/main.py --task PushT-v0
```

Use `--num_samples 0` for baseline (direct policy) or `--num_samples 16` for CAGE mode.

### Zero-shot Real-to-Sim Demo

Real-time object detection (D415 + stereo) with domain-randomized Newton physics simulation. See [setup tutorial](https://rice-robotpi-lab.github.io/ManiDreams/zeroshot_real2sim.html) and [demo videos](https://rice-robotpi-lab.github.io/ManiDreams/index.html#dris-real2sim).

## Model Weights

The pixel-based pushing task requires a diffusion model checkpoint. Download `push16.pt` from [Google Drive](https://drive.google.com/file/d/1OBTPrz3g2i7OzF2M0-Zdt8ISFGINJhnG/view?usp=sharing) and place it at:

```
examples/physics/push_backend_learned/models/push16/model/push16.pt
```

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
├── physics/             # Backend wrappers (ManiSkill, Newton, Diamond diffusion)
├── samplers/            # Trained policies and checkpoints
├── tasks/               # Main scripts (pushing, catching, picking, real2sim)
└── executors/           # Task-specific executors (simulation, D415+FFS)
```

## License

MIT License
