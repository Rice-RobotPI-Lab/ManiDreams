# Installation Guide

## Prerequisites

- Python >= 3.8
- Core dependencies: `numpy`, `gymnasium`, `mani-skill`, `opencv-python`, `transforms3d`

## Install from Source

```bash
cd /path/to/ManiDreams
pip install -e .
```

## Optional Dependencies

Install task-specific extras as needed:

```bash
# Object catching (RL policy solver)
pip install -e ".[catching]"

# Diamond diffusion world model (pixel-based pushing)
pip install -e ".[diffusion]"

# PPO training and evaluation
pip install -e ".[ppo]"

# Robustness evaluation scripts
pip install -e ".[eval]"

# Documentation tools
pip install -e ".[docs]"

# Everything
pip install -e ".[all]"
```

## Verify Installation

```bash
python examples/tasks/object_pushing/main.py
```
