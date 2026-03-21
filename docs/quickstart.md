# Quick Start

## Object Pushing (Simulation)

Multi-object herding with orbital cage constraints in parallel ManiSkill environments.

```bash
python examples/tasks/object_pushing/main.py
```

## Object Pushing (Diffusion World Model)

Cage-constrained planning using a learned diffusion world model. Requires downloading model weights (see [Model Weights](installation.md)).

```bash
python examples/tasks/object_pushing/main_pixel.py
```

With iterative feedback loop between diffusion planning and real execution:

```bash
python examples/tasks/object_pushing/main_pixel_feedback.py
```

## Object Picking

MPPI-based trajectory optimization with 3D cage constraints for pick-and-place.

```bash
python examples/tasks/object_picking/main.py
```

## Object Catching

Plate-based ball catching with optional CAGE enhancement.

```bash
# Baseline (direct policy)
python examples/tasks/object_catching/main.py --num_samples 0

# CAGE mode (8 candidate actions, 16 DRIS copies)
python examples/tasks/object_catching/main.py --num_samples 8 --num_objs_tsip 16
```

## ManiSkill Default Tasks

PushCube, PickCube, and PushT with PPO policies and optional CAGE enhancement.

```bash
python examples/tasks/maniskill_defaults/main.py --task PushCube-v1
python examples/tasks/maniskill_defaults/main.py --task PickCube-v1
python examples/tasks/maniskill_defaults/main.py --task PushT-v0
```

Use `--num_samples 0` for baseline (direct policy) or `--num_samples 16` for CAGE mode.

---

For detailed task descriptions and component configurations, see [Runnable Examples](runnable_examples.md) and [ManiSkill Defaults](maniskill_defaults.md). To create your own task, see [Custom Tasks](custom_tasks.md).
