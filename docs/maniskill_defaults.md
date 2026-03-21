# ManiSkill Defaults

Standard ManiSkill benchmark tasks with PPO policies and optional CAGE enhancement. These demonstrate how ManiDreams wraps any existing RL policy with uncertainty-aware action selection.

---

## Common Architecture

All three tasks share the same component structure:

| Component | Implementation | Details |
|---|---|---|
| TSIP Backend | `DRISBackend` | ManiSkill vectorized env with DRIS state conversion |
| TSIP | `SimulationBasedTSIP` | Parallel rollout across DRIS copies |
| Cage | `DRISCage` | Goal-conditioned cost with variance penalty (lambda_var=0.1) |
| Solver | `PolicySampler` | Samples from PPO policy distribution |
| Sampler | `PPOPolicyAdapter` | Wraps trained PPO `ActorCritic` for stochastic sampling |
| Executor | ManiSkill `VectorEnv` | Standard gym vectorized environment |

**Two modes:**
- **Baseline** (`--num_samples 0`): Deterministic policy output, no CAGE evaluation
- **CAGE** (`--num_samples N`): Sample N actions from policy distribution, evaluate via TSIP + DRISCage, select best

---

## PushCube

Push a cube to a target location on a tabletop.

```bash
# Baseline
python examples/tasks/maniskill_defaults/main.py --task PushCube-v1

# CAGE mode
python examples/tasks/maniskill_defaults/main.py --task PushCube-v1 \
    --num_samples 16 --n_dris_copies 16
```

## PickCube

Pick up a cube and move it to a goal position.

```bash
# Baseline
python examples/tasks/maniskill_defaults/main.py --task PickCube-v1

# CAGE mode with perturbation
python examples/tasks/maniskill_defaults/main.py --task PickCube-v1 \
    --num_samples 16 --n_dris_copies 8 \
    --lambda_var 0.2 \
    --pose_noise 0.05 0.05 0.0 0.0 0.0 0.1
```

## PushT

Push a T-shaped block to match a target pose.

```bash
# Baseline
python examples/tasks/maniskill_defaults/main.py --task PushT-v0

# CAGE mode
python examples/tasks/maniskill_defaults/main.py --task PushT-v0 \
    --num_samples 8 --n_dris_copies 4
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `PushCube-v1` | ManiSkill task ID |
| `--num_samples` | `0` | Number of candidate actions (0 = baseline) |
| `--n_dris_copies` | `16` | DRIS copies per evaluation environment |
| `--action_chunk` | `2` | Multi-step action chunking |
| `--lambda_var` | `0.1` | Variance penalty weight in DRISCage |
| `--pose_noise` | `[0.02,0.02,0,0,0,0.15]` | Pose perturbation for domain randomization |
| `--physics_noise` | `[0.2,0.3]` | Mass and friction perturbation |
| `--checkpoint` | auto-detected | Path to PPO model checkpoint |
| `--num_episodes` | `20` | Number of evaluation episodes |

Checkpoints are auto-detected from `examples/samplers/maniskill_defaults/ckpts/`.
