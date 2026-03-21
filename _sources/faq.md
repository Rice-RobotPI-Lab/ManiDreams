# FAQ & Troubleshooting

## Frequently Asked Questions

### What is the difference between Baseline and CAGE mode?

**Baseline mode** (`num_samples=1`): The policy outputs a single action, which is directly executed. No TSIP evaluation, no cage checking.

**CAGE mode** (`num_samples > 1`): The policy samples N candidate actions, all are evaluated in parallel via TSIP, scored by the cage, and the best valid action is selected. This adds a safety layer on top of the base policy.

Switching between modes requires only changing the `--num_samples` argument.

### Can I use ManiDreams without ManiSkill?

ManiDreams's core abstractions (DRIS, Cage, TSIPBase, SolverBase) are framework-agnostic. However, the provided `SimulationBasedTSIP` and `SimulationBackend` implementations are built around ManiSkill/SAPIEN environments. To use a different simulator, implement your own `TSIPBase` subclass.

### How do I use a learned world model instead of a physics simulator?

Replace `SimulationBasedTSIP` with `LearningBasedTSIP` and provide a `LearnedBackend` implementation for your model. See the [TSIP](tsip.md) page for details and the Diamond integration as an example.

### What is DRIS and why does it matter?

DRIS (Domain-Randomized Instance Set) is the universal state representation that flows through the entire pipeline. It decouples the observation format from the algorithms, allowing the same cage and solver to work with state vectors, images, or point clouds. See the [DRIS](dris.md) page for details.

### How many parallel environments do I need?

The number of parallel environments (`num_envs`) should match or exceed the number of candidate actions (`num_samples`). Each candidate action is evaluated in a separate environment simultaneously. Typical values: 8–16 for PolicySampler, 16–64 for GeometricOptimizer.

## Troubleshooting

### `mani-skill` installation fails

ManiSkill requires specific system dependencies. Install it separately first:

```bash
pip install mani-skill
```

If GPU simulation is needed, ensure you have a compatible NVIDIA driver and CUDA toolkit installed. See the [ManiSkill documentation](https://maniskill.readthedocs.io/) for detailed requirements.

### CUDA out of memory during `dream()`

Reduce the number of parallel environments (`num_envs`) or the number of candidate actions (`num_samples`). For MPPI, also reduce `num_iterations` or `horizon`.

### Executor environment differs from TSIP results

This is expected behavior. The executor creates an independent environment instance and does not share state with the TSIP. Minor differences in physics stepping or object placement may lead to slightly different outcomes. This decoupling is intentional — see [Why ManiDreams?](why_manidreams.md) for design principles.
