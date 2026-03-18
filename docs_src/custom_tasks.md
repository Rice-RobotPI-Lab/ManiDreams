# Custom Tasks Guide

To integrate a new manipulation task into ManiDreams, implement three components: a **SimulationBackend**, a **Cage**, and a **main script** that wires them together.

## Step 1: Implement a SimulationBackend

Create a backend that tells ManiDreams how to interact with your environment:

```python
from manidreams.physics.simulation_tsip import SimulationBackend
from manidreams.base.dris import DRIS

class MyTaskBackend(SimulationBackend):
    def create_environment(self, env_config):
        return gym.make("MyTask-v1", **env_config)

    def get_state(self, env):
        """Extract state for broadcasting to parallel eval environments."""
        return {
            'object_pose': env.unwrapped.obj.pose.raw_pose,
            'robot_qpos': env.unwrapped.agent.robot.get_qpos(),
        }

    def set_state(self, env, state):
        """Set state on parallel eval environments."""
        if 'object_pose' in state:
            env.unwrapped.obj.set_pose(...)

    def state2dris(self, observations, env_indices=None, env_config=None):
        """Convert environment observations to DRIS objects."""
        dris_list = []
        for i in range(num_envs):
            dris_list.append(DRIS(
                observation=observations[i],
                context={'position': obj_positions[i]},
            ))
        return dris_list

    def step_act(self, actions, env=None, cage=None, single_action=False):
        obs, reward, terminated, truncated, info = env.step(actions)
        return obs

    # Required stubs (minimal implementations):
    def reset_environment(self, env, seed=None, options=None):
        return env.reset(seed=seed, options=options)
    def get_action_space(self, env):
        return env.action_space
    def load_env(self, context): pass
    def load_object(self, context): pass
    def load_robot(self, context): pass
```

## Step 2: Implement a Cage

Define the spatial constraint for your task:

```python
from manidreams.base.cage import Cage
from manidreams.base.dris import DRIS

class MyTaskCage(Cage):
    def __init__(self, target_pos, radius=0.1):
        state_space = gym.spaces.Box(-np.inf, np.inf, shape=(3,))
        super().__init__(state_space, time_varying=False)
        self.target_pos = np.asarray(target_pos)
        self.radius = radius
        self.parameters = {'target_pos': self.target_pos, 'radius': self.radius}
        self.initialized = True

    def _define_parameters(self):
        return {'target_pos': self.target_pos, 'radius': self.radius}

    def _update_from_parameters(self):
        self.target_pos = self.parameters['target_pos']
        self.radius = self.parameters['radius']

    def set_cage(self, region):
        for key in ['target_pos', 'radius']:
            if key in region:
                self.parameters[key] = region[key]
        self._update_from_parameters()

    def initialize(self):
        self.initialized = True

    def evaluate(self, dris_input):
        """Cost = distance to target. Lower is better."""
        dris_list = dris_input if isinstance(dris_input, list) else [dris_input]
        costs = []
        for dris in dris_list:
            pos = dris.context.get('position', dris.observation[:3])
            dist = float(np.linalg.norm(pos - self.target_pos))
            costs.append(dist)
        return costs

    def validate(self, dris_input):
        """Valid if within radius of target."""
        costs = self.evaluate(dris_input)
        return [c <= self.radius for c in costs]

    def get_boundary(self):
        return {'center': self.target_pos.tolist(), 'radius': self.radius}
```

## Step 3: Wire Them Together

```python
from manidreams.physics.simulation_tsip import SimulationBasedTSIP
from manidreams.solvers.samplers.policy_sampler import PolicySampler
from manidreams.env import ManiDreamsEnv

# 1. Create TSIP with your backend
backend = MyTaskBackend()
tsip = SimulationBasedTSIP(
    backend=backend,
    env_config={'num_envs': 8, 'obs_mode': 'state', 'sim_backend': 'gpu'},
)

# 2. Create cage
cage = MyTaskCage(target_pos=[0.5, 0.0, 0.1])

# 3. Load policy and create solver
policy = load_my_policy(checkpoint)
solver = PolicySampler(
    policy_model=policy,
    num_samples=8,        # Evaluate 8 candidates per step
    deterministic=False,  # Sample stochastically for diverse candidates
)

# 4. Create ManiDreams environment
env = ManiDreamsEnv(
    tsip=tsip,
    action_space=tsip.env.action_space,
    solver=solver,
    cage=cage,
    max_timesteps=200,
)

# 5. Dream: plan under cage constraints
obs, info = env.reset()
solver.initialize(env, obs)
trajectory, actions, cage_params = env.dream(horizon=200)

# 6. Execute on independent executor (optional)
for action in actions:
    executor.execute(action)
```

---

## Switching to a Learned World Model

Replace `SimulationBasedTSIP` with `LearningBasedTSIP` for world-model-based planning. The cage and solver remain unchanged:

```python
from manidreams.physics.learned_tsip import LearningBasedTSIP
from examples.physics.push_backend_learned import DiffusionBackend

backend = DiffusionBackend()
tsip = LearningBasedTSIP(
    backend=backend,
    model_config={'checkpoint': 'path/to/diamond_model'},
)

# The rest of the pipeline (Cage, Solver, main loop) stays identical.
```

---

## Using DRIS for Uncertainty-Aware Evaluation

The DRIS framework adds domain randomization on top of standard ManiSkill tasks:

```python
from examples.physics.maniskill_default_tasks import DRISBackend
from manidreams.physics.simulation_tsip import SimulationBasedTSIP
from manidreams.cages.dris_cage import DRISCage

# Create backend with 8 DRIS copies per environment
backend = DRISBackend(
    task_id="PushCube-v1",
    n_dris_copies=8,
    pose_noise=(0.03, 0.03, 0.0, 0.0, 0.0, 0.05),  # dx,dy,dz,droll,dpitch,dyaw
)

tsip = SimulationBasedTSIP(backend=backend, env_config={...})

# After TSIP.next(), each DRIS contains:
#   dris.context['mean_position']  → [3] mean across 8 copies
#   dris.context['variance']       → [3] variance across 8 copies
#   dris.context['dris_poses']     → [8, 7] individual copy poses

# DRISCage uses this for uncertainty-aware evaluation:
cage = DRISCage(
    goal_pos=goal_pos,
    lambda_var=0.1,        # Weight for variance penalty
    success_radius=0.05,
)
# cost = ||mean_pos - goal|| + 0.1 * sum(variance)
```
