"""
Task-specific hyperparameter presets for ManiSkill PPO training.

Presets are extracted from verified training commands in
resources/baselines/ppo/examples.sh and the ManiSkill documentation.

ppo.py default values (used when not overridden by a preset):
    env_id            = "PickCube-v1"
    num_envs          = 512
    num_eval_envs     = 8
    total_timesteps   = 10_000_000
    learning_rate     = 3e-4
    num_steps         = 50
    num_eval_steps    = 50
    gamma             = 0.8
    gae_lambda        = 0.9
    update_epochs     = 4
    num_minibatches   = 32
    clip_coef         = 0.2
    ent_coef          = 0.0
    vf_coef           = 0.5
    max_grad_norm     = 0.5
    target_kl         = 0.1
    reward_scale      = 1.0
    eval_freq         = 25
    seed              = 1
    control_mode      = "pd_joint_delta_pos"
    obs_mode          = "state"
    sim_backend       = "physx_cuda"

See: https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/index.html
"""

TASK_PRESETS = {
    "PickCube-v1": {
        # max_episode_steps=50
        "num_envs": 1024,
        "total_timesteps": 50_000_000,
        "num_steps": 100,
        "num_eval_steps": 100,
        "gamma": 0.9,
        "ent_coef": 0.02,
        "eval_freq": 10,
        "update_epochs": 8,
        "num_minibatches": 32,
        "control_mode": "pd_ee_delta_pos",
    },
    "PushCube-v1": {
        # max_episode_steps=50
        "num_envs": 2048,
        "total_timesteps":8_000_000,
        "num_steps": 20,
        "num_eval_steps": 20,
        "eval_freq": 20,
        "update_epochs": 8,
        "num_minibatches": 32,
        "control_mode": "pd_ee_delta_pos",
    },
    "PushT-v1": {
        # max_episode_steps=100
        "num_envs": 1024,
        "total_timesteps": 25_000_000,
        "num_steps": 100,
        "num_eval_steps": 100,
        "gamma": 0.99,
        "eval_freq": 20,
        "update_epochs": 8,
        "num_minibatches": 32,
        "control_mode": "pd_joint_delta_pos",
    },
    "PullCube-v1": {
        # max_episode_steps=50
        "num_envs": 1024,
        "total_timesteps": 10_000_000,
        "update_epochs": 8,
        "num_minibatches": 32,
    },
    "RollBall-v1": {
        # max_episode_steps=80
        "num_envs": 1024,
        "total_timesteps": 20_000_000,
        "num_steps": 80,
        "num_eval_steps": 80,
        "gamma": 0.95,
        "update_epochs": 8,
        "num_minibatches": 32,
    },
}

# Default task when --task is not specified
DEFAULT_TASK = "PushCube-v1"

# ppo.py default Args values (for reference / display)
PPO_DEFAULTS = {
    "env_id": "PickCube-v1",
    "num_envs": 512,
    "num_eval_envs": 8,
    "total_timesteps": 10_000_000,
    "learning_rate": 3e-4,
    "num_steps": 50,
    "num_eval_steps": 50,
    "gamma": 0.8,
    "gae_lambda": 0.9,
    "update_epochs": 4,
    "num_minibatches": 32,
    "clip_coef": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.1,
    "reward_scale": 1.0,
    "eval_freq": 25,
    "seed": 1,
    "control_mode": "pd_joint_delta_pos",
}


def get_preset(task_id: str) -> dict:
    """Get hyperparameter preset for a task.

    Args:
        task_id: ManiSkill task ID (e.g., "PushCube-v1")

    Returns:
        Dictionary of hyperparameter overrides

    Raises:
        ValueError: If task_id is not supported
    """
    if task_id not in TASK_PRESETS:
        raise ValueError(
            f"Task '{task_id}' not supported. "
            f"Supported tasks: {list(TASK_PRESETS.keys())}"
        )
    return TASK_PRESETS[task_id].copy()


def list_tasks() -> list:
    """List all supported task IDs."""
    return list(TASK_PRESETS.keys())
