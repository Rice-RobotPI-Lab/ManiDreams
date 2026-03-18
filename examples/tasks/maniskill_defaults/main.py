"""
ManiSkill Tabletop Tasks - Baseline and CAGE-enhanced Execution

Supports two execution modes controlled by --num_samples:

1. CAGE mode (--num_samples N, N > 0, default N=8):
   - Executor environment + N parallel TSIP evaluation environments (with DRIS copies)
   - Samples N candidate actions from PPO policy distribution
   - Evaluates them via TSIP simulation + DRISCage cost
   - Selects the best action (lowest cost)
   - Two visualization windows

2. Baseline mode (--num_samples 0):
   - Single executor environment
   - Direct PPO policy output (deterministic mean)
   - One visualization window

Usage:
    cd ManiDreams/

    # Single episode demo (default)
    python examples/tasks/maniskill_defaults/main.py --task PushCube-v1

    # Multiple episodes to measure success rate
    python examples/tasks/maniskill_defaults/main.py --task PushCube-v1 --num_episodes 10

    # CAGE mode with action chunks (multi-step planning)
    python examples/tasks/maniskill_defaults/main.py --task PushCube-v1 --action_chunk 4

    # Baseline mode (no TSIP)
    python examples/tasks/maniskill_defaults/main.py \
        --task PushCube-v1 --num_samples 0

    # Explicit checkpoint
    python examples/tasks/maniskill_defaults/main.py \
        --task PushCube-v1 --checkpoint /path/to/checkpoint.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Path constants (CWD-independent)
SCRIPT_DIR = Path(__file__).resolve().parent          # maniskill_defaults/
MANIDREAMS_ROOT = SCRIPT_DIR.parents[2]                 # ManiDreams/
CKPTS_DIR = MANIDREAMS_ROOT / "examples" / "samplers" / "maniskill_defaults" / "ckpts"

# Auto-setup PYTHONPATH
sys.path.insert(0, str(MANIDREAMS_ROOT))
sys.path.insert(0, str(MANIDREAMS_ROOT / "src"))

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from examples.samplers.maniskill_defaults import Agent
from examples.samplers.maniskill_defaults.task_presets import DEFAULT_TASK, get_preset
from examples.physics.maniskill_default_tasks.task_config import get_task_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint auto-detection
# ---------------------------------------------------------------------------
def find_checkpoint(task_id, explicit_path=None):
    """Find checkpoint from examples/samplers/maniskill_defaults/ckpts/."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            logger.error(f"Checkpoint not found: {explicit_path}")
            sys.exit(1)
        return str(p)

    # Map task_id to ckpt filename: "PushCube-v1" -> "pushcube.pt"
    ckpt_name = task_id.split("-")[0].lower() + ".pt"
    ckpt_path = CKPTS_DIR / ckpt_name

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        if CKPTS_DIR.exists():
            available = [f.name for f in CKPTS_DIR.glob("*.pt")]
            logger.error(f"  Available: {available}")
        sys.exit(1)

    logger.info(f"  Auto-detected checkpoint: {ckpt_path}")
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# PPO Policy Adapter for PolicySampler interface
# ---------------------------------------------------------------------------
class PPOPolicyAdapter:
    """Adapt PPO Agent to PolicySampler's expected policy interface.

    PolicySampler calls policy.get_action(latent, tcpn, deterministic).
    PPO Agent has get_action(x, deterministic). This adapter bridges the gap.
    """

    def __init__(self, agent):
        self.agent = agent

    def get_action(self, x, tcpn=None, deterministic=False):
        return self.agent.get_action(x, deterministic=deterministic)


class GripperLockMapper:
    """Lock gripper action to fully closed (-1.0) for push tasks."""

    def register_ref_obs(self, obs):
        pass

    def map(self, obs, action):
        if isinstance(action, torch.Tensor):
            action = action.clone()
        else:
            action = np.array(action, copy=True)
        action[..., -1] = -1.0
        if isinstance(action, torch.Tensor):
            result = action.cpu().numpy()
        else:
            result = action
        # Squeeze spurious leading batch dim: (1, action_dim) -> (action_dim,)
        if result.ndim == 2 and result.shape[0] == 1:
            result = result[0]
        return result


class ObsTruncateMapper:
    """Truncate TSIP observations to match executor obs dimension.

    The TSIP (DRIS) env and the executor env (via ManiSkillVectorEnv) may
    have different obs dimensions. This mapper truncates to the executor's
    obs size when re-encoding latent during chunked planning.
    """

    def __init__(self):
        self.exec_obs_dim = None

    def register_ref_obs(self, obs):
        self.exec_obs_dim = obs.shape[-1]

    def map(self, obs):
        if self.exec_obs_dim is not None and obs.shape[-1] > self.exec_obs_dim:
            obs = obs[..., :self.exec_obs_dim]
        return obs, None


# ---------------------------------------------------------------------------
# State extraction: executor -> state dict for backend.set_state()
# ---------------------------------------------------------------------------
def get_executor_state(exec_env, task_config):
    """Extract robot + object state from the executor environment.

    Returns a state dict compatible with DRISBackend.set_state(),
    which handles broadcasting (1 -> N) and DRIS re-randomization.
    """
    base = exec_env.unwrapped if hasattr(exec_env, 'unwrapped') else exec_env
    while hasattr(base, '_env'):
        base = base._env
    if hasattr(base, 'unwrapped'):
        base = base.unwrapped

    def to_numpy(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

    robot = base.agent.robot
    target = getattr(base, task_config.target_attr)
    goal_obj = getattr(base, task_config.goal_attr)

    return {
        'qpos': to_numpy(robot.get_qpos()),        # [1, dof]
        'qvel': to_numpy(robot.get_qvel()),        # [1, dof]
        'target_pose': to_numpy(target.pose.raw_pose),  # [1, 7]
        'goal_pose': to_numpy(goal_obj.pose.raw_pose),  # [1, 7]
    }


def get_goal_position(env, task_config):
    """Extract goal position from the environment after reset."""
    base = env.unwrapped if hasattr(env, 'unwrapped') else env
    while hasattr(base, '_env'):
        base = base._env
    if hasattr(base, 'unwrapped'):
        base = base.unwrapped

    goal_obj = getattr(base, task_config.goal_attr)
    goal_pos = goal_obj.pose.p
    if hasattr(goal_pos, 'cpu'):
        goal_pos = goal_pos.cpu().numpy()
    return np.asarray(goal_pos).flatten()[:3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ManiSkill Tabletop Task -- Baseline and CAGE-enhanced Execution",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PPO checkpoint (.pt). Auto-detected from samplers/maniskill_defaults/ckpts/ if omitted")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of CAGE action samples (default 8). 0=baseline, N>0=CAGE mode")
    parser.add_argument("--n_dris_copies", type=int, default=16,
                        help="Number of DRIS copies per eval env")
    parser.add_argument("--lambda_var", type=float, default=0.1,
                        help="DRIS variance penalty weight in cost function")
    parser.add_argument("--pose_noise", type=float, nargs=6,
                        default=[0.02, 0.02, 0.0, 0.0, 0.0, 0.15],
                        help="DRIS pose noise (dx dy dz droll dpitch dyaw)")
    parser.add_argument("--physics_noise", type=float, nargs=2,
                        default=[0.2, 0.3],
                        metavar=("DFRIC", "DMASS_RATIO"),
                        help="DRIS physics noise: friction +/-delta and mass multiplier range "
                             "(e.g. --physics_noise 0.2 0.3 -> fric in [base-0.2,base+0.2], "
                             "mass in [base*0.7,base*1.3]). Default: no noise.")
    parser.add_argument("--control_mode", type=str, default=None,
                        help="Control mode (must match training)")
    parser.add_argument("--action_chunk", type=int, default=2,
                        help="Action chunk length (1=single-step, >1=multi-step trajectory planning)")
    parser.add_argument("--num_episodes", type=int, default=20,
                        help="Number of episodes to run (default: 1)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Max episode duration in seconds (fallback if env has no max_episode_steps)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tsip_camera", type=str, default="closeup",
                        choices=["closeup", "topdown"],
                        help="TSIP viewer camera preset")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    use_cage = args.num_samples > 0
    task_config = get_task_config(args.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = find_checkpoint(args.task, args.checkpoint)

    # Lock gripper to fully closed for PushCube only (PPO was trained with open gripper for PushT)
    lock_gripper = args.task == "PushCube-v1"

    # Auto-detect control_mode from task preset if not specified
    if args.control_mode is None:
        preset = get_preset(args.task)
        args.control_mode = preset.get("control_mode")
        if args.control_mode:
            logger.info(f"  Control mode from preset: {args.control_mode}")

    logger.info("=" * 60)
    logger.info(f"ManiSkill Tabletop -- {'CAGE' if use_cage else 'Baseline'} Mode")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Episodes: {args.num_episodes}")
    if use_cage:
        logger.info(f"  Candidates: {args.num_samples}, DRIS copies: {args.n_dris_copies}")
        logger.info(f"  Lambda variance: {args.lambda_var}")
        logger.debug(f"  Pose noise: {args.pose_noise}")
        logger.debug(f"  Physics noise: dfric={args.physics_noise[0]}, dmass_ratio={args.physics_noise[1]}")
        if args.action_chunk > 1:
            logger.info(f"  Action chunk: {args.action_chunk} steps")
    logger.info("=" * 60)

    # ==================================================================
    # Step 1: Create TSIP eval environment (CAGE mode only)
    # ==================================================================
    tsip = None
    tsip_backend = None

    if use_cage:
        from manidreams.physics.simulation_tsip import SimulationBasedTSIP
        from examples.physics.maniskill_default_tasks.dris_backend import DRISBackend

        logger.info("Step 1: Creating TSIP with DRIS evaluation environments...")
        tsip_backend = DRISBackend(
            task_id=args.task,
            n_dris_copies=args.n_dris_copies,
            pose_noise=tuple(args.pose_noise),
            physics_noise=tuple(args.physics_noise),
        )
        tsip_env_config = {
            'num_envs': args.num_samples,
            'obs_mode': 'state',
            'render_mode': 'human',
            'sim_backend': 'gpu',
            'parallel_in_single_scene': True,
            'sim_config': dict(spacing=1.2),
        }
        if args.control_mode:
            tsip_env_config['control_mode'] = args.control_mode

        tsip = SimulationBasedTSIP(
            backend=tsip_backend,
            env_config=tsip_env_config,
        )
        logger.info(f"  TSIP: {args.num_samples} envs x {args.n_dris_copies} DRIS copies")
    else:
        logger.info("Step 1: Skipping TSIP (baseline mode)")

    # ==================================================================
    # Step 2: Create executor environment (standard, no DRIS)
    # ==================================================================
    logger.info("Step 2: Creating executor environment...")
    exec_kwargs = dict(
        num_envs=1,
        obs_mode='state',
        render_mode='human',
        sim_backend='gpu',
        sensor_configs=dict(shader_pack='rt-fast'),
        human_render_camera_configs=dict(shader_pack='rt-fast'),
        viewer_camera_configs=dict(shader_pack='rt-fast'),
    )
    if args.control_mode:
        exec_kwargs['control_mode'] = args.control_mode

    exec_raw = gym.make(args.task, **exec_kwargs)
    if isinstance(exec_raw.action_space, gym.spaces.Dict):
        exec_raw = FlattenActionSpaceWrapper(exec_raw)
    exec_env = ManiSkillVectorEnv(exec_raw, 1,
                                   ignore_terminations=False,
                                   record_metrics=True)
    logger.info(f"  Executor: 1 env (standard {args.task})")

    # ==================================================================
    # Step 3: Load PPO agent
    # ==================================================================
    logger.info("Step 3: Loading PPO agent...")
    agent = Agent(exec_env).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    logger.info(f"  Loaded from {checkpoint}")

    # ==================================================================
    # Step 4: Reset and get goal position
    # ==================================================================
    logger.info("Step 4: Resetting environments...")
    obs, _ = exec_env.reset()

    if tsip is not None:
        tsip.reset()

    goal_pos = get_goal_position(exec_env, task_config)
    logger.debug(f"  Goal position: {goal_pos}")

    # ==================================================================
    # Step 5: Create solver (CAGE mode only)
    # ==================================================================
    solver = None
    cage = None

    if use_cage:
        from manidreams.solvers.samplers.policy_sampler import PolicySampler
        from manidreams.cages.dris_cage import DRISCage

        logger.info("Step 5: Creating DRISCage + PolicySampler...")
        cage = DRISCage(
            goal_pos=goal_pos,
            lambda_var=args.lambda_var,
            success_radius=0.05,
        )
        solver = PolicySampler(
            policy_model=PPOPolicyAdapter(agent),
            obs_encoder=None,
            obs_mapper=ObsTruncateMapper(),
            action_mapper=GripperLockMapper() if lock_gripper else None,
            num_samples=args.num_samples,
            horizon=args.action_chunk,
            deterministic=False,
            device=str(device),
        )
        # Initialize solver with flat obs
        solver.initialize(exec_env, obs)
        logger.info(f"  DRISCage: goal={goal_pos}, lambda_var={args.lambda_var}")
    else:
        logger.info("Step 5: Skipping solver (baseline mode)")

    # ==================================================================
    # Step 6: Render and wait
    # ==================================================================
    if tsip is not None:
        tsip.env.render()
        # Set TSIP viewer camera based on preset
        TSIP_CAMERA_PRESETS = {
            "closeup": {
                "eye": [1.6, 1.6, 0.5],
                "target": [0.0, 0.0, 0.0],
            },
            "topdown": {
                "eye": [0.0, 0.0, 3.0],
                "target": [0.0, 0.0, 0.0],
            },
        }
        tsip_base = tsip.env.unwrapped if hasattr(tsip.env, 'unwrapped') else tsip.env
        if hasattr(tsip_base, '_viewer') and tsip_base._viewer is not None:
            from mani_skill.utils import sapien_utils
            preset = TSIP_CAMERA_PRESETS[args.tsip_camera]
            cam_pose = sapien_utils.look_at(preset["eye"], preset["target"])
            tsip_base._viewer.set_camera_pose(cam_pose.sp)
    exec_env.render()

    if use_cage:
        logger.info(f"  Two visualization windows:")
        logger.info(f"    - TSIP: {args.num_samples} eval envs (DRIS)")
        logger.info(f"    - Executor: 1 env (standard)")
    else:
        logger.info(f"  One visualization window")

    time.sleep(2)

    # ==================================================================
    # Step 7: Main execution loop
    # ==================================================================
    max_episode_steps = getattr(exec_raw.unwrapped, '_max_episode_steps', None)
    control_freq = getattr(exec_raw.unwrapped, 'control_freq', 20)

    # Per-episode step budget: use env's max_episode_steps if available,
    # otherwise fall back to duration * control_freq
    if max_episode_steps is not None:
        steps_per_episode = max_episode_steps
    else:
        steps_per_episode = int(args.duration * control_freq)

    logger.info(f"Step 7: Running {args.num_episodes} episode(s), "
          f"up to {steps_per_episode} steps each...")
    logger.info("-" * 60)

    # Aggregate statistics across all episodes
    total_cost = 0.0
    total_valid = 0
    total_steps = 0
    total_plans = 0
    episode_successes_once = []
    episode_successes_end = []
    action_chunk = args.action_chunk

    for episode in range(args.num_episodes):
        if episode > 0:
            # Reset for new episode
            obs, _ = exec_env.reset()
            goal_pos = get_goal_position(exec_env, task_config)

            if use_cage:
                cage.goal_pos = np.asarray(goal_pos, dtype=np.float32)
                solver.update_after_step(obs)
                if tsip is not None:
                    exec_state = get_executor_state(exec_env, task_config)
                    tsip_backend.set_state(tsip.env, exec_state)

            if args.num_episodes > 1:
                logger.info(f"--- Episode {episode + 1}/{args.num_episodes} "
                      f"(goal={goal_pos}) ---")

        ep_cost = 0.0
        ep_valid = 0
        ep_steps = 0
        ep_plans = 0
        ep_success_once = False
        ep_success_end = False

        # Action chunk buffer
        action_buffer = []
        chunk_idx = 0

        for step in range(steps_per_episode):
            step_start = time.time()

            if use_cage:
                # --- CAGE mode ---
                need_replan = (action_chunk <= 1) or (chunk_idx >= len(action_buffer))

                if need_replan:
                    # 1. Read executor state (robot + object)
                    exec_state = get_executor_state(exec_env, task_config)

                    # 2. Sync executor -> TSIP via backend
                    tsip_backend.set_state(tsip.env, exec_state)

                    # 3. Build DRIS from executor state
                    current_dris = tsip_backend.state2dris(None)[0]

                    # 4. Solver: sample N -> evaluate -> select best
                    result, costs, validations = solver.solve(
                        action_space=None,
                        cage=cage,
                        tsip=tsip,
                        current_dris=current_dris,
                        verbose=args.verbose,
                        on_plan_step=lambda t, best_idx, c, v: tsip.env.render(),
                    )

                    best_idx = np.argmin(costs)
                    ep_cost += costs[best_idx]
                    if validations[best_idx]:
                        ep_valid += 1
                    ep_plans += 1

                    if action_chunk > 1:
                        # result is a trajectory (list of actions)
                        action_buffer = result
                        chunk_idx = 0
                        ctrl_t = action_buffer[chunk_idx]
                        chunk_idx += 1
                    else:
                        # result is a single action
                        ctrl_t = result
                else:
                    # Execute next action from planned chunk (no replanning)
                    ctrl_t = action_buffer[chunk_idx]
                    chunk_idx += 1

                # 5. Execute action on executor
                action_tensor = torch.from_numpy(
                    np.array(ctrl_t)
                ).unsqueeze(0).to(device)
                if lock_gripper:
                    action_tensor[..., -1] = -1.0
                obs, _rew, _term, _trunc, info = exec_env.step(action_tensor)

                # 6. Update solver latent state
                solver.update_after_step(obs)

            else:
                # --- Baseline mode ---
                with torch.no_grad():
                    action = agent.get_action(obs, deterministic=True)
                if lock_gripper:
                    action[..., -1] = -1.0
                obs, _rew, _term, _trunc, info = exec_env.step(action)

            ep_steps += 1

            # Render
            if tsip is not None:
                tsip.env.render()
            exec_env.render()

            # Progress logging
            if use_cage and (args.verbose or step % 10 == 0):
                elapsed = time.time() - step_start
                chunk_info = f", chunk={chunk_idx}/{len(action_buffer)}" if action_chunk > 1 else ""
                logger.info(f"  Step {step:3d}: cost={costs[best_idx]:.3f}, "
                      f"valid={validations[best_idx]}, time={elapsed*1000:.1f}ms{chunk_info}")

            # Real-time pacing
            elapsed = time.time() - step_start
            target_dt = 1.0 / control_freq
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            # Track success_once from episode metrics (cumulative OR)
            if "episode" in info and "success_once" in info["episode"]:
                sval = info["episode"]["success_once"]
                if hasattr(sval, 'item'):
                    sval = sval.item()
                if bool(sval):
                    ep_success_once = True

            # Check episode termination
            if _term.any() or _trunc.any():
                # On auto-reset, completed episode's data is in final_info
                if "final_info" in info:
                    fi = info["final_info"]
                    if "episode" in fi and "success_once" in fi["episode"]:
                        sval = fi["episode"]["success_once"]
                        if hasattr(sval, 'item'):
                            sval = sval.item()
                        if bool(sval):
                            ep_success_once = True
                    if "success" in fi:
                        sval = fi["success"]
                        if hasattr(sval, 'item'):
                            sval = sval.item()
                        ep_success_end = bool(sval)
                else:
                    # No auto-reset wrapper or single env
                    if "success" in info:
                        sval = info["success"]
                        if hasattr(sval, 'item'):
                            sval = sval.item()
                        ep_success_end = bool(sval)
                break

        # Episode summary
        total_cost += ep_cost
        total_valid += ep_valid
        total_steps += ep_steps
        total_plans += ep_plans
        episode_successes_once.append(ep_success_once)
        episode_successes_end.append(ep_success_end)

        status_once = "SUCCESS_ONCE" if ep_success_once else ""
        status_end = "SUCCESS_END" if ep_success_end else ""
        status = status_end or status_once or "FAIL"
        if use_cage:
            chunk_info = f", plans={ep_plans}" if action_chunk > 1 else ""
            logger.info(f"  Episode {episode + 1}: {status} ({ep_steps} steps)"
                  f", avg_cost={ep_cost / max(ep_plans, 1):.3f}, "
                  f"valid_ratio={ep_valid / max(ep_plans, 1) * 100:.0f}%{chunk_info}")
        else:
            logger.info(f"  Episode {episode + 1}: {status} ({ep_steps} steps)")

    # ==================================================================
    # Step 8: Summary
    # ==================================================================
    logger.info("-" * 60)
    n_once = sum(episode_successes_once)
    n_end = sum(episode_successes_end)
    n_ep = max(args.num_episodes, 1)
    logger.info(f"Execution Summary:")
    logger.info(f"  Mode: {'CAGE' if use_cage else 'Baseline'}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Success once:   {n_once}/{args.num_episodes} ({n_once / n_ep * 100:.0f}%)")
    logger.info(f"  Success at end: {n_end}/{args.num_episodes} ({n_end / n_ep * 100:.0f}%)")
    logger.info(f"  Total steps: {total_steps}")
    if use_cage:
        logger.info(f"  Total plans: {total_plans}")
        logger.info(f"  Average cost: {total_cost / max(total_plans, 1):.3f}")
        logger.info(f"  Valid action ratio: {total_valid / max(total_plans, 1) * 100:.1f}%")
        if action_chunk > 1:
            logger.info(f"  Action chunk: {action_chunk} (replan every {action_chunk} steps)")

    exec_env.close()
    if tsip is not None:
        tsip.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
