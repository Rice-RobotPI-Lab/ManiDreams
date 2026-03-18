"""
Object Catching Task - Unified Baseline and CAGE-enhanced Plate Catching

Supports two execution modes controlled by --num_samples:

1. Baseline mode (--num_samples 0, default):
   - Single executor environment
   - Direct policy output (deterministic mean)
   - No TSIP, no CAGE evaluation

2. CAGE mode (--num_samples N, N > 0):
   - Executor environment + N parallel TSIP evaluation environments
   - Samples N candidate actions from policy distribution
   - Evaluates them via TSIP simulation + PlateCage constraints
   - Selects the best action based on cost

Usage:
    cd ManiDreams/

    # Baseline mode (no TSIP, direct policy)
    python examples/tasks/object_catching/main.py
    python examples/tasks/object_catching/main.py --num_samples 0

    # CAGE mode (8 parallel evaluation environments)
    python examples/tasks/object_catching/main.py --num_samples 8
"""

import argparse
import logging
import numpy as np
import time
import torch
import sys
import os

# ============================================================================
# Auto-setup PYTHONPATH
# ============================================================================
current_file = os.path.abspath(__file__)
examples_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
manidreams_pkg_dir = os.path.dirname(examples_dir)  # ManiDreams/
manidreams_root = os.path.dirname(manidreams_pkg_dir)  # Repository root

if manidreams_pkg_dir not in sys.path:
    sys.path.insert(0, manidreams_pkg_dir)
sys.path.insert(0, os.path.join(manidreams_pkg_dir, "src"))

# ============================================================================
# Imports
# ============================================================================
import gymnasium as gym

from examples.samplers.catching import (
    ObsDRISMapper,
    ActionMapperStep,
    ActorCritic,
    PointCloudAE
)
from examples.physics.catch_backend import CatchBackend
from examples.executors import CatchingTaskExecutor

logger = logging.getLogger(__name__)

# Checkpoint directory
ckpts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'samplers', 'catching', 'ckpts')


def main():
    # ========================================================================
    # Parse arguments
    # ========================================================================
    parser = argparse.ArgumentParser(description="Object Catching Task")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of CAGE action samples. 0=baseline (no TSIP), N>0=CAGE mode")
    parser.add_argument("--num_objs", type=int, default=1,
                        help="Number of balls in executor environment")
    parser.add_argument("--num_objs_tsip", type=int, default=16,
                        help="Number of balls per TSIP evaluation environment (DRIS)")
    parser.add_argument("--render_device", type=str, default="gpu",
                        choices=["cpu", "gpu"],
                        help="Render device")
    parser.add_argument("--shader", type=str, default="rt-fast",
                        choices=["default", "rt-fast", "rt"],
                        help="Shader type")
    parser.add_argument("--sim_device", type=str, default="cpu",
                        choices=["cpu", "gpu", "physx_cuda"],
                        help="Simulation device for executor")
    parser.add_argument("--action_chunk", type=int, default=8,
                        help="Action chunk length (horizon). 1=replan every step, N>1=optimize N-step trajectory")
    parser.add_argument("--discount", type=float, default=0.95,
                        help="Discount factor for multi-step cost accumulation (action_chunk > 1)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration in seconds")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug information")
    parser.add_argument("--render_planning", action="store_true", default=True,
                        help="Render TSIP env at each planning step within action chunk")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    np.set_printoptions(suppress=True, precision=3)

    # Determine mode
    use_cage = args.num_samples > 0

    logger.info("=" * 60)
    if use_cage:
        logger.info("Object Catching Task - CAGE Mode")
    else:
        logger.info("Object Catching Task - Baseline Mode")
    logger.info("=" * 60)
    logger.info(f"  num_samples: {args.num_samples}")
    logger.info(f"  num_objs (executor): {args.num_objs}")
    if use_cage:
        logger.info(f"  num_objs_tsip: {args.num_objs_tsip}")
        if args.action_chunk > 1:
            logger.info(f"  action_chunk: {args.action_chunk} (horizon={args.action_chunk}, discount={args.discount})")

    # ========================================================================
    # Step 1: Create TSIP (CAGE mode only)
    # ========================================================================
    tsip = None
    tsip_backend = None

    if use_cage:
        from manidreams.physics.simulation_tsip import SimulationBasedTSIP
        from manidreams.cages.plate_cage import PlateCage

        logger.info(f"Step 1: Creating TSIP with {args.num_samples} evaluation environments...")
        tsip_backend = CatchBackend()
        eval_env_config = {
            'num_envs': args.num_samples,
            'num_objs': args.num_objs_tsip,
            'sim_backend': 'gpu',
            'render_mode': 'human',
            'render_backend': args.render_device,
            'shader': 'default',
            'pause_on_start': True
        }
        tsip = SimulationBasedTSIP(
            backend=tsip_backend,
            env_config=eval_env_config,
            context_info=eval_env_config
        )
        logger.info(f"  Created TSIP: {args.num_samples} envs x {args.num_objs_tsip} balls each")
    else:
        logger.info("Step 1: Skipping TSIP (baseline mode)")

    # ========================================================================
    # Step 2: Create Executor
    # ========================================================================
    logger.info("Step 2: Creating executor...")
    executor = CatchingTaskExecutor()
    exec_env_config = {
        'num_envs': 1,
        'num_objs': args.num_objs,
        'sim_backend': 'gpu' if use_cage else args.sim_device,
        'render_mode': 'human',
        'render_backend': args.render_device,
        'shader': args.shader,
        'pause_on_start': True
    }
    executor.initialize(exec_env_config)
    exec_env = executor.env  # Direct env reference for policy init, render, etc.
    logger.info(f"  Created Executor: 1 env x {args.num_objs} ball(s)")

    # ========================================================================
    # Step 3: Load Policy and Encoder
    # ========================================================================
    logger.info("Step 3: Loading policy and encoder...")
    obs, _ = exec_env.reset()

    actual_env = exec_env.unwrapped
    device = actual_env.device if hasattr(actual_env, 'device') else 'cpu'

    # Load policy model
    policy_path = os.path.join(ckpts_dir, 'policy_ckpt.pt')
    model = ActorCritic(exec_env, obs_dim=64, action_dim=5, cond="film").to(device)
    model.load_state_dict(torch.load(policy_path, map_location=device))
    model.eval()
    logger.info(f"  Loaded policy from {policy_path}")

    # Load DRIS encoder
    dris_path = os.path.join(ckpts_dir, 'dris_encoder.pt')
    dris_model = PointCloudAE(obs_dim=6, latent_dim=64, point_size=200).to(device)
    dris_model.load_state_dict(torch.load(dris_path, map_location=device))
    dris_model.eval()
    logger.info(f"  Loaded encoder from {dris_path}")

    # ========================================================================
    # Step 4: Create Mappers
    # ========================================================================
    logger.info("Step 4: Creating observation and action mappers...")
    obs_mapper = ObsDRISMapper(
        exec_env, device=str(device),
        use_full_obs=True,
        use_guassian_distr=False
    )
    action_mapper = ActionMapperStep(
        exec_env, device=str(device),
        vel_match=False,
        use_full_obs=True
    )

    # ========================================================================
    # Step 5: Create Solver (mode-dependent)
    # ========================================================================
    from manidreams.solvers.samplers.policy_sampler import PolicySampler

    if use_cage:
        logger.info("Step 5: Creating PlateCage + PolicySampler (CAGE mode)...")
        cage = PlateCage(
            plate_radius=0.12,
            dist_threshold=0.1,
            vel_threshold=0.2,
            dist_weight=0.7,
            vel_weight=0.3
        )
        solver = PolicySampler(
            policy_model=model,
            obs_encoder=dris_model,
            obs_mapper=obs_mapper,
            action_mapper=action_mapper,
            num_samples=args.num_samples,
            horizon=args.action_chunk,
            discount=args.discount,
            deterministic=False,
            device=str(device)
        )
    else:
        logger.info("Step 5: Creating PolicySampler (baseline)...")
        cage = None
        solver = PolicySampler(
            policy_model=model,
            obs_encoder=dris_model,
            obs_mapper=obs_mapper,
            action_mapper=action_mapper,
            device=str(device)
        )

    solver.initialize(exec_env, obs)

    # ========================================================================
    # Step 6: Reset and render
    # ========================================================================
    logger.info("Step 6: Resetting environments...")
    obs, _ = executor.reset()
    solver.initialize(exec_env, obs)

    if tsip is not None:
        tsip.reset()

    # Render both environments (TSIP first, then executor -- matching original order)
    if use_cage:
        tsip.env.render()
    exec_env.render()

    if use_cage:
        logger.info(f"  Two visualization windows:")
        logger.info(f"    - TSIP: {args.num_samples} parallel envs, each with {args.num_objs_tsip} balls")
        logger.info(f"    - Executor: 1 env with {args.num_objs} ball(s)")
        logger.info("  Simulation is PAUSED. Press SPACE in the GUI window to start.")
    else:
        logger.info(f"  One visualization window: 1 env with {args.num_objs} ball(s)")

    time.sleep(5)  # Wait for GUI initialization (matching original)

    # ========================================================================
    # Step 7: Main execution loop
    # ========================================================================
    control_freq = actual_env.control_freq if hasattr(actual_env, 'control_freq') else 20
    duration = args.duration
    time_step = 1 / control_freq
    num_steps = int(duration / time_step)

    action_chunk = args.action_chunk
    logger.info(f"Step 7: Executing {num_steps} timesteps...")
    logger.info(f"Duration: {duration}s, Time step: {time_step}s, Control freq: {control_freq}Hz")
    if action_chunk > 1:
        logger.info(f"Action chunk: {action_chunk} steps (replan every {action_chunk * time_step:.3f}s)")
    logger.info("-" * 60)

    total_cost = 0.0
    valid_count = 0
    plan_count = 0

    # Action chunk buffer: stores planned trajectory for multi-step execution
    action_buffer = []
    chunk_idx = 0

    for step in range(num_steps):
        step_start = time.time()

        if use_cage:
            need_replan = (action_chunk <= 1) or (chunk_idx >= len(action_buffer))

            if need_replan:
                # Plan: sample + evaluate + select (single action or full chunk)
                current_state = executor.get_state()
                current_dris = executor.state_to_dris(current_state, env_config={'num_envs': 1})[0]
                tsip_backend.set_state(tsip.env, current_state)

                # Planning step callback for TSIP visualization
                plan_step_cb = None
                if args.render_planning:
                    def plan_step_cb(step, best_idx, costs, validations):
                        tsip.env.render()

                result, costs, validations = solver.solve(
                    action_space=None,
                    cage=cage,
                    tsip=tsip,
                    current_dris=current_dris,
                    verbose=args.verbose,
                    on_plan_step=plan_step_cb
                )

                # Track statistics
                best_idx = np.argmin(costs)
                total_cost += costs[best_idx]
                if validations[best_idx]:
                    valid_count += 1
                plan_count += 1

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
                # Execute next action from planned chunk
                ctrl_t = action_buffer[chunk_idx]
                chunk_idx += 1
        else:
            # Baseline mode: direct policy output
            ctrl_t, _, _ = solver.solve(None, None, None, None)

        # Execute action
        obs, feedback = executor.execute(ctrl_t)

        # Update solver state
        solver.update_after_step(obs)

        # Render
        if tsip is not None:
            tsip.env.render()
        exec_env.render()

        # Log progress
        if use_cage and (args.verbose or step % 10 == 0):
            step_time = time.time() - step_start
            chunk_info = f", chunk={chunk_idx}/{len(action_buffer)}" if action_chunk > 1 else ""
            logger.info(f"Step {step:3d}: cost={costs[best_idx]:.3f}, "
                  f"valid={validations[best_idx]}, "
                  f"time={step_time*1000:.1f}ms{chunk_info}")

        # Delay for real-time execution
        elapsed = time.time() - step_start
        if elapsed < time_step:
            time.sleep(time_step - elapsed)

    # ========================================================================
    # Step 8: Summary and cleanup
    # ========================================================================
    logger.info("-" * 60)
    logger.info("Execution Summary:")
    logger.info(f"  Mode: {'CAGE' if use_cage else 'Baseline'}")
    logger.info(f"  Total steps: {num_steps}")
    if use_cage:
        logger.info(f"  Action chunk: {action_chunk}")
        logger.info(f"  Planning calls: {plan_count}")
        logger.info(f"  Average cost: {total_cost / plan_count:.3f}" if plan_count > 0 else "  Average cost: N/A")
        logger.info(f"  Valid plan ratio: {valid_count / plan_count * 100:.1f}%" if plan_count > 0 else "  Valid plan ratio: N/A")

    executor.close()
    if tsip is not None:
        tsip.close()
    logger.info("Object catching task completed!")


if __name__ == "__main__":
    main()
