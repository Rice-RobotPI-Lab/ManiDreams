"""
Object Picking Task - Picking Colored Box Targets

Demonstrates positioning near colored box target objects using the
ManiDreams framework with MPPI-based cage-constrained control.

Architecture (two-phase: plan then execute):
- Phase 1 (Planning): TSIP with N parallel envs + ManiDreamsEnv.dream()
  Plans a full action sequence using MPPI in simulation (16 cards, no GUI).
- Phase 2 (Execution): Independent executor with 1 env (1 card, with GUI)
  Replays the planned action sequence, then closes gripper and retracts.

Usage:
    cd ManiDreams/
    python examples/tasks/object_picking/main.py
"""

import logging
import numpy as np
import gymnasium as gym
import time

# Auto-setup PYTHONPATH
import sys
import os

current_file = os.path.abspath(__file__)
manidreams_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

if manidreams_root not in sys.path:
    sys.path.insert(0, manidreams_root)
sys.path.insert(0, os.path.join(manidreams_root, "src"))

from manidreams.env import ManiDreamsEnv
from manidreams.cages.geometric3d import Geometric3DCage
from manidreams.solvers.optimizers.mppi_optimizer import MPPIOptimizer
from manidreams.physics.simulation_tsip import SimulationBasedTSIP
from examples.physics.pick_backend import PickBackend
from examples.executors import PickingTaskExecutor

logger = logging.getLogger(__name__)


def generate_cage_trajectory(gripper_init_pos, target_avg_pos, wall_pos, total_timesteps=50,
                           fixed_orientation=None):
    """
    Generate 2-segment cage trajectory for picking task.

    Trajectory segments:
    - Segment 1 (33%): Gripper -> Target (approach from above)
    - Segment 2 (67%): Target -> Wall (push toward wall)

    Args:
        gripper_init_pos: [x, y, z] - Initial gripper position
        target_avg_pos: [x, y, z] - Average position of target objects
        wall_pos: [x, y, z] - Wall position
        total_timesteps: Total number of timesteps
        fixed_orientation: [roll, pitch, yaw] - Fixed orientation for all segments
                          Default: [pi/3, 0, 0]

    Returns:
        List of [x, y, z, roll, pitch, yaw] (6D pose)
    """
    trajectory = []

    if fixed_orientation is None:
        fixed_orientation = [np.pi / 3, 0.0, 0.0]

    roll, pitch, yaw = fixed_orientation

    gripper_x, gripper_y, gripper_z = gripper_init_pos
    target_x, target_y, target_z = target_avg_pos
    wall_x, wall_y, wall_z = wall_pos
    target_y = target_y - 0.1

    # Segment 1: Gripper -> Target (33%)
    segment1_steps = int(total_timesteps * 0.33)
    for i in range(segment1_steps):
        t = i / segment1_steps
        x = gripper_x * (1 - t) + target_x * t
        y = gripper_y * (1 - t) + target_y * t
        z = gripper_z * (1 - t) + target_z * t
        trajectory.append([x, y, z, roll, pitch, yaw])

    # Segment 2: Target -> Wall (67%)
    segment2_steps = total_timesteps - segment1_steps
    for i in range(segment2_steps):
        t = i / segment2_steps
        x = target_x + (wall_x - target_x) * t * 0.75
        y = target_y + (wall_y - target_y) * t * 0.75
        z = target_z
        trajectory.append([x, y, z, roll, pitch, yaw])

    return trajectory


def main():
    """Object picking demonstration with colored box targets"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    logger.info("ManiDreams Object Picking - Colored Box Targets")
    logger.info("=" * 60)

    try:
        # ================================================================
        # PHASE 1: PLANNING
        # Plan a full action sequence using MPPI + TSIP (16 cards, no GUI)
        # ================================================================
        logger.info("=" * 60)
        logger.info("PHASE 1: PLANNING")
        logger.info("=" * 60)

        # ====================================================================
        # Step 1: Configuration
        # ====================================================================
        NUM_PARALLEL = 10
        HORIZON = 100

        base_config = {
            'obs_mode': 'state_dict',
            'control_mode': 'pd_joint_delta_pos',
            'sim_backend': 'gpu',
            'robot_uids': 'floating_panda_gripper_fin',
            'robot_init_qpos_noise': 0.08,
            'num_target_objects': 16,
            'num_clutter_objects': 0,
            'pile_spacing': 0.012,
        }

        logger.info(f"  MPPI parallel envs: {NUM_PARALLEL}")
        logger.info(f"  Planning horizon: {HORIZON}")
        logger.info(f"  Target objects (planning): {base_config['num_target_objects']}")

        # ====================================================================
        # Step 2: Create TSIP for MPPI planning (no GUI)
        # ====================================================================
        logger.info("Step 2: Creating TSIP for planning...")
        tsip_backend = PickBackend()
        tsip_config = {
            **base_config,
            'num_envs': 1,
            'render_mode': 'human',
            'enable_gui': True,
            'pause_on_start': True,
        }
        tsip = SimulationBasedTSIP(
            backend=tsip_backend,
            env_config=tsip_config,
            context_info=tsip_config,
            num_rollout_envs=NUM_PARALLEL
        )
        logger.info(f"  TSIP: 1 main env + {NUM_PARALLEL} rollout envs (no GUI)")

        # ====================================================================
        # Step 3: Create action space
        # ====================================================================
        action_mean = np.array([0.0, -0.15, -0.225, 0.0, 0.0, 0.0], dtype=np.float32)
        action_range = np.array([0.02, 0.02, 0.000, 0.0, 0.03, 0.03], dtype=np.float32)
        # action_range = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        action_space = gym.spaces.Box(
            low=action_mean - action_range,
            high=action_mean + action_range,
            dtype=np.float32
        )

        # ====================================================================
        # Step 4: Create ManiDreamsEnv, reset, get initial positions
        # ====================================================================
        logger.info("Step 4: Creating ManiDreamsEnv and getting initial positions...")
        env = ManiDreamsEnv(tsip=tsip, action_space=action_space)
        initial_dris, info = env.reset(seed=42)

        # Get target positions from TSIP env (after reset)
        gripper_init_pos = [0.0, -0.5, 0.3]
        state = tsip_backend.get_state(tsip.env)

        if 'obj_positions' in state and len(state['obj_positions']) > 0:
            obj_positions = state['obj_positions']
            avg_pos = np.mean(obj_positions[:, :2], axis=0)
            target_avg_pos = [float(avg_pos[0]), float(avg_pos[1]), 0.05]
        else:
            target_avg_pos = [0.0, 0.0, 0.05]

        wall_pos = tsip_backend.wall_position
        logger.debug(f"  Gripper init: {gripper_init_pos}")
        logger.debug(f"  Target avg pos: {target_avg_pos}")
        logger.debug(f"  Wall pos: {wall_pos}")

        # ====================================================================
        # Step 5: Generate cage trajectory + create cage and solver
        # ====================================================================
        logger.info("Step 5: Creating cage trajectory, cage, and MPPI solver...")

        cage_trajectory = generate_cage_trajectory(
            gripper_init_pos=gripper_init_pos,
            target_avg_pos=target_avg_pos,
            wall_pos=wall_pos,
            total_timesteps=100
        )

        cage = Geometric3DCage(
            state_space=gym.spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            trajectory=cage_trajectory,
            radius=0.15,
            time_varying=True,
            cost_type='variance'
        )
        cage.initialize()
        logger.info(f"  Cage: {len(cage_trajectory)} waypoints, radius={cage.radius}")

        mppi_config = {
            'horizon': 3,
            'num_samples': NUM_PARALLEL,
            'num_elites': 3,
            'num_iterations': 2,
            'temperature': 1.0,
            'discount': 0.95,
            'init_std_scale': 0.3,
            'min_std_scale': 0.05,
            'max_std_scale': 0.6,
        }
        solver = MPPIOptimizer(config=mppi_config)
        logger.info(f"  MPPI: {mppi_config['num_samples']} samples, "
              f"horizon={mppi_config['horizon']}, "
              f"{mppi_config['num_iterations']} iterations")

        # ====================================================================
        # Step 6: Plan -- dream (full MPPI planning loop)
        # ====================================================================
        logger.info(f"Step 6: Running dream(horizon={HORIZON})...")
        logger.info("-" * 60)

        trajectory, action_history, cage_history = env.dream(
            horizon=HORIZON,
            cage=cage,
            solver=solver,
            verbose=True
        )

        logger.info("-" * 60)
        logger.info(f"  Planning complete! {len(action_history)} actions planned.")

        # ====================================================================
        # Step 7: Close planning environments
        # ====================================================================
        logger.info("Step 7: Closing planning environments...")
        env.close()
        logger.info("  Planning environments closed.")

        # ================================================================
        # PHASE 2: EXECUTION
        # Replay planned actions in executor (1 card, with GUI)
        # ================================================================
        logger.info("=" * 60)
        logger.info("PHASE 2: EXECUTION")
        logger.info("=" * 60)

        # ====================================================================
        # Step 8: Create executor (1 env, 1 card, GUI)
        # ====================================================================
        logger.info("Step 8: Creating executor (1 card, with GUI)...")
        executor = PickingTaskExecutor()
        executor_config = {
            **base_config,
            'num_envs': 1,
            'num_target_objects': 1,
            'num_clutter_objects': 0,
            'render_mode': 'human',
            'shader': 'rt-fast',
            'enable_gui': True,
            'pause_on_start': True,
        }
        executor.initialize(executor_config)
        executor.reset()
        executor.env.render()
        logger.info(f"  Executor: 1 env, 1 card, GUI enabled")
        logger.info(f"  Simulation is PAUSED. Press SPACE in the GUI window to start.")

        time.sleep(3)

        # ====================================================================
        # Step 9: Replay planned action sequence
        # ====================================================================
        logger.info(f"Step 9: Replaying {len(action_history)} planned actions...")
        logger.info("-" * 60)

        for i, action in enumerate(action_history):
            step_start = time.time()

            # Reconstruct cage state at this timestep (same as during planning)
            cage.apply_controller_updates(i)

            # Set cage and execute planned action
            executor.set_cage(cage)
            obs, feedback = executor.execute(action)
            executor.env.render()

            if i % 10 == 0:
                step_time = time.time() - step_start
                cage_info = cage_history[i] if i < len(cage_history) else {}
                logger.info(f"  Replay {i:3d}/{len(action_history)}: "
                      f"cage={[f'{c:.2f}' for c in cage.center]}, "
                      f"time={step_time*1000:.0f}ms")

        logger.info("-" * 60)
        logger.info(f"Replay completed!")

        # ====================================================================
        # Step 10: Close gripper and retract
        # ====================================================================
        executor.close_and_retract(move_direction=(0, -0.1, 0.1), num_steps=50)

        logger.info("Check the GUI to see the picking results.")

        # Keep GUI open
        logger.info("GUI is open. Press Enter to exit...")
        input()

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ====================================================================
        # Cleanup
        # ====================================================================
        if 'executor' in locals():
            executor.close()
        if 'env' in locals() and hasattr(env, 'tsip'):
            try:
                env.close()
            except Exception:
                pass
        logger.info("Object picking task completed!")


if __name__ == "__main__":
    logger.info("Starting ManiDreams Object Picking Demo")
    logger.info("=" * 60)
    main()
