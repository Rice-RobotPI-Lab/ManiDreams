"""
Object Pushing with Pixel-based TSIP

Complete 1:1 replication of Diamond's play_cage16.py AutoPlayGame.run() method.
Uses ManiDreams framework (TSIP, Cage, Solver) but replicates Diamond's exact execution flow.

Usage Examples:
    # Default: Realtime display with 500 steps (Diamond default)
    python main_pixel.py

    # Custom max steps
    python main_pixel.py --max-steps 1000

    # No visualization (fastest)
    python main_pixel.py --no-realtime

Interactive Controls:
    SPACE       - Pause/Resume execution
    RIGHT ARROW - Single step (when paused)
    ESC         - Exit program
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import gymnasium as gym
import sys
import os
import time
from datetime import datetime

# Add ManiDreams to path
current_file = os.path.abspath(__file__)
manidreams_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if manidreams_root not in sys.path:
    sys.path.insert(0, manidreams_root)
sys.path.insert(0, os.path.join(manidreams_root, "src"))

from manidreams.physics.learned_tsip import LearningBasedTSIP
from manidreams.cages.pixel_cage import CircularPixelCage
from manidreams.solvers.optimizers.pixel_optimizer import PixelOptimizer
from examples.physics.push_backend_learned import DiffusionBackend, DiffusionVisualizer
from examples.executors import PushingTaskExecutor

logger = logging.getLogger(__name__)


def main():
    """
    Main function replicating Diamond's AutoPlayGame.run() (play_cage16.py:260-330).

    Key differences from old main_pixel.py:
    1. Uses while True loop instead of for loop
    2. Initializes cage AFTER first PSS detection (inside loop)
    3. ALWAYS sets action_timer=4 and rest_counter=2 (even when action is None)
    4. Decrements timer AFTER execution
    5. Always executes world model (action 0 when resting/satisfied)
    6. Uses ep_length >= max_timesteps as termination condition
    7. Adds time.sleep(1.0/15) for FPS control
    """
    parser = argparse.ArgumentParser(description="ManiDreams Object Pushing - Diamond Replication")
    parser.add_argument('--model-dir', type=Path,
                       default=None,
                       help='Path to DIAMOND model directory (required)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum timesteps (Diamond default: 500)')
    parser.add_argument('--no-realtime', action='store_true', default=False,
                       help='Disable realtime pygame visualization')
    parser.add_argument('--record-video', action='store_true', default=False,
                       help='Enable video recording')
    parser.add_argument('--video-fps', type=int, default=300,
                       help='Frame rate (Diamond default: 15)')
    parser.add_argument('--scale-factor', type=int, default=4,
                       help='Display/video upscaling factor')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output (slower but more detailed)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    if args.model_dir is None:
        parser.error("--model-dir is required. Example: --model-dir /path/to/diamond/models/push16")

    logger.info("=" * 60)
    logger.info("ManiDreams Object Pushing - Diamond Replication")
    logger.info("Replicating play_cage16.py:260-330 (AutoPlayGame.run)")
    logger.info("=" * 60)

    # ========== Setup Visualizer ==========
    visualizer = None
    enable_realtime = not args.no_realtime
    enable_video = args.record_video

    if enable_realtime or enable_video:
        output_path = None
        if enable_video:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path('videos') / f'caging_{timestamp}.mp4'

        visualizer = DiffusionVisualizer(
            output_path=output_path,
            scale_factor=args.scale_factor,
            fps=args.video_fps,
            cage_params={
                'radius': 22,
                'orbit_radius': 42,
                'orbit_speed': 0.013
            },
            enable_realtime=enable_realtime,
            enable_video=enable_video
        )

        status = []
        if enable_realtime:
            status.append("realtime display")
        if enable_video:
            status.append(f"video recording: {output_path}")
        logger.info(f"[Setup] Visualization: {' + '.join(status)}")
    else:
        logger.info(f"[Setup] Running without visualization")

    # ========== Initialize Backend ==========
    logger.info("[Setup] Loading DIAMOND backend...")
    backend = DiffusionBackend(visualizer=visualizer)
    model_config = {
        'model_dir': args.model_dir,
        'model_name': 'push16'
    }

    # ========== Create TSIP ==========
    logger.info("[Setup] Initializing TSIP...")
    tsip = LearningBasedTSIP(
        backend=backend,
        model_config=model_config,
        context_info={'task': 'object_pushing', 'observation': 'pixel'}
    )

    # ========== Create Cage ==========
    logger.info("[Setup] Creating CircularPixelCage...")
    cage = CircularPixelCage(
        radius=22,
        orbit_radius=42,
        orbit_speed=0.013,
        time_varying=True
    )

    # ========== Create Solver ==========
    solver = PixelOptimizer(config={'distance_weight_power': 0.4})
    action_space = gym.spaces.Discrete(16)

    # ========== Reset Environment ==========
    logger.info("[Setup] Resetting environment...")
    current_dris = tsip.reset()
    logger.debug(f"Initial observation shape: {current_dris.observation.shape}")
    logger.debug(f"Initial observation stats: mean={current_dris.observation.mean():.6f}, std={current_dris.observation.std():.6f}")
    logger.debug(f"Initial observation range: [{current_dris.observation.min():.6f}, {current_dris.observation.max():.6f}]")

    # ========== Initialize Executor ==========
    logger.info("=" * 60)
    logger.info("=== INITIALIZING EXECUTOR ===")
    logger.info("=" * 60)
    logger.info("Creating independent simulation environment for execution...")

    executor = PushingTaskExecutor()

    executor_config = {
        'render_mode': 'human',  # Show GUI for execution
        'sim_backend': 'gpu',
        'control_mode': 'pd_joint_delta_pos',
        'seed': 123,
        'shader': 'rt-fast'
    }
    executor.initialize(executor_config)
    logger.info("Executor initialized successfully")
    logger.info("Note: Object initial position will be set to match first cage position after planning")

    logger.info("=" * 60)
    logger.info(f"PHASE 1: Planning with Diffusion Model (max steps: {args.max_steps})")
    logger.info(f"Collecting action sequence for later execution")
    logger.info("=" * 60)

    # ========== Main Loop Variables (play_cage16.py:260-265) ==========
    action_timer = 0
    rest_counter = 0
    current_action = None
    ep_length = 0
    max_timesteps = args.max_steps

    # Storage for action sequence and cage history
    planned_actions = []
    cage_history = []

    # ========== Main Loop: Planning Phase (play_cage16.py:267-330) ==========
    try:
        logger.info("[Planning] Running diffusion model to collect actions...")
        while True:

            # Get PSS center for cage initialization (only once per step)
            pss_center = None
            if cage.cage_center is None:  # Only check if cage not initialized yet
                pss_center = cage.get_pss_center(current_dris)
                if pss_center is not None:
                    cage.initialize(current_dris)

            # Update cage position (play_cage16.py:274)
            cage.update_cage_position()

            # Save cage state for executor
            # CRITICAL: Convert pixel coordinates to physical coordinates
            # Pixel space: [0, 64] x [0, 64] with origin at top-left
            # Physical space: [-0.2, 0.2] x [-0.2, 0.2] with origin at center
            if cage.cage_center is not None:
                # Convert pixel cage center to physical coordinates
                # Pixel to physical mapping (64x64 image, 0.4m workspace):
                # physical_x = (pixel_x - 32) * (0.4 / 64) = (pixel_x - 32) * 0.00625
                # physical_y = (32 - pixel_y) * (0.4 / 64) = (32 - pixel_y) * 0.00625
                pixel_to_meter = 0.4 / 64.0  # 0.00625 m/pixel
                physical_center_x = (cage.cage_center[1] - 32) * pixel_to_meter
                physical_center_y = - (32 - cage.cage_center[0]) * pixel_to_meter
                physical_center = np.array([physical_center_x, physical_center_y])

                # Convert pixel radius to physical radius
                physical_radius = cage.radius * pixel_to_meter

                cage_history.append({
                    'center': physical_center,
                    'radius': physical_radius,
                    'timestep': ep_length
                })

            # Update backend's cage info for visualization (only if visualizer active)
            if visualizer and cage.cage_center is not None:
                backend.set_cage_info(cage.cage_center, cage.radius)

            # ========== Action Decision Logic (play_cage16.py:276-288) ==========
            if action_timer <= 0:
                if rest_counter > 0:
                    # Resting period (play_cage16.py:277-280)
                    current_action = None
                    rest_counter -= 1
                    action_timer = 1
                    logger.debug(f"[ACTION] RESTING (rest_counter={rest_counter})")
                else:
                    # Compute new action using cage (play_cage16.py:282-288)
                    # This is the ONLY place we need to compute direction
                    direction_index = cage.compute_direction(current_dris)
                    logger.info(f"[STEP {ep_length}] {'='*50}")

                    if direction_index is None:
                        # PSS inside cage (play_cage16.py:283-285)
                        current_action = None
                        logger.debug(f"[ACTION] PSS satisfied, current_action=None")
                    else:
                        # PSS outside cage, push in direction (play_cage16.py:286-288)
                        current_action = direction_index
                        logger.info(f"[ACTION] New action: {current_action}")

                    # CRITICAL: ALWAYS set timer and rest_counter (play_cage16.py:289-290)
                    # This happens regardless of whether current_action is None or not
                    action_timer = 4  # cage_controller.action_duration
                    rest_counter = 2

            # ========== Store Action for Later Execution ==========
            # Diamond ALWAYS executes env.step(), even when current_action is None
            # CRITICAL: When current_action is None (rest/satisfied), Diamond uses
            # CSGOAction(keys=[], steering_value=0.0) which encodes to all-zero vector,
            # NOT action 0 which would be [1,0,0,0,...,0]!
            if current_action is None:
                # Use special value -1 to signal "empty action" (all zeros)
                execute_action = -1
            else:
                execute_action = current_action

            # Store action for later execution
            planned_actions.append(execute_action)

            # Execute world model step in diffusion model (for planning)
            next_dris = tsip.next(current_dris, execute_action, cage)

            # Update current state (play_cage16.py:307)
            current_dris = next_dris

            # ========== Timer Decrement AFTER Execution (play_cage16.py:308) ==========
            # CRITICAL: Decrement happens AFTER env.step(), not before
            action_timer -= 1

            # ========== Update Episode Length ==========
            ep_length += 1

            # ========== Termination Check (play_cage16.py:310-317) ==========
            if ep_length >= max_timesteps:
                logger.info(f"[TERMINATION] Reached max timesteps: {ep_length}/{max_timesteps}")
                break

            # ========== FPS Control (play_cage16.py:327) ==========
            # Note: Removed time.sleep() for maximum execution speed
            # Only needed if you want real-time visualization sync

    except KeyboardInterrupt:
        logger.info("[EXIT] Planning interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"[ERROR] Exception at step {ep_length}/{max_timesteps}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        logger.error("=" * 60)

    # ========== Planning Summary ==========
    logger.info("=" * 60)
    logger.info(f"Planning Phase Complete")
    logger.info("=" * 60)
    logger.info(f"Total steps planned: {ep_length}")
    logger.info(f"Total actions collected: {len(planned_actions)}")
    logger.info(f"Cage positions saved: {len(cage_history)}")
    if cage.cage_center is not None:
        logger.info(f"Final cage center (pixel): ({cage.cage_center[0]:.1f}, {cage.cage_center[1]:.1f})")
        if len(cage_history) > 0:
            final_physical = cage_history[-1]['center']
            logger.info(f"Final cage center (physical): ({final_physical[0]:.3f}, {final_physical[1]:.3f}) m")
    else:
        logger.info(f"Final cage center: Not initialized (no PSS detected)")

    # ========== PHASE 2: Execution in Real Simulation ==========
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Executing Actions in Real Simulation")
    logger.info("=" * 60)

    try:
        if len(planned_actions) > 0:
            logger.info(f"[Execution] Preparing to execute {len(planned_actions)} actions...")
            logger.debug(f"First 10 actions: {planned_actions[:10]}")
            logger.debug(f"Last 10 actions: {planned_actions[-10:]}")

            # Show cage position conversion examples
            if len(cage_history) >= 10:
                logger.debug(f"Cage position examples (pixel -> physical):")
                for i in [0, len(cage_history)//2, -1]:
                    cage_info = cage_history[i]
                    logger.debug(f"  Step {cage_info['timestep']}: center={cage_info['center']}, radius={cage_info['radius']:.3f}m")

            # CRITICAL: Set object initial position to match first cage position
            if len(cage_history) > 0:
                first_cage = cage_history[0]
                initial_position = first_cage['center']
                logger.info(f"[Setup] Setting object initial position to match first cage:")
                logger.debug(f"  Initial cage center: [{initial_position[0]:.3f}, {initial_position[1]:.3f}] m")

                # Set the state using executor's environment
                logger.info(f"  Setting executor object position...")
                executor.env.reset()  # Reset to default first

                # Get unwrapped environment
                actual_env = executor.env.unwrapped if hasattr(executor.env, 'unwrapped') else executor.env

                # Set object pose
                import torch
                from mani_skill.utils.structs import Pose
                obj_pose = Pose.create_from_pq(
                    p=torch.tensor([[initial_position[0], initial_position[1], 0.05]],
                                   device=actual_env.device, dtype=torch.float32),
                    q=torch.tensor([[1.0, 0.0, 0.0, 0.0]],
                                   device=actual_env.device, dtype=torch.float32)
                )
                actual_env.ycb_object.set_pose(obj_pose)

                # Update GPU state if needed
                if actual_env.device.type == "cuda":
                    actual_env.scene._gpu_apply_all()
                    actual_env.scene.px.gpu_update_articulation_kinematics()
                    actual_env.scene._gpu_fetch_all()

                logger.info(f"  Object position set successfully")

            # Sample actions: take every 6th step (matching Diamond's action timing)
            # Diamond has action_timer=4 + rest_counter=2 = 6 steps per action decision
            logger.info(f"[Execution] Sampling actions (every 6 steps, skip rest=-1)...")
            filtered_actions = []
            filtered_cage_history = []

            action_interval = 6  # Diamond's action decision interval

            for i in range(0, len(planned_actions), action_interval):
                action = planned_actions[i]

                # Skip rest actions (-1)
                if action == -1:
                    continue

                # Keep this action
                filtered_actions.append(action)
                filtered_cage_history.append(cage_history[i])

            logger.info(f"  Total planned steps: {len(planned_actions)}")
            logger.info(f"  Sampled actions (every {action_interval} steps): {len(filtered_actions)}")
            logger.info(f"  Expected actions (~steps/6): {len(planned_actions) // action_interval}")

            # Show filtered action sequence
            if len(filtered_actions) > 0:
                logger.debug(f"  Sampled action sequence:")
                logger.debug(f"    First 20: {filtered_actions[:20]}")
                if len(filtered_actions) > 20:
                    logger.debug(f"    Last 20:  {filtered_actions[-20:]}")

            logger.info(f"[Execution] Starting optimized action execution...")
            observations, feedbacks = executor.execute(
                filtered_actions,
                get_feedback=True,
                cage_history=filtered_cage_history
            )

            logger.info(f"Executed {len(observations)} actions")

            # Get final state
            logger.info(f"[Observation] Getting final state from executor...")
            final_obs = executor.get_obs()
            logger.debug(f"Final object position: {final_obs['object_pos']}")
            logger.debug(f"Final object quaternion: {final_obs['object_quat']}")
            logger.debug(f"Final gripper position: {final_obs['gripper_pos']}")
        else:
            logger.info("Warning: No actions were planned, skipping execution phase")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"[ERROR] Exception during execution")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        logger.error("=" * 60)

    # ========== Cleanup and Summary ==========
    logger.info("=" * 60)
    logger.info(f"Final Summary")
    logger.info("=" * 60)
    logger.info(f"Phase 1 (Planning): {ep_length} steps")
    logger.info(f"  Total actions planned: {len(planned_actions)}")
    if len(planned_actions) > 0 and 'filtered_actions' in locals():
        logger.info(f"Phase 2 (Execution): {len(filtered_actions)} actions (optimized)")
        logger.info(f"  Skipped: {len(planned_actions) - len(filtered_actions)} actions")
        logger.info(f"  Efficiency: {(1 - len(filtered_actions)/len(planned_actions))*100:.1f}% reduction")
    else:
        logger.info(f"Phase 2 (Execution): {len(planned_actions) if len(planned_actions) > 0 else 0} actions")

    # Finalize visualizer
    if visualizer:
        visualizer.finalize()

    # Close executor
    executor.close()
    logger.info("Executor closed")

    # Close TSIP
    tsip.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
