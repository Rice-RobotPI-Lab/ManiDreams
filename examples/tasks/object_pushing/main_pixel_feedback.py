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
from manidreams.base.dris import DRIS
import torch

logger = logging.getLogger(__name__)


def generate_synthetic_observation(
    object_position_physical: np.ndarray,
    object_radius_physical: float = 0.03,
    image_size: int = 64,
    pixel_to_meter: float = 0.4 / 64.0
) -> np.ndarray:
    """
    Generate synthetic 64x64x3 RGB image from physical object position.

    Creates an image matching the DIAMOND diffusion model training format:
    - Gray background: RGB (230, 230, 230)
    - Red circular object (PSS): RGB (203, 70, 75)
    - Coordinate system: Physical origin (0,0) maps to image center (32, 32)

    NOTE: DIAMOND uses 64x64 low-res observations (obs_buffer), with optional
    128x128 upsampling (obs_full_res_buffer). This function generates the
    low-res 64x64 version.

    Args:
        object_position_physical: [x, y] in meters (physical coordinates)
                                 Physical space: [-0.2, 0.2] x [-0.2, 0.2] m
        object_radius_physical: Object radius in meters (default 3cm)
        image_size: Output image size (default 64x64 for DIAMOND low-res)
        pixel_to_meter: Conversion factor (default 0.4/64 = 0.00625 m/pixel)
                       Workspace: 0.4m x 0.4m -> 64 x 64 pixels

    Returns:
        Synthetic image (64, 64, 3) in [0, 1] range (HWC numpy array)

    Coordinate Mapping:
        Physical (0, 0) -> Pixel (32, 32) [image center]
        Physical (+x, +y) -> Pixel (right, down)

        Formulas:
        pixel_row = 32 - physical_y / pixel_to_meter
        pixel_col = 32 + physical_x / pixel_to_meter
    """
    meter_to_pixel = 1.0 / pixel_to_meter
    center = image_size // 2  # 32 for 64x64

    # Convert physical coordinates to pixel coordinates
    # Physical space: x-right, y-up
    # Pixel space: row-down, col-right
    # Physical (0,0) -> Pixel (64, 64)

    # Physical [x, y] -> Pixel [row, col]
    # CRITICAL: DIAMOND diffusion model uses Y-down coordinate system!
    # Executor reports Y-up, but diffusion expects Y-down representation
    # Therefore: Use DIRECT mapping (no inversion)

    # X coordinate: physical_x -> pixel_col (direct)
    pixel_col =  object_position_physical[1] / pixel_to_meter + center

    # Y coordinate: physical_y -> pixel_row (DIRECT, no inversion for diffusion)
    pixel_row =  object_position_physical[0] / pixel_to_meter + center

    pixel_radius = object_radius_physical / pixel_to_meter

    # Debug: Log coordinate mapping
    logger.debug(f"    [Coordinate Mapping]")
    logger.debug(f"      Physical: ({object_position_physical[0]:.3f}, {object_position_physical[1]:.3f}) m")
    logger.debug(f"      Pixel:    (row={pixel_row:.1f}, col={pixel_col:.1f})")
    logger.debug(f"      Radius:   {pixel_radius:.1f} pixels ({object_radius_physical*1000:.1f} mm)")

    # Create gray background (230, 230, 230)
    background_color = np.array([230, 230, 230], dtype=np.float32) / 255.0
    image = np.ones((image_size, image_size, 3), dtype=np.float32) * background_color

    # Draw red circular object (203, 70, 75) - represents PSS
    pss_color = np.array([203, 70, 75], dtype=np.float32) / 255.0

    # Create coordinate grids
    row_grid, col_grid = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((col_grid - pixel_col)**2 + (row_grid - pixel_row)**2)
    mask = distance <= pixel_radius

    # Check if object is visible
    num_pixels = np.sum(mask)
    if num_pixels == 0:
        logger.debug(f"    WARNING: Object not visible in image (outside bounds or too small)")
    else:
        logger.debug(f"    Object visible: {num_pixels} pixels")

    image[mask] = pss_color

    return image  # Shape: (128, 128, 3), range: [0, 1]


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
                       default=Path('/home/vector/Research/OpenCage/diamond/models/push16'),
                       help='Path to DIAMOND model directory')
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
    logger.info(f"ITERATIVE PLANNING AND EXECUTION")
    logger.info(f"Running feedback loop: Plan -> Execute -> Observe -> Reset DRIS")
    logger.info("=" * 60)

    # ========== Configuration ==========
    num_rounds = 10
    actions_per_round = 120  # 60 steps ~ 10 actions after sampling
    cumulative_timestep = 0

    # Storage for global history
    all_planned_actions = []
    all_executed_actions = []

    # ========== Multi-Round Iterative Loop ==========
    try:
        for round_idx in range(num_rounds):
            logger.info("=" * 60)
            logger.info(f"=== ROUND {round_idx + 1}/{num_rounds} ===")
            logger.info("=" * 60)

            # ========== PHASE A: PLANNING ==========
            logger.info(f"[Planning] Running diffusion for {actions_per_round} steps...")
            logger.info(f"  Starting from timestep: {cumulative_timestep}")

            planned_actions = []
            cage_history = []

            # Planning loop variables
            action_timer = 0
            rest_counter = 0
            current_action = None

            for step_idx in range(actions_per_round):
                # Get PSS center for cage initialization (only once per step)
                if cage.cage_center is None:
                    pss_center = cage.get_pss_center(current_dris)
                    if pss_center is not None:
                        cage.initialize(current_dris)

                # Update cage position
                cage.update_cage_position()

                # Save cage state for executor
                # IMPORTANT: Convert pixel coordinates to physical coordinates
                # cage.cage_center: [pixel_row, pixel_col] (image format)
                # physical_center: [physical_x, physical_y] (executor format)

                if cage.cage_center is not None:
                    pixel_to_meter = 0.8 / 128
                    center = 64
                    logger.debug(f"cage.cage_center: {cage.cage_center}")
                    # Pixel [row, col] -> Physical [x, y]
                    # col -> physical_x (direct mapping)
                    physical_center_x = (cage.cage_center[1] - center) * pixel_to_meter

                    # row -> physical_y (inverted: row up means y down)
                    physical_center_y = (cage.cage_center[0] - center) * pixel_to_meter
                    logger.debug(f"physical_center_x: {physical_center_x}, physical_center_y: {physical_center_y}")

                    physical_center = np.array([physical_center_x, physical_center_y])
                    physical_radius = cage.radius * pixel_to_meter

                    cage_history.append({
                        'center': physical_center,
                        'radius': physical_radius,
                        'timestep': cage.time  # FIXED: Use cage.time instead of environment timestep
                    })

                # Visualize cage
                if visualizer and cage.cage_center is not None:
                    backend.set_cage_info(cage.cage_center, cage.radius)

                # Action decision logic
                if action_timer <= 0:
                    if rest_counter > 0:
                        current_action = None
                        rest_counter -= 1
                        action_timer = 1
                    else:
                        direction_index = cage.compute_direction(current_dris)

                        # Log every 6 steps
                        if step_idx % 6 == 0:
                            logger.info(f"  Step {cumulative_timestep + step_idx}: action={direction_index}")

                        if direction_index is None:
                            current_action = None
                        else:
                            current_action = direction_index

                        action_timer = 4
                        rest_counter = 2

                # Store action
                execute_action = -1 if current_action is None else current_action
                planned_actions.append(execute_action)

                # Execute in diffusion model
                next_dris = tsip.next(current_dris, execute_action, cage)
                current_dris = next_dris

                action_timer -= 1

            logger.info(f"Planned {len(planned_actions)} steps")
            all_planned_actions.extend(planned_actions)

            # ========== PHASE B: EXECUTION ==========
            # Sample actions (every 6 steps, skip -1)
            logger.info(f"[Execution] Sampling actions...")
            filtered_actions = []
            filtered_cage_history = []

            for i in range(0, len(planned_actions), 6):
                if planned_actions[i] != -1:
                    filtered_actions.append(planned_actions[i])
                    filtered_cage_history.append(cage_history[i])

            logger.info(f"  Sampled {len(filtered_actions)} actions (every 6 steps, skipped -1)")

            # First round: set initial position
            if round_idx == 0 and len(filtered_cage_history) > 0:
                initial_position = filtered_cage_history[0]['center']
                logger.info(f"[Setup] Setting initial object position: [{initial_position[0]:.3f}, {initial_position[1]:.3f}] m")

                executor.env.reset()
                actual_env = executor.env.unwrapped if hasattr(executor.env, 'unwrapped') else executor.env

                obj_pose = torch.tensor([[initial_position[0], initial_position[1], 0.05, 1.0, 0.0, 0.0, 0.0]],
                                       device=actual_env.device, dtype=torch.float32)
                from mani_skill.utils.structs import Pose
                obj_pose_struct = Pose.create_from_pq(
                    p=obj_pose[:, :3],
                    q=obj_pose[:, 3:]
                )
                actual_env.ycb_object.set_pose(obj_pose_struct)

                if actual_env.device.type == "cuda":
                    actual_env.scene._gpu_apply_all()
                    actual_env.scene.px.gpu_update_articulation_kinematics()
                    actual_env.scene._gpu_fetch_all()

                logger.info(f"  Initial position set")

            # Execute actions
            if len(filtered_actions) > 0:
                observations, feedbacks = executor.execute(
                    filtered_actions,
                    get_feedback=True,
                    cage_history=filtered_cage_history
                )
                logger.info(f"Executed {len(filtered_actions)} actions")
                all_executed_actions.extend(filtered_actions)

            # ========== PHASE C: OBSERVATION ==========
            logger.info(f"[Observation] Getting current object position...")
            current_obs = executor.get_obs()
            object_pos = current_obs['object_pos'][:2]  # [x, y]
            logger.info(f"  Object position: [{object_pos[0]:.3f}, {object_pos[1]:.3f}] m")

            # ========== PHASE D: RESET DRIS (Using Unified Interface) ==========
            logger.info(f"[Reset DRIS] Generating synthetic observation and updating TSIP...")

            # 1. Generate synthetic image from physical feedback
            synthetic_image = generate_synthetic_observation(
                object_position_physical=object_pos,
                object_radius_physical=0.1,  # 2x radius for visibility (3cm -> 6cm)
                image_size=64,  # DIAMOND low-res (obs_buffer)
                pixel_to_meter=0.8 / 64.0
            )

            logger.debug(f"  Generated synthetic image: shape={synthetic_image.shape}, "
                  f"range=[{synthetic_image.min():.2f}, {synthetic_image.max():.2f}]")
            logger.debug(f"  Format: HWC numpy array, dtype={synthetic_image.dtype}")

            # 2. Debug visualization: Display synthetic image for user verification
            logger.debug(f"[DEBUG] Displaying synthetic observation for verification...")
            logger.debug(f"  Close the window to continue execution")

            import matplotlib.pyplot as plt

            # Create figure with comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: Synthetic image (what we're sending to TSIP)
            axes[0].imshow(synthetic_image)
            axes[0].set_title(f'Synthetic Observation (Round {round_idx + 1})\n'
                            f'HWC [0,1] format: {synthetic_image.shape}', fontsize=10)
            axes[0].axis('off')

            # Add object position text
            axes[0].text(0.5, -0.1,
                        f'Object Position: [{object_pos[0]:.3f}, {object_pos[1]:.3f}] m\n'
                        f'Timestep: {cumulative_timestep + actions_per_round}',
                        transform=axes[0].transAxes,
                        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Right: Current DRIS observation (before update)
            if current_dris and current_dris.observation is not None:
                axes[1].imshow(current_dris.observation)
                axes[1].set_title(f'Current DRIS Observation (Before Update)\n'
                                f'Shape: {current_dris.observation.shape}', fontsize=10)
            else:
                axes[1].text(0.5, 0.5, 'No current DRIS',
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Current DRIS Observation (Before Update)', fontsize=10)
            axes[1].axis('off')

            plt.suptitle(f'Feedback Loop Observation Update - Round {round_idx + 1}/{num_rounds}',
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show(block=True)  # Block until user closes window
            plt.close(fig)

            logger.debug(f"  Visualization closed, continuing execution...")

            # 3. Verify format compatibility with DiffusionBackend
            logger.debug(f"[Format Verification]")
            logger.debug(f"  Synthetic image format:")
            logger.debug(f"    - Shape: {synthetic_image.shape} (expected: (64, 64, 3) HWC)")
            logger.debug(f"    - Dtype: {synthetic_image.dtype} (expected: float32 or float64)")
            logger.debug(f"    - Range: [{synthetic_image.min():.3f}, {synthetic_image.max():.3f}] (expected: [0, 1])")

            # Validate format
            assert synthetic_image.shape == (64, 64, 3), f"Invalid shape: {synthetic_image.shape}, expected (64, 64, 3)"
            assert synthetic_image.min() >= 0 and synthetic_image.max() <= 1, f"Invalid range: [{synthetic_image.min()}, {synthetic_image.max()}]"
            logger.debug(f"  Format validation passed (64x64 low-res for obs_buffer)")

            # 4. Create new DRIS with feedback observation
            new_dris = DRIS(
                observation=synthetic_image,  # HWC [0,1]
                context={},
                metadata={'timestep': cumulative_timestep + actions_per_round}
            )

            # 5. Update TSIP state (automatically syncs to backend)
            logger.info(f"[Updating TSIP State]")
            tsip.set_dris(new_dris)
            current_dris = new_dris

            logger.debug(f"  tsip.set_dris() called")
            logger.debug(f"  Backend.set_dris() converts HWC [0,1] -> CHW [-1,1] tensor")
            logger.debug(f"  Backend state synchronized (timestep: {cumulative_timestep + actions_per_round})")

            # 6. Find closest cage point in trajectory and reset cage to that timestep
            logger.info(f"[Cage Trajectory Synchronization]")
            logger.info(f"  Finding closest cage point to object position...")

            if len(cage_history) > 0:
                # Current object position (physical coordinates)
                current_object_pos = object_pos  # [x, y] in meters

                # Find closest cage point in trajectory
                min_distance = float('inf')
                closest_idx = 0

                for idx, cage_point in enumerate(cage_history):
                    cage_center_physical = cage_point['center']  # [x, y] in meters

                    # Calculate Euclidean distance
                    distance = np.linalg.norm(current_object_pos - cage_center_physical)

                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx

                # Get the closest cage point
                closest_cage = cage_history[closest_idx]
                closest_timestep = closest_cage['timestep']
                closest_center_physical = closest_cage['center']

                logger.debug(f"  Object position: [{current_object_pos[0]:.3f}, {current_object_pos[1]:.3f}] m")
                logger.debug(f"  Closest cage: [{closest_center_physical[0]:.3f}, {closest_center_physical[1]:.3f}] m")
                logger.debug(f"  Distance: {min_distance:.3f} m")
                logger.debug(f"  Trajectory index: {closest_idx}/{len(cage_history)}")
                logger.debug(f"  Cage time: {closest_timestep:.4f}")

                # Reset cage to the closest trajectory point
                # Since cage is time-varying, just set time - cage will compute position automatically
                cage.time = closest_timestep

                # Compute cage position from current time (without incrementing time)
                cage.compute_position_from_time()

                logger.info(f"  Cage time reset to {closest_timestep:.4f}")
                logger.debug(f"    Cage position computed from time (time-varying cage)")
                if cage.cage_center is not None:
                    logger.debug(f"    Cage position: [row={cage.cage_center[0]:.1f}, col={cage.cage_center[1]:.1f}]")
            else:
                logger.info(f"  No cage history available, skipping synchronization")

            # Update cumulative timestep
            cumulative_timestep += actions_per_round

            # Continue to next round
            if round_idx < num_rounds - 1:
                logger.info(f"Continuing to Round {round_idx + 2}...")

    except KeyboardInterrupt:
        logger.info("[EXIT] Execution interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"[ERROR] Exception during execution")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        logger.error("=" * 60)

    # ========== Final Summary ==========
    logger.info("=" * 60)
    logger.info(f"ITERATIVE EXECUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rounds: {num_rounds}")
    logger.info(f"Actions per round: {actions_per_round}")
    logger.info(f"Total steps planned: {len(all_planned_actions)}")
    logger.info(f"Total actions executed: {len(all_executed_actions)}")
    if 'object_pos' in locals():
        logger.info(f"Final object position: [{object_pos[0]:.3f}, {object_pos[1]:.3f}] m")
    if cage.cage_center is not None:
        logger.debug(f"Final cage center (pixel): ({cage.cage_center[0]:.1f}, {cage.cage_center[1]:.1f})")

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
