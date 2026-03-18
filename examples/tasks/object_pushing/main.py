"""
Object Pushing Task Example - Using Backend Classes Directly

This example shows how to use the PushBackend that consolidates
all ManiSkill functionality, eliminating the need for the env.py wrapper.
Demonstrates the complete cage-constrained pipeline with parallel execution.

Now you can run directly without setting PYTHONPATH:
python ManiDreams/examples/tasks/object_pushing/main.py

"""

import logging
import numpy as np
import gymnasium as gym

# Auto-setup PYTHONPATH - add ManiDreams root directory to path
import sys
import os

# Get the ManiDreams root directory (4 levels up from this file to reach ManiDreams/)
current_file = os.path.abspath(__file__)
manidreams_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

# Add to Python path if not already there
if manidreams_root not in sys.path:
    sys.path.insert(0, manidreams_root)
    logging.getLogger(__name__).info(f"Auto-added ManiDreams root to Python path: {manidreams_root}")
sys.path.insert(0, os.path.join(manidreams_root, "src"))

from manidreams.env import ManiDreamsEnv
from manidreams.cages.geometric import CircularCage
from manidreams.solvers.optimizers.geometric_optimizer import GeometricOptimizer
from manidreams.physics.simulation_tsip import SimulationBasedTSIP
from examples.physics.push_backend_sim import PushBackend

logger = logging.getLogger(__name__)


def main():
    """Object pushing demonstration using backend classes directly"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    logger.info("ManiDreams Object Pushing - Backend Architecture")
    logger.info("=" * 60)
    
    try:
        # 1. Initialize PushBackend (contains all ManiSkill functionality)
        logger.info("Initializing PushBackend...")
        backend = PushBackend()

        # Temporarily use original registered environment for 16-env visualization
        logger.info("Note: Using original registered environment for multi-env visualization")
        
        # 2. Configure environment with parallel envs matching action space
        # We need 16 environments to evaluate all 16 discrete push actions in parallel
        env_config = {
            'env_id': 'PushT-multi',
            'num_envs': 16,  # Match the number of discrete actions for parallel evaluation
            'obs_mode': 'state_dict',  # Use state observations like reference script (not rgb for GUI)
            'control_mode': 'pd_joint_delta_pos',
            'render_mode': 'human',  # Keep GUI for visualization
            'enable_gui': False,
            'pause_on_start': False,  # Pause to show GUI
            'sim_backend': 'gpu',
            # Object pushing specific parameters
            'num_objects': 64,
            'object_type': 'polygon_te',  # Object type corresponding to load_object function
            'masses': np.random.uniform(0.5, 1.0, 64),
            'frictions': np.random.uniform(0.2, 0.5, 64),
        }
        
        # 3. Initialize TSIP with backend and context
        logger.info("Initializing TSIP...")
        tsip = SimulationBasedTSIP(
            backend=backend, 
            env_config=env_config,
            context_info=env_config  # Pass config as context
        )
        
        # 4. Create circular cage constraint for object containment
        logger.info("Creating circular cage constraint...")
        # The cage defines the region where we want to keep the objects
        cage = CircularCage(
            state_space=gym.spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),
            center=[-0.2, 0.0],  # Orbit center offset so orbit starts at [0,0]
            radius=0.28,  # Radius defining the circular boundary
            time_varying=True,  # Enable time-varying cage for circular trajectory
            orbit_radius=0.2,  # Radius of the orbital motion
            orbit_speed=0.1     # Speed of the orbital motion (radians per timestep)
        )
        cage.initialize()
        
        # 5. Create solver
        logger.info("Creating geometric solver...")
        solver = GeometricOptimizer(config={
            'lambda1': 1.0,
            'lambda2': 0.5,
            'validation_threshold': 0.01
        })
        
        # 6. Create ManiDreams environment
        logger.info("Creating ManiDreams environment...")
        env = ManiDreamsEnv(
            tsip=tsip,
            action_space=gym.spaces.Discrete(16),  # 16 push directions
            cage=cage,
            solver=solver
        )
        
        # 7. Reset environment to initialize everything properly
        logger.info("Resetting environment...")
        # First reset the ManiDreams environment which will internally reset the TSIP
        initial_dris, info = env.reset(seed=42)
        logger.info(f"Environment initialized: {info.get('initialized', False)}")
        
        # 8. Import and create executor first
        logger.info("=" * 60)
        logger.info("=== INITIALIZING EXECUTOR ===")
        logger.info("=" * 60)
        logger.info("Creating independent simulation environment for execution...")

        from examples.executors import PushingTaskExecutor

        # Create executor with independent simulation
        executor = PushingTaskExecutor()

        # Initialize with configuration for independent environment
        executor_config = {
            'render_mode': 'human',  # Show GUI for execution
            'sim_backend': 'gpu',
            'control_mode': 'pd_joint_delta_pos',
            'seed': 123,  # Different seed for variety
            'shader': 'rt-fast'  # Use ray tracing shader for best visual quality
        }
        executor.initialize(executor_config)
        logger.info("Executor initialized successfully")

        # 9. Execute cage-constrained planning with execution loop (5 rounds)
        logger.info("=" * 60)
        logger.info("=== ITERATIVE CAGE-CONSTRAINED EXECUTION ===")
        logger.info("=" * 60)
        logger.info("Running 5 rounds of: Plan 20 actions -> Execute -> Get obs -> Reset DRIS -> Repeat")

        num_rounds = 10
        actions_per_round = 15

        all_trajectories = []
        all_action_histories = []
        all_cage_histories = []

        # Track cumulative timestep for cage trajectory continuity
        cumulative_timestep = 0

        for round_idx in range(num_rounds):
            logger.info("=" * 60)
            logger.info(f"=== ROUND {round_idx + 1}/{num_rounds} ===")
            logger.info("=" * 60)

            # Step 1: Plan actions using dream()
            # Pass cumulative_timestep so cage trajectory continues from where it left off
            logger.info(f"[Planning] Running dream() for {actions_per_round} steps...")
            logger.info(f"  Starting from global timestep: {cumulative_timestep}")

            trajectory, action_history, cage_history = env.dream(
                horizon=actions_per_round,
                start_timestep=cumulative_timestep,
                verbose=False
            )
            logger.info(f"Generated {len(action_history)} planned actions")

            # Update cumulative timestep for next round
            cumulative_timestep += actions_per_round

            # Store planning results
            all_trajectories.extend(trajectory)
            all_action_histories.extend(action_history)
            all_cage_histories.extend(cage_history)

            # Step 2: Execute planned actions in separate simulation
            logger.info(f"[Execution] Executing {len(action_history)} actions in independent simulation...")
            observations, feedbacks = executor.execute(
                action_history,
                get_feedback=True,
                cage_history=cage_history
            )
            logger.info(f"Executed {len(observations)} actions")

            # Step 3: Get current observation from executor
            logger.info(f"[Observation] Getting current state from executor...")
            current_obs = executor.get_obs()
            logger.debug(f"Object position: {current_obs['object_pos']}")
            logger.debug(f"Object quaternion: {current_obs['object_quat']}")
            logger.debug(f"Gripper position: {current_obs['gripper_pos']}")

            # Step 4: Directly set state to planning environment
            # backend.set_state() will automatically convert single object to 64 objects
            logger.info(f"[Set State] Updating all planning environments with executor observation...")
            logger.debug(f"  Converting single YCB object -> 64 polygon objects (all at same position)")

            backend.set_state(tsip.env, current_obs)

            logger.info(f"State updated in all 16 planning environments")
            logger.debug(f"  All 64 objects synchronized to position: {current_obs['object_pos']}")

            # Step 5: Update env.current_dris to reflect the new state
            logger.info(f"[Update DRIS] Synchronizing env.current_dris with new state...")

            # Get the updated DRIS from TSIP
            updated_dris_list = tsip.get_dris()
            if updated_dris_list and len(updated_dris_list) > 0:
                # Update env's current_dris with the first environment's DRIS
                env.current_dris = updated_dris_list[0]
                logger.info(f"env.current_dris updated")
            else:
                logger.error(f"Warning: Could not get DRIS from TSIP")

            # Continue to next round
            if round_idx < num_rounds - 1:
                logger.info(f"Continuing to Round {round_idx + 2}...")

        # Final summary
        logger.info("=" * 60)
        logger.info("=== EXECUTION COMPLETE ===")
        logger.info("=" * 60)
        logger.info(f"Total rounds: {num_rounds}")
        logger.info(f"Actions per round: {actions_per_round}")
        logger.info(f"Total actions planned: {len(all_action_histories)}")
        logger.info(f"Total actions executed: {len(all_action_histories)}")
        logger.info(f"Total trajectory length: {len(all_trajectories)}")

        # Clean up executor
        executor.close()
        logger.info("Executor closed")
    
    except ImportError as e:
        logger.error(f"Warning: ManiSkill not available - {e}")
        logger.error("Running simplified demonstration without ManiSkill...")
        
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()