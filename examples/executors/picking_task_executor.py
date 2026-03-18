"""Picking task executor — wraps CustomPickingEnv execution environment."""

import logging
from typing import Dict, Any, Tuple, Union, List
from .maniskill_executor_base import ManiSkillExecutorBase
from examples.physics.pick_backend import PickBackend

logger = logging.getLogger(__name__)


class PickingTaskExecutor(ManiSkillExecutorBase):
    """Executor for object picking task.

    Creates independent CustomPickingEnv environment for execution.
    Replays pre-planned action sequences from MPPI planning phase.
    Uses backend.step_act() for 6D offset → PD control conversion.

    Typical usage (two-phase plan-then-execute):
        1. Plan with ManiDreamsEnv.dream() → action_history
        2. Create executor (1 env, 1 card)
        3. Replay: set_cage(cage) → execute(action) for each planned action
        4. close_and_retract() to grasp and lift
    """

    def __init__(self):
        super().__init__()
        self.backend = PickBackend()
        self.current_obs = None
        self._current_cage = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Create picking execution environment."""
        self.env = self.backend.create_environment(config)
        import torch
        actual_env = self.env.unwrapped
        self.device = actual_env.device if hasattr(actual_env, 'device') else torch.device('cpu')
        self.initialized = True

    def set_cage(self, cage) -> None:
        """Set current cage for action conversion in step_act().

        Must be called before execute() — step_act() uses cage.center
        to compute the reference position for 6D offset actions.
        """
        self._current_cage = cage

    def execute(self, actions: Union[Any, List[Any]],
                get_feedback: bool = True) -> Union[Tuple[Any, Dict], Tuple[List[Any], List[Dict]]]:
        """Execute 6D offset action(s) via backend.step_act()."""
        if not isinstance(actions, list):
            return self._execute_single(actions, get_feedback)
        observations, feedbacks = [], []
        for action in actions:
            obs, fb = self._execute_single(action, get_feedback)
            observations.append(obs)
            feedbacks.append(fb)
        return observations, feedbacks

    def _execute_single(self, action: Any,
                        get_feedback: bool = True) -> Tuple[Any, Dict]:
        """Execute one 6D offset action: backend.step_act(action, env, cage)."""
        obs = self.backend.step_act(
            action, self.env.unwrapped,
            cage=self._current_cage, single_action=True,
            visualize_cage=False
        )
        self.current_obs = obs
        return obs, {}

    def get_obs(self) -> Any:
        """Get current observation (last obs from env)."""
        return self.current_obs

    def get_state(self) -> Dict:
        """Get state from execution env (for TSIP sync).

        Returns dict with: obj_pose [num_objects, 7], obj_positions [num_objects, 3].
        """
        return self.backend.get_state(self.env)

    def state_to_dris(self, state: Dict, **kwargs):
        """Convert state to DRIS (delegates to backend)."""
        return self.backend.state2dris(state, **kwargs)

    def close_and_retract(self, move_direction=(0, -0.1, 0.1), num_steps=50):
        """Close gripper and retract in specified direction.

        Two phases:
        1. Close gripper at current position (10 PD steps)
        2. Move linearly in move_direction with gripper closed (num_steps PD steps)

        Args:
            move_direction: (dx, dy, dz) displacement in world coordinates
            num_steps: Number of PD control steps for the retract motion
        """
        import torch

        env = self.env.unwrapped
        device = env.device

        logger.info("Executing closing action: close gripper + retract")
        logger.info("  Move direction: %s", move_direction)

        # Compute desired position by undoing control_ee_pose offsets
        current_qpos = env.agent.robot.get_qpos()
        current_pos = current_qpos[:, :3]
        current_ori = current_qpos[:, 3:6]

        pos_offset = torch.tensor([0, 0, -0.33], device=device, dtype=torch.float32)
        desired_pos = current_pos - pos_offset.unsqueeze(0)

        displacement = torch.tensor(move_direction, device=device, dtype=torch.float32)

        ori_offset = torch.tensor([0, torch.pi, 0], device=device, dtype=torch.float32)
        target_ori = current_ori - ori_offset.unsqueeze(0)

        # Phase 1: Close gripper at current position
        logger.info("  Phase 1: Closing gripper...")
        for _ in range(10):
            action_vec = self.backend.control_ee_pose(
                env, desired_pos, target_ori,
                Kp_pos=self.backend.Kp_pos, Kp_ori=self.backend.Kp_ori
            )
            action_vec[:, -1] = -0.9
            env.step(action_vec)
            if hasattr(env, 'render_mode') and env.render_mode == 'human':
                try:
                    env.render()
                except AttributeError:
                    pass

        # Phase 2: Move in direction with gripper closed
        logger.info("  Phase 2: Moving gripper by %s over %d steps...", move_direction, num_steps)
        for step in range(num_steps):
            t = (step + 1) / num_steps
            interp_pos = desired_pos + displacement.unsqueeze(0) * t

            action_vec = self.backend.control_ee_pose(
                env, interp_pos, target_ori,
                Kp_pos=self.backend.Kp_pos, Kp_ori=self.backend.Kp_ori
            )
            action_vec[:, -1] = -0.9
            env.step(action_vec)
            if hasattr(env, 'render_mode') and env.render_mode == 'human':
                try:
                    env.render()
                except AttributeError:
                    pass

        logger.info("  Closing action completed!")

    def reset(self) -> Tuple[Any, Dict]:
        obs, info = self.env.reset()
        self.current_obs = obs
        # Hide goal_site (green cage sphere) — not needed during execution replay
        actual_env = self.env.unwrapped
        if hasattr(actual_env, 'goal_site'):
            import torch
            from mani_skill.utils.structs import Pose
            hidden_pos = torch.tensor([0, 0, -10], device=self.device, dtype=torch.float32)
            actual_env.goal_site.set_pose(Pose.create_from_pq(p=hidden_pos))
        return obs, info

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
