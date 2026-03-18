"""Catching task executor — wraps PlateCatch execution environment."""

from typing import Dict, Any, Tuple, Union, List
from .maniskill_executor_base import ManiSkillExecutorBase
from examples.physics.catch_backend import CatchBackend


class CatchingTaskExecutor(ManiSkillExecutorBase):
    """Executor for plate catching task.

    Creates independent PlateCatch-v1 environment for execution.
    Provides get_state() for TSIP synchronization in CAGE mode.
    Works in both baseline mode (no TSIP) and CAGE mode.
    """

    def __init__(self):
        super().__init__()
        self.backend = CatchBackend()
        self.current_obs = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Create PlateCatch execution environment."""
        self.env = self.backend.create_environment(config)
        import torch
        actual_env = self.env.unwrapped
        self.device = actual_env.device if hasattr(actual_env, 'device') else torch.device('cpu')
        self.initialized = True

    def execute(self, actions: Union[Any, List[Any]],
                get_feedback: bool = True) -> Union[Tuple[Any, Dict], Tuple[List[Any], List[Dict]]]:
        """Execute action(s) via env.step(). Handles single or list."""
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
        """Execute one action: env.step(action)."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_obs = obs
        feedback = {}
        if get_feedback:
            feedback = {'reward': reward, 'terminated': terminated,
                        'truncated': truncated, 'info': info}
        return obs, feedback

    def get_obs(self) -> Any:
        """Get current observation (last obs from env)."""
        return self.current_obs

    def get_state(self) -> Dict:
        """Get state from execution env (for TSIP sync).

        Returns dict with: obj_pos, obj_vel, tcp_pos, tcp_quat, qpos, qvel.
        """
        return self.backend.get_state(self.env)

    def state_to_dris(self, state: Dict, **kwargs):
        """Convert state to DRIS (delegates to backend)."""
        return self.backend.state2dris(state, **kwargs)

    def reset(self) -> Tuple[Any, Dict]:
        obs, info = self.env.reset()
        self.current_obs = obs
        return obs, info

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
