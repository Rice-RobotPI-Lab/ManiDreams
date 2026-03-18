"""Simulation environment executor."""

from typing import List, Dict, Any, Tuple, Optional
from ..base.executor import ExecutorBase


class SimulationExecutor(ExecutorBase):
    """Execute actions in simulation environment."""
    
    def __init__(self, env: Any = None, name: Optional[str] = None):
        """Initialize with simulation environment."""
        super().__init__(name or "SimulationExecutor")
        self.env = env
        self.current_obs = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize simulation environment."""
        if 'env' in config:
            self.env = config['env']
        
        if self.env is None:
            raise ValueError("Simulation environment required")
        
        self.current_obs = self.env.reset()
        self.initialized = True
    
    def execute_sequence(self, action_sequence: List[Any], 
                        get_feedback: bool = True) -> Tuple[List[Any], List[Dict]]:
        """Execute action sequence in simulation."""
        if not self.initialized:
            raise RuntimeError("Executor not initialized")
        
        observations = []
        feedback_info = []
        
        for action in action_sequence:
            obs, info = self.execute_single(action, get_feedback)
            observations.append(obs)
            feedback_info.append(info)
        
        return observations, feedback_info
    
    def execute_single(self, action: Any, get_feedback: bool = True) -> Tuple[Any, Dict]:
        """Execute single action in simulation."""
        if not self.initialized or self.env is None:
            raise RuntimeError("Executor not initialized")
        
        # Execute action
        obs, reward, done, truncated, info = self.env.step(action)
        self.current_obs = obs
        
        # Collect feedback
        feedback = {}
        if get_feedback:
            feedback = {
                'reward': reward,
                'done': done,
                'truncated': truncated,
                'info': info
            }
        
        return obs, feedback
    
    def reset(self) -> Any:
        """Reset simulation environment."""
        if self.env is None:
            raise RuntimeError("No environment available")
        
        self.current_obs = self.env.reset()
        return self.current_obs
    
    def get_obs(self) -> Any:
        """
        Get current observation from simulation.

        Returns:
            Current observation (typically the last observation from env.step())
        """
        return self.current_obs
    
    def close(self) -> None:
        """Close simulation environment."""
        if self.env is not None and hasattr(self.env, 'close'):
            self.env.close()