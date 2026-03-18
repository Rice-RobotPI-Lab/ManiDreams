"""Real-world hardware executor."""

from typing import List, Dict, Any, Tuple, Optional, Union
from ..base.executor import ExecutorBase


class RealWorldExecutor(ExecutorBase):
    """Execute actions on real hardware. Subclass for your robot."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "RealWorldExecutor")

    def initialize(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    def execute(self, actions: Union[Any, List[Any]],
                get_feedback: bool = True) -> Union[Tuple[Any, Dict], Tuple[List[Any], List[Dict]]]:
        """Execute action(s). List → sequence, single → one step."""
        if isinstance(actions, list):
            observations, feedbacks = [], []
            for action in actions:
                obs, fb = self._execute_single(action, get_feedback)
                observations.append(obs)
                feedbacks.append(fb)
            return observations, feedbacks
        return self._execute_single(actions, get_feedback)

    def _execute_single(self, action: Any,
                        get_feedback: bool = True) -> Tuple[Any, Dict]:
        """Execute one action. Subclass must override."""
        raise NotImplementedError

    def reset(self) -> Any:
        raise NotImplementedError

    def get_obs(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass
