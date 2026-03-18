"""Executors for simulation and hardware deployment."""

from .maniskill_executor_base import ManiSkillExecutorBase
from .pushing_task_executor import PushingTaskExecutor
from .catching_task_executor import CatchingTaskExecutor
from .picking_task_executor import PickingTaskExecutor
from .franka_executor import FrankaExecutor

# For backwards compatibility
ManiSkillExecutor = PushingTaskExecutor

__all__ = [
    "ManiSkillExecutorBase",
    "PushingTaskExecutor",
    "CatchingTaskExecutor",
    "PickingTaskExecutor",
    "ManiSkillExecutor",
    "FrankaExecutor",
]