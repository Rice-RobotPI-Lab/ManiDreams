"""
Task Configuration for ManiSkill DRIS Backend

Defines the configuration for each supported ManiSkill tabletop task,
including target object attributes, creation methods, and parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ObjectConfig:
    """Configuration for creating a DRIS object copy."""
    builder_type: str  # "cube", "sphere", "tee"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a ManiSkill tabletop task."""

    # Task identification
    task_id: str                              # e.g., "PushCube-v1"
    base_class_path: str                      # Full import path to base env class

    # Target object identification
    target_attr: str                          # Attribute name for target object (e.g., "obj", "cube")

    # Object creation config
    object_config: ObjectConfig               # How to create DRIS copies

    # Goal information
    goal_attr: str                            # Attribute for goal position (e.g., "goal_region")
    goal_pose_method: str = "pose.p"          # How to get goal position

    # Reward configuration
    reward_uses_goal_distance: bool = True    # Whether reward is based on distance to goal

    # Additional target objects (for tasks with multiple objects like PullCubeTool)
    additional_objects: List[str] = field(default_factory=list)


# ============================================================================
# Task Configurations
# ============================================================================

TASK_CONFIGS: Dict[str, TaskConfig] = {

    # PickCube-v1: Pick up a cube and move to goal
    "PickCube-v1": TaskConfig(
        task_id="PickCube-v1",
        base_class_path="mani_skill.envs.tasks.tabletop.pick_cube.PickCubeEnv",
        target_attr="cube",
        object_config=ObjectConfig(
            builder_type="cube",
            params={
                "half_size": 0.02,
                "color": [1.0, 0.0, 0.0, 1.0],  # Red
                "body_type": "dynamic",
            }
        ),
        goal_attr="goal_site",
        goal_pose_method="pose.p",
        reward_uses_goal_distance=True,
    ),

    # PushCube-v1: Push a cube to goal
    "PushCube-v1": TaskConfig(
        task_id="PushCube-v1",
        base_class_path="mani_skill.envs.tasks.tabletop.push_cube.PushCubeEnv",
        target_attr="obj",
        object_config=ObjectConfig(
            builder_type="cube",
            params={
                "half_size": 0.02,
                "color": np.array([12, 42, 160, 255]) / 255,  # Blue
                "body_type": "dynamic",
            }
        ),
        goal_attr="goal_region",
        goal_pose_method="pose.p",
        reward_uses_goal_distance=True,
    ),

    # PushT-v1: Push a T-shaped block to match goal T
    "PushT-v1": TaskConfig(
        task_id="PushT-v1",
        base_class_path="mani_skill.envs.tasks.tabletop.push_t.PushTEnv",
        target_attr="tee",
        object_config=ObjectConfig(
            builder_type="tee",
            params={
                # T-shape: Two boxes
                # First box (horizontal): half_size=[0.1, 0.025, 0.02] at y=-0.0375
                # Second box (vertical): half_size=[0.025, 0.075, 0.02] at y=0.0625
                "mass": 0.8,
                "static_friction": 3.0,
                "dynamic_friction": 3.0,
                "color": np.array([194, 19, 22, 255]) / 255,  # Red
                "body_type": "dynamic",
                # T-shape geometry
                "box1_half_size": [0.1, 0.025, 0.02],
                "box1_offset": [0.0, -0.0375, 0.0],
                "box2_half_size": [0.025, 0.075, 0.02],
                "box2_offset": [0.0, 0.0625, 0.0],
            }
        ),
        goal_attr="goal_tee",
        goal_pose_method="pose.p",
        reward_uses_goal_distance=True,
    ),

    # PullCube-v1: Pull a cube to goal
    "PullCube-v1": TaskConfig(
        task_id="PullCube-v1",
        base_class_path="mani_skill.envs.tasks.tabletop.pull_cube.PullCubeEnv",
        target_attr="obj",
        object_config=ObjectConfig(
            builder_type="cube",
            params={
                "half_size": 0.02,
                "color": np.array([12, 42, 160, 255]) / 255,  # Blue
                "body_type": "dynamic",
            }
        ),
        goal_attr="goal_region",
        goal_pose_method="pose.p",
        reward_uses_goal_distance=True,
    ),

    # RollBall-v1: Roll a ball to goal
    "RollBall-v1": TaskConfig(
        task_id="RollBall-v1",
        base_class_path="mani_skill.envs.tasks.tabletop.roll_ball.RollBallEnv",
        target_attr="ball",
        object_config=ObjectConfig(
            builder_type="sphere",
            params={
                "radius": 0.035,
                "color": [0.0, 0.2, 0.8, 1.0],  # Blue
                "body_type": "dynamic",
            }
        ),
        goal_attr="goal_region",
        goal_pose_method="pose.p",
        reward_uses_goal_distance=True,
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    """Get configuration for a task."""
    if task_id not in TASK_CONFIGS:
        raise ValueError(
            f"Task '{task_id}' not supported. "
            f"Supported tasks: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_id]


def list_supported_tasks() -> List[str]:
    """List all supported task IDs."""
    return list(TASK_CONFIGS.keys())
