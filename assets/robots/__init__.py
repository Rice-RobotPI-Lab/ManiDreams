"""
ManiDreams Custom Robot Registry.

This module provides custom robot agents that are registered with ManiSkill
upon import. These robots use local URDF and mesh assets bundled with ManiDreams,
rather than relying on ManiSkill's internal asset directory.

Usage:
    # Import to register all custom robots with ManiSkill
    import assets.robots

    # Or import specific robots
    from assets.robots import FloatingPandaGripperFin

    # Then use in environment config
    env_config = {
        'robot_uids': 'floating_panda_gripper_fin',
        ...
    }

Available Robots:
    - FloatingPandaGripperFin: Floating Panda gripper with FinRay fingers
"""

from .floating_panda_gripper import FloatingPandaGripperFin

__all__ = [
    "FloatingPandaGripperFin",
]
