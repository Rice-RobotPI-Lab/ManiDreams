"""
ManiDreams Asset Management Utilities.

Provides path utilities for accessing bundled robot assets and other resources.
This allows ManiDreams to use custom robots without depending on ManiSkill's
internal asset directory.

Usage:
    from assets import MANIDREAMS_ASSET_DIR, MANIDREAMS_ROBOT_DIR

    # Get path to a specific robot's URDF
    urdf_path = MANIDREAMS_ROBOT_DIR / "floating_panda_gripper" / "panda_fin_gripper.urdf"
"""

from pathlib import Path

# Root directory for all ManiDreams assets
MANIDREAMS_ASSET_DIR = Path(__file__).parent

# Directory containing robot definitions (URDF, meshes, agent classes)
MANIDREAMS_ROBOT_DIR = MANIDREAMS_ASSET_DIR / "robots"


def get_robot_path(robot_name: str) -> Path:
    """Get the directory path for a specific robot.

    Args:
        robot_name: Name of the robot (e.g., "floating_panda_gripper")

    Returns:
        Path to the robot's asset directory
    """
    return MANIDREAMS_ROBOT_DIR / robot_name


def get_robot_urdf(robot_name: str, urdf_filename: str = None) -> Path:
    """Get the URDF file path for a specific robot.

    Args:
        robot_name: Name of the robot (e.g., "floating_panda_gripper")
        urdf_filename: Optional specific URDF filename. If not provided,
                      defaults to "{robot_name}.urdf"

    Returns:
        Path to the robot's URDF file
    """
    robot_dir = get_robot_path(robot_name)
    if urdf_filename is None:
        urdf_filename = f"{robot_name}.urdf"
    return robot_dir / urdf_filename


__all__ = [
    "MANIDREAMS_ASSET_DIR",
    "MANIDREAMS_ROBOT_DIR",
    "get_robot_path",
    "get_robot_urdf",
]
