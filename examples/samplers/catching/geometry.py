"""
Geometry utility functions for rotation conversions and manipulations.

Migrated from plate_catcher_demo/controller/geometry.py
"""

import torch
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Fixed a bug in ManiSkill's quaternion_to_axis_angle function.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    rotvec = quaternions[..., 1:] / sin_half_angles_over_angles
    angles = rotvec.norm(dim=-1)

    idx = angles > torch.pi
    if idx.sum() > 0:
        axes = rotvec[idx, :] / angles[idx]
        rotvec[idx, :] = - axes * (2 * torch.pi - angles)

    return rotvec


def find_minimum_delta_rot_to_up(rot: torch.Tensor) -> torch.Tensor:
    """
    Find the minimum delta rotation to align the given rotation with the up direction (z-axis).

    Args:
        rot: Rotation vector in axis-angle representation.

    Returns:
        Minimum delta rotation vector to align with the up direction.
    """
    rotmats = axis_angle_to_matrix(rot)  # (N, 3, 3)
    zaxes = rotmats[:, :, 2]  # (N, 3)
    upaxes = torch.zeros_like(zaxes)
    upaxes[:, 2] = 1.0
    raxes = torch.cross(zaxes, upaxes, dim=-1)  # (N, 3)

    delta_rot = torch.zeros_like(rot)  # (N, 3)
    idx = raxes.norm(dim=-1) > 1e-6
    if idx.sum() > 0:
        # dot product between zaxes and upaxes
        angles = torch.acos(torch.clamp(
            torch.sum(zaxes[idx, :] * upaxes[idx, :], dim=-1), -1.0, 1.0
        )).unsqueeze(-1)  # (N, 1)
        delta_rot[idx, :] = angles * raxes[idx, :] / raxes[idx, :].norm(dim=-1, keepdim=True)
    return delta_rot


def find_minimum_delta_rot_from_up(normal: torch.Tensor) -> torch.Tensor:
    """
    Find the minimum rotation to align the up direction (z-axis) with the given normal.

    Args:
        normal: Target normal vector.

    Returns:
        Minimum rotation vector to align up direction with the target normal.
    """
    up = torch.tensor(
        [0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype
    ).unsqueeze(0).expand_as(normal)
    raxes = torch.cross(up, normal, dim=-1)  # (N, 3)

    delta_rot = torch.zeros_like(normal)  # (N, 3)
    idx = raxes.norm(dim=-1) > 1e-6
    if idx.sum() > 0:
        # dot product between zaxes and upaxes
        angles = torch.acos(torch.clamp(
            torch.sum(up[idx, :] * normal[idx, :], dim=-1), -1.0, 1.0
        )).unsqueeze(-1)  # (N, 1)
        delta_rot[idx, :] = angles * raxes[idx, :] / raxes[idx, :].norm(dim=-1, keepdim=True)
    return delta_rot


def find_nearest_quat_to_normal(quat: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """
    Find the nearest rotation to align the z-axis with the given normal.

    Args:
        quat: Quaternion rotation (real part first).
        normal: Target normal vector.

    Returns:
        Nearest quaternion to align z-axis with the target normal.
    """
    rotmats = quaternion_to_matrix(quat)  # (N, 3, 3)
    zaxes = rotmats[:, :, 2]  # (N, 3)
    zaxes = zaxes / zaxes.norm(dim=-1, keepdim=True)
    raxes = torch.cross(zaxes, normal, dim=-1)  # (N, 3)

    delta_rot = torch.zeros_like(normal)  # (N, 3)
    idx = raxes.norm(dim=-1) > 1e-6
    if idx.sum() > 0:
        # dot product between zaxes and upaxes
        angles = torch.acos(torch.clamp(
            torch.sum(zaxes[idx, :] * normal[idx, :], dim=-1), -1.0, 1.0
        )).unsqueeze(-1)  # (N, 1)
        delta_rot[idx, :] = angles * raxes[idx, :] / raxes[idx, :].norm(dim=-1, keepdim=True)

    delta_rotmat = axis_angle_to_matrix(delta_rot)
    new_rotmat = delta_rotmat @ rotmats
    new_quat = matrix_to_quaternion(new_rotmat)
    return new_quat
