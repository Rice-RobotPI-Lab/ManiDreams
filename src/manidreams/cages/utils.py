"""
Shared utility functions for cage implementations.

Provides common position extraction and distance evaluation logic
to avoid code duplication across geometric cage classes.
"""

from typing import List, Union
import numpy as np

# Type hint for DRIS without circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base.dris import DRIS


def extract_position_from_dris(dris: 'DRIS') -> np.ndarray:
    """
    Extract 3D position from DRIS observation.

    Handles various observation formats:
    - 1D array: returns first 3 elements
    - 2D array: returns mean of first 3 columns (center of mass)
    - Torch tensor: converts to numpy first

    Args:
        dris: DRIS containing state observation

    Returns:
        3D position as numpy array [x, y, z]
    """
    obs = dris.observation

    # Convert torch tensor to numpy if needed
    if hasattr(obs, 'cpu'):
        obs = obs.cpu().numpy()

    # Extract 3D position
    if isinstance(obs, np.ndarray):
        if len(obs.shape) == 1 and obs.shape[0] >= 3:
            # Single object: use first 3 dimensions [x, y, z]
            return obs[:3].copy()
        elif len(obs.shape) == 2 and obs.shape[1] >= 3:
            # Multiple objects: use center of mass
            positions = obs[:, :3]
            return np.mean(positions, axis=0)

    # Fallback
    return np.array([0.0, 0.0, 0.0])


def evaluate_distance_to_center(
    dris_input: Union['DRIS', List['DRIS']],
    center: np.ndarray
) -> List[float]:
    """
    Evaluate distance from DRIS position(s) to cage center.

    Generic distance evaluation for spherical cages.

    Args:
        dris_input: Single DRIS or list of DRIS states
        center: Cage center position (3D numpy array)

    Returns:
        List of distances (lower is better)
    """
    dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

    costs = []
    for dris in dris_list:
        position = extract_position_from_dris(dris)
        distance = np.linalg.norm(position - center)
        costs.append(distance)

    return costs


def evaluate_position_variance(
    dris_input: Union['DRIS', List['DRIS']],
    center: np.ndarray
) -> List[float]:
    """
    Evaluate clustering cost for card gathering during pushing.

    Composite cost focused on x-axis compactness (lateral gathering):
      1) X-axis variance: primary compactness signal
      2) X-centering: penalize centroid deviating from cage center in x
      3) Outlier penalty: penalize the most spread-out card to prevent escapes

    Z-axis variance is excluded (cards are flat on table, z-var is noise).
    Y-axis variance is excluded (push direction, no need to cluster along y).

    Args:
        dris_input: Single DRIS or list of DRIS states
        center: Cage center position (used for x-centering)

    Returns:
        List of clustering costs (lower is better)
    """
    dris_list = dris_input if isinstance(dris_input, list) else [dris_input]

    costs = []
    for dris in dris_list:
        obs = dris.observation

        # Convert torch tensor to numpy if needed
        if hasattr(obs, 'cpu'):
            obs = obs.cpu().numpy()

        if not isinstance(obs, np.ndarray):
            costs.append(0.0)
            continue

        # Reshape flattened observation: [num_objects * 7] -> [num_objects, 7]
        num_objects = len(obs) // 7
        if num_objects < 2:
            costs.append(0.0)
            continue

        positions = obs[:num_objects * 7].reshape(num_objects, 7)[:, :3]  # [N, 3]
        x_positions = positions[:, 0]  # [N]

        # Term 1: X-axis variance (primary compactness signal)
        x_variance = float(np.var(x_positions))

        # Term 2: X-centering (penalize centroid offset from cage center)
        mean_x = np.mean(x_positions)
        x_centering = (mean_x - center[0]) ** 2

        # Term 3: Outlier penalty (max distance from centroid in x)
        x_outlier = float(np.max(np.abs(x_positions - mean_x)))

        cost = 2.0 * x_variance + 1.0 * x_centering + 0.5 * x_outlier

        costs.append(cost)

    return costs


def validate_within_radius(
    dris_input: Union['DRIS', List['DRIS']],
    center: np.ndarray,
    radius: float
) -> List[bool]:
    """
    Validate if DRIS position(s) are within cage radius.

    Args:
        dris_input: Single DRIS or list of DRIS states
        center: Cage center position (3D numpy array)
        radius: Cage radius

    Returns:
        List of bools (True if within radius)
    """
    costs = evaluate_distance_to_center(dris_input, center)
    return [cost <= radius for cost in costs]
