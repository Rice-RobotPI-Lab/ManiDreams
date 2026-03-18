"""
Diffusion Model Backend for ManiDreams Framework

Replicates PlayCageEnv behavior from Diamond's play_cage16.py.
Maintains observation tensor state internally like PlayCageEnv.
Uses internalized Diamond inference code (no external Diamond dependency).
"""

import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Add ManiDreams root to path for imports
manidreams_root = Path(__file__).parent.parent.parent.parent
if str(manidreams_root) not in sys.path:
    sys.path.insert(0, str(manidreams_root))
sys.path.insert(0, str(manidreams_root / "src"))

from manidreams.base.dris import DRIS
from .diamond import Agent, WorldModelEnv, load_model_config


class DiffusionBackend:
    """
    Backend that replicates PlayCageEnv behavior.

    Key difference from original: Maintains self.obs tensor internally,
    just like PlayCageEnv.obs (play_cage_env.py:237).
    """

    def __init__(self, visualizer=None):
        self.wm_env = None
        self.agent = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = visualizer

        # Key: Maintain observation tensor like PlayCageEnv
        self.obs = None  # Tensor observation (like PlayCageEnv.obs)
        self.t = 0       # Timestep counter (like PlayCageEnv.t)

        # Action space (set dynamically when model is loaded)
        self.num_actions = 8  # Default, will be overridden by env config

        # For visualization
        self._current_cage_center = None
        self._current_cage_radius = None

        logger.info("DiffusionBackend initialized with device: %s", self.device)

    def load_model(self, model_config: Dict[str, Any]):
        """
        Load model and create WorldModelEnv.

        Reads config from model_dir/config/ YAML files (no Hydra dependency).
        Dynamically reads num_actions from env config YAML.
        """
        model_name = model_config.get('model_name', 'push')
        model_dir = Path(model_config['model_dir'])
        wm_env_style = model_config.get('wm_env_style', 'fast')

        logger.info("Loading model: %s from %s", model_name, model_dir)

        # Load configs from YAML (replaces Hydra compose + instantiate)
        agent_cfg, wm_env_cfg, num_actions = load_model_config(model_dir, model_name, wm_env_style)

        logger.info("Detected num_actions from env config: %d", num_actions)
        self.num_actions = num_actions

        # Create agent
        logger.info("Creating agent with %d actions...", num_actions)
        self.agent = Agent(agent_cfg).to(self.device).eval()

        # Load weights
        ckpt_path = model_dir / f'model/{model_name}.pt'
        logger.info("Loading checkpoint: %s", ckpt_path)
        self.agent.load(ckpt_path)

        # Create WorldModelEnv
        spawn_dir = model_dir / 'spawn'
        sl = agent_cfg.denoiser.inner_model.num_steps_conditioning
        if self.agent.upsampler is not None:
            sl = max(sl, agent_cfg.upsampler.inner_model.num_steps_conditioning)

        logger.info("Creating WorldModelEnv...")
        self.wm_env = WorldModelEnv(
            self.agent.denoiser,
            self.agent.upsampler,
            self.agent.rew_end_model,
            spawn_dir,
            1,  # num_envs
            sl,
            wm_env_cfg,
            return_denoising_trajectory=True
        )

        logger.info("Model loaded successfully")
        return self.wm_env

    def reset(self) -> DRIS:
        """
        Reset environment (replicates PlayCageEnv.reset() line 132-147)

        Returns:
            DRIS with observation as HWC [0,1] numpy array
        """
        if self.wm_env is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # PlayCageEnv:133
        self.obs, _ = self.wm_env.reset()
        self.t = 0

        # Convert to frame (play_cage16.py:263-267)
        frame = self._tensor_to_frame(self.obs)

        return DRIS(
            observation=frame,
            context={},
            metadata={'timestep': 0}
        )

    def predict_step(self, model: Any, current_state: Any, action: Any) -> Tuple[np.ndarray, Dict]:
        """
        Execute one step (replicates PlayCageEnv.step() line 190-239)

        Args:
            model: Ignored (we use self.wm_env)
            current_state: Ignored (we maintain self.obs internally)
            action: Action index (0-15) or one-hot tensor

        Returns:
            (next_frame, info) where next_frame is HWC [0,1] numpy
        """
        # Convert action to one-hot tensor
        if isinstance(action, int):
            if action == -1:
                # Empty action (rest/satisfied): all zeros, like CSGOAction(keys=[])
                action_tensor = torch.zeros(1, self.num_actions, dtype=torch.float32, device=self.device)
            else:
                # Normal action: one-hot encoding
                action_tensor = torch.zeros(1, self.num_actions, dtype=torch.float32, device=self.device)
                action_tensor[0, action] = 1.0
        else:
            action_tensor = action.to(self.device)

        # Execute world model step (PlayCageEnv:196)
        next_obs, rew, end, trunc, env_info = self.wm_env.step(action_tensor)

        # Update self.obs (PlayCageEnv:237) - KEY DIFFERENCE
        self.obs = next_obs
        self.t += 1

        # Convert to frame
        next_frame = self._tensor_to_frame(next_obs)

        # Visualize if visualizer available
        if self.visualizer:
            self._visualize_frame(next_frame, action)

        # Return
        info = {
            'reward': rew.item() if torch.is_tensor(rew) else rew,
            'terminated': end,
            'truncated': trunc
        }

        return next_frame, info

    def _tensor_to_frame(self, obs_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert observation tensor to frame (play_cage16.py:263-267)

        Diamond processing:
            frame = obs.cpu().numpy()
            if len(frame.shape) == 4: frame = frame[0]
            frame = frame.transpose(1, 2, 0)  # CHW -> HWC
            frame = (frame + 1) / 2           # [-1,1] -> [0,1]
        """
        frame = obs_tensor.cpu().numpy()
        if len(frame.shape) == 4:
            frame = frame[0]
        frame = frame.transpose(1, 2, 0)  # CHW -> HWC
        frame = (frame + 1) / 2           # [-1, 1] -> [0, 1]
        return frame

    def _visualize_frame(self, frame, action):
        """Call visualizer if available"""
        dris = DRIS(
            observation=frame,
            context={},
            metadata={'timestep': self.t}
        )

        should_continue = self.visualizer.process_frame(
            dris=dris,
            cage_center=self._current_cage_center,
            radius=self._current_cage_radius,
            direction_index=action if isinstance(action, int) else None,
            timestep=self.t,
            action_info={'action': action}
        )

        if not should_continue:
            raise KeyboardInterrupt("User requested exit via visualizer")

    def set_cage_info(self, cage_center, cage_radius):
        """Set current cage info for visualization"""
        self._current_cage_center = cage_center
        self._current_cage_radius = cage_radius

    def get_observation_space(self):
        """Return observation space"""
        return {
            'shape': (64, 64, 3),
            'dtype': np.float32
        }

    def get_dris(self) -> DRIS:
        """
        Get current DRIS from backend internal state.

        Converts internal tensor (CHW [-1,1]) to DRIS with HWC [0,1] observation.

        Returns:
            DRIS object with current observation
        """
        from manidreams.base.dris import DRIS

        if self.obs is None:
            return None

        # Convert internal tensor to HWC [0,1] frame
        frame = self._tensor_to_frame(self.obs)

        return DRIS(
            observation=frame,
            context={},
            metadata={'timestep': self.t}
        )

    def set_dris(self, dris: DRIS) -> None:
        """
        Set backend internal state from DRIS.

        Converts DRIS observation (HWC [0,1]) to internal tensor (CHW [-1,1])
        and synchronizes WorldModelEnv internal state.

        Args:
            dris: DRIS object with observation to set

        Example:
            # After executor feedback
            synthetic_image = generate_synthetic_observation(object_pos)
            new_dris = DRIS(
                observation=synthetic_image,
                context={},
                metadata={'timestep': 100}
            )
            backend.set_dris(new_dris)
        """
        import torch

        if dris is None or dris.observation is None:
            raise ValueError("DRIS or DRIS.observation is None")

        observation = dris.observation

        # Convert to CHW [-1,1] tensor
        if isinstance(observation, np.ndarray):
            # Check if HWC or CHW format
            if observation.shape[-1] == 3:  # HWC format (expected)
                obs_chw = observation.transpose(2, 0, 1)  # HWC → CHW
            else:  # CHW format
                obs_chw = observation

            # Normalize to [-1, 1] range if needed
            if obs_chw.max() <= 1.0:  # Assume [0, 1] range
                obs_chw = obs_chw * 2 - 1  # [0,1] → [-1,1]

            # Convert to tensor with batch dimension
            if len(obs_chw.shape) == 3:  # (3, H, W)
                obs_tensor = torch.from_numpy(obs_chw).unsqueeze(0).float()
            else:  # Already has batch dimension
                obs_tensor = torch.from_numpy(obs_chw).float()

            self.obs = obs_tensor.to(self.device)

        elif torch.is_tensor(observation):
            # Already a tensor
            if len(observation.shape) == 3:  # (3, H, W)
                observation = observation.unsqueeze(0)
            self.obs = observation.to(self.device)
        else:
            raise ValueError(
                f"Unsupported observation type in DRIS: {type(observation)}. "
                f"Expected numpy array or torch tensor."
            )

        # Update timestep from metadata
        if dris.metadata and 'timestep' in dris.metadata:
            self.t = dris.metadata['timestep']

        # CRITICAL: Update WorldModelEnv internal state
        # Without this, wm_env.step() will use old state!
        if self.wm_env is not None:
            # WorldModelEnv maintains a temporal buffer: [num_envs, seq_len, C, H, W]
            # For num_envs=1: [1, seq_len, 3, 128, 128]
            # We need to update the LAST frame in the buffer: obs_buffer[:, -1]
            # which has shape [1, 3, 128, 128] (matches self.obs shape)

            if hasattr(self.wm_env, 'obs_buffer'):
                # self.obs: [1, 3, H, W]
                # obs_buffer[:, -1]: [1, 3, H, W]
                # Direct assignment (shapes match)
                self.wm_env.obs_buffer[:, -1] = self.obs
                logger.debug("Synchronized WorldModelEnv.obs_buffer[:, -1]: shape=%s", self.wm_env.obs_buffer[:, -1].shape)

            # Also update full-res buffer if upsampling is used
            if hasattr(self.wm_env, 'obs_full_res_buffer') and self.wm_env.obs_full_res_buffer is not None:
                # obs_full_res_buffer expects 128x128, but self.obs is 64x64
                # Use simple bilinear upsampling
                import torch.nn.functional as F
                obs_upsampled = F.interpolate(
                    self.obs,
                    size=(128, 128),
                    mode='bilinear',
                    align_corners=False
                )
                self.wm_env.obs_full_res_buffer[:, -1] = obs_upsampled
                logger.debug("Synchronized WorldModelEnv.obs_full_res_buffer[:, -1]: shape=%s (upsampled)", self.wm_env.obs_full_res_buffer[:, -1].shape)

        logger.debug("DRIS set: shape=%s, timestep=%d", self.obs.shape, self.t)

    def get_action_space(self):
        """Return action space (dynamic based on loaded model)"""
        import gymnasium as gym
        # Return dynamic action space based on loaded model
        num_actions = getattr(self, 'num_actions', 8)  # Default to 8 if not yet loaded
        return gym.spaces.Discrete(num_actions)
