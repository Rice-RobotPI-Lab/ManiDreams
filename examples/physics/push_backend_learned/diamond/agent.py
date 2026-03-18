"""Diamond Agent (inference only, no ActorCritic/training)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .denoiser import Denoiser, DenoiserConfig
from .rew_end_model import RewEndModel, RewEndModelConfig
from .config import extract_state_dict


@dataclass
class AgentConfig:
    denoiser: DenoiserConfig
    upsampler: Optional[DenoiserConfig]
    rew_end_model: Optional[RewEndModelConfig]
    num_actions: int

    def __post_init__(self) -> None:
        self.denoiser.inner_model.num_actions = self.num_actions
        if self.upsampler is not None:
            self.upsampler.inner_model.num_actions = self.num_actions
        if self.rew_end_model is not None:
            self.rew_end_model.num_actions = self.num_actions


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.denoiser = Denoiser(cfg.denoiser)
        self.upsampler = Denoiser(cfg.upsampler) if cfg.upsampler is not None else None
        self.rew_end_model = RewEndModel(cfg.rew_end_model) if cfg.rew_end_model is not None else None

    @property
    def device(self):
        return self.denoiser.device

    def load(
        self,
        path_to_ckpt: Path,
        load_denoiser: bool = True,
        load_upsampler: bool = True,
        load_rew_end_model: bool = True,
    ) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device, weights_only=False)
        if load_denoiser:
            self.denoiser.load_state_dict(extract_state_dict(sd, "denoiser"))
        if load_upsampler and self.upsampler is not None:
            self.upsampler.load_state_dict(extract_state_dict(sd, "upsampler"))
        if load_rew_end_model and self.rew_end_model is not None:
            self.rew_end_model.load_state_dict(extract_state_dict(sd, "rew_end_model"))
