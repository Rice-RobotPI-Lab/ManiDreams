"""Denoiser for diffusion-based world model (inference only)."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from .inner_model import InnerModel, InnerModelConfig


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    noise_previous_obs: bool
    upsampling_factor: Optional[int] = None


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_upsampler = cfg.upsampling_factor is not None
        cfg.inner_model.is_upsampler = self.is_upsampler
        self.inner_model = InnerModel(cfg.inner_model)

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Optional[Tensor], cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs, act)

    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, sigma_cond: Optional[Tensor], obs: Tensor, act: Optional[Tensor]) -> Tensor:
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised
