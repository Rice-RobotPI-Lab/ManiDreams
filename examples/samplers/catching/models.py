"""
Neural network models for RL policy learning.

Migrated from plate_catcher_demo/models/models.py

Contains:
- MLP: Multi-layer perceptron with orthogonal initialization
- ActorCritic: PPO-style actor-critic network with optional FiLM conditioning
- FiLM: Feature-wise Linear Modulation layer
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)


def init_weights(sequential, std=np.sqrt(2), bias_const=0.0):
    """Initialize all Linear layers in a Sequential module."""
    for module in sequential:
        if isinstance(module, nn.Linear):
            layer_init(module, std, bias_const)


class MLP(nn.Module):
    """Multi-layer perceptron with ELU activation and orthogonal initialization."""

    def __init__(self, in_size, out_size, hidden_sizes):
        super(MLP, self).__init__()
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (in_size,) + hidden_sizes + (out_size,)
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            if i == len(self.layer_sizes) - 2:
                layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            else:
                layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        init_weights(self.mlp)

    def forward(self, x):
        return self.mlp(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: y = gamma(n) * z + beta(n)"""

    def __init__(self, z_dim=128, n_dim=3, hidden_size=64, init_scale=0.1):
        super(FiLM, self).__init__()
        self.z_dim = z_dim  # latent DRIS dimension
        self.n_dim = n_dim  # plate normal dimension
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(n_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * z_dim)
        )
        # init weights
        nn.init.xavier_uniform_(self.net[-1].weight, gain=init_scale)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, n):
        gb = self.net(n)  # (batch_size, 2 * z_dim)
        gamma, beta = gb.chunk(2, dim=-1)  # each of shape (batch_size, z_dim)
        return gamma * z + beta


class ActorCritic(nn.Module):
    """
    PPO-style Actor-Critic network with optional conditioning.

    Supports three conditioning modes:
    - None: No conditioning, uses raw observation
    - "film": Feature-wise Linear Modulation conditioning
    - "nembed": Normal vector embedding concatenation
    """

    def __init__(
        self,
        envs,
        obs_dim=None,
        action_dim=None,
        hidden_sizes=(256, 256, 256),
        cond="film"
    ):
        super(ActorCritic, self).__init__()

        if isinstance(envs, ManiSkillVectorEnv):
            self.num_objs = envs._env.num_objs
        elif isinstance(envs, gym.vector.SyncVectorEnv):
            self.num_objs = envs.envs[0].num_objs
        elif isinstance(envs, gym.vector.AsyncVectorEnv):
            self.num_objs = 200  # HARDCODED, need to be fixed
        else:
            self.num_objs = envs.num_objs

        if obs_dim is None:
            obs_dim = np.prod(envs.single_observation_space.shape)

        if action_dim is None:
            action_dim = np.prod(envs.single_action_space.shape)

        assert cond in [None, "film", "nembed"]
        self.cond = cond

        if self.cond == "film":
            self.film = FiLM(z_dim=obs_dim, n_dim=2, hidden_size=64, init_scale=0.1)
            self.post_ln = nn.LayerNorm(obs_dim)

            self.critic = MLP(obs_dim, 1, hidden_sizes)
            self.actor_mean = MLP(obs_dim, action_dim, hidden_sizes)
        elif self.cond == "nembed":
            nembed_dim = 64
            self.nembed = nn.Sequential(
                nn.Linear(2, nembed_dim),
                nn.ReLU()
            )
            init_weights(self.nembed)

            self.fuse_norm = nn.LayerNorm(obs_dim + nembed_dim)

            self.critic = MLP(obs_dim + nembed_dim, 1, hidden_sizes)
            self.actor_mean = MLP(obs_dim + nembed_dim, action_dim, hidden_sizes)
        else:
            self.critic = MLP(obs_dim, 1, hidden_sizes)
            self.actor_mean = MLP(obs_dim, action_dim, hidden_sizes)

        # policy output layer with scale 0.01 * np.sqrt(2)
        layer_init(self.actor_mean.mlp[-1], std=0.01 * np.sqrt(2))

        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, x, n):
        """Get value estimate for given observation and normal."""
        if self.cond == "film":
            z_f = self.film(x, n)
            z_f = self.post_ln(z_f)
        elif self.cond == "nembed":
            n_e = self.nembed(n)
            z_f = self.fuse_norm(torch.cat([x, n_e], dim=-1))
        else:
            z_f = x
        return self.critic(z_f)

    def get_action(self, x, n, deterministic=False):
        """Get action for given observation and normal."""
        if self.cond == "film":
            z_f = self.film(x, n)
            z_f = self.post_ln(z_f)
        elif self.cond == "nembed":
            n_e = self.nembed(n)
            z_f = self.fuse_norm(torch.cat([x, n_e], dim=-1))
        else:
            z_f = x
        action_mean = self.actor_mean(z_f)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, n, action=None):
        """Get action, log probability, entropy, and value for PPO training."""
        if self.cond == "film":
            z_f = self.film(x, n)
            z_f = self.post_ln(z_f)
        elif self.cond == "nembed":
            n_e = self.nembed(n)
            z_f = self.fuse_norm(torch.cat([x, n_e], dim=-1))
        else:
            z_f = x
        action_mean = self.actor_mean(z_f)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(z_f)
