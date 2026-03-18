"""
Configuration loading for Diamond models.

Replaces Hydra's compose() + instantiate() with simple PyYAML-based loading.
Reads the same YAML files from model_dir/config/ but constructs dataclasses directly.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import yaml


def extract_state_dict(state_dict: OrderedDict, module_name: str) -> OrderedDict:
    """Extract sub-module state dict by prefix (from Diamond's utils.py)."""
    return OrderedDict({k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def _resolve_yaml_values(d):
    """
    Recursively resolve special string values from YAML:
    1. OmegaConf ${eval:'...'} expressions -> evaluated Python values
    2. Numeric strings that PyYAML safe_load missed (e.g., '2e-3') -> float/int

    PyYAML's safe_load doesn't recognize all numeric formats (e.g., '2e-3' without
    a decimal point is treated as a string). This function fixes that.
    """
    if isinstance(d, dict):
        return {k: _resolve_yaml_values(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_resolve_yaml_values(v) for v in d]
    elif isinstance(d, str):
        # Handle ${eval:'...'} expressions (OmegaConf syntax)
        if d.startswith("${eval:") and d.endswith("}"):
            expr = d[len("${eval:"):-1]
            if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
                expr = expr[1:-1]
            return eval(expr)
        # Try to parse numeric strings that PyYAML missed (e.g., '2e-3')
        try:
            return int(d)
        except ValueError:
            pass
        try:
            return float(d)
        except ValueError:
            pass
    return d


def _strip_target(d: dict) -> dict:
    """Remove Hydra's _target_ key from a dict."""
    return {k: v for k, v in d.items() if k != '_target_'}


def _build_inner_model_config(raw: dict):
    from .inner_model import InnerModelConfig
    raw = _strip_target(raw)
    return InnerModelConfig(**raw)


def _build_denoiser_config(raw: dict):
    from .denoiser import DenoiserConfig
    raw = _strip_target(raw)
    raw['inner_model'] = _build_inner_model_config(raw['inner_model'])
    return DenoiserConfig(**raw)


def _build_rew_end_model_config(raw: dict):
    from .rew_end_model import RewEndModelConfig
    raw = _strip_target(raw)
    return RewEndModelConfig(**raw)


def _build_sampler_config(raw: dict):
    from .diffusion_sampler import DiffusionSamplerConfig
    raw = _strip_target(raw)
    return DiffusionSamplerConfig(**raw)


def _build_agent_config(raw: dict, num_actions: int):
    from .agent import AgentConfig

    raw = _strip_target(raw)

    denoiser = _build_denoiser_config(raw['denoiser'])

    upsampler = None
    if raw.get('upsampler') is not None:
        upsampler = _build_denoiser_config(raw['upsampler'])

    rew_end_model = None
    if raw.get('rew_end_model') is not None:
        rew_end_model = _build_rew_end_model_config(raw['rew_end_model'])

    return AgentConfig(
        denoiser=denoiser,
        upsampler=upsampler,
        rew_end_model=rew_end_model,
        num_actions=num_actions,
    )


def _build_wm_env_config(raw: dict):
    from .world_model_env import WorldModelEnvConfig

    raw = _resolve_yaml_values(raw)
    raw = _strip_target(raw)

    sampler_next_obs = _build_sampler_config(raw['diffusion_sampler_next_obs'])

    sampler_upsampling = None
    if raw.get('diffusion_sampler_upsampling') is not None:
        sampler_upsampling = _build_sampler_config(raw['diffusion_sampler_upsampling'])

    return WorldModelEnvConfig(
        horizon=raw['horizon'],
        num_batches_to_preload=raw['num_batches_to_preload'],
        diffusion_sampler_next_obs=sampler_next_obs,
        diffusion_sampler_upsampling=sampler_upsampling,
    )


def load_model_config(
    model_dir,
    model_name: str,
    wm_env_style: str = 'fast',
):
    """
    Load Diamond model configuration from YAML files.

    Replaces Hydra's compose() + instantiate() with direct PyYAML loading.

    Args:
        model_dir: Path to model directory (e.g., diamond/models/push16)
        model_name: Model name (e.g., 'push16')
        wm_env_style: WorldModelEnv config style ('fast' or 'default')

    Returns:
        (agent_config, wm_env_config, num_actions)
    """
    model_dir = Path(model_dir)

    # 1. Load env config to get num_actions
    env_cfg_path = model_dir / f'config/env/{model_name}.yaml'
    with open(env_cfg_path) as f:
        env_raw = yaml.safe_load(f)
    num_actions = env_raw['num_actions']

    # 2. Load agent config and build AgentConfig
    agent_cfg_path = model_dir / f'config/agent/{model_name}.yaml'
    with open(agent_cfg_path) as f:
        agent_raw = yaml.safe_load(f)
    agent_config = _build_agent_config(agent_raw, num_actions)

    # 3. Load world_model_env config
    #    Try model-specific config first, then fall back to root diamond config
    wm_cfg_path = model_dir / f'config/world_model_env/{wm_env_style}.yaml'
    if not wm_cfg_path.exists():
        # Fall back: try root diamond config directory
        root_wm_cfg = model_dir.parent.parent / f'config/world_model_env/{wm_env_style}.yaml'
        if root_wm_cfg.exists():
            wm_cfg_path = root_wm_cfg
        else:
            raise FileNotFoundError(
                f"WorldModelEnv config not found at {wm_cfg_path} or {root_wm_cfg}"
            )

    with open(wm_cfg_path) as f:
        wm_raw = yaml.safe_load(f)
    wm_env_config = _build_wm_env_config(wm_raw)

    return agent_config, wm_env_config, num_actions
