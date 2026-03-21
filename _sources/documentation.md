# 💤 ManiDreams Documentation (v0.1.0)

```{image} _static/media/splash.svg
:alt: ManiDreams
:width: 100%
```

**ManiDreams**: An Open-Source Library for Robust Object Manipulation via Uncertainty-aware Task-specific Intuitive Physics

ManiDreams implements a three-layer modular architecture that separates abstract interfaces from concrete algorithm implementations and task-specific integrations. The framework enables cage-constrained action selection — where a virtual constraint (cage) bounds a Domain-Randomized Instance Set (DRIS) to prevent divergence — across diverse manipulation tasks, physics backends, and solver strategies.

```{raw} html
<div style="display:flex;gap:0.75rem;margin:1rem 0;">
  <a href="index.html" style="flex:1;text-align:center;padding:0.5rem 1rem;border:1px solid #333;border-radius:0.25rem;text-decoration:none;color:#333;font-weight:500;">Project Page</a>
  <a href="https://github.com/Rice-RobotPI-Lab/ManiDreams" style="flex:1;text-align:center;padding:0.5rem 1rem;border:1px solid #333;border-radius:0.25rem;text-decoration:none;color:#333;font-weight:500;">GitHub</a>
</div>
```

:::{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
:::

:::{toctree}
:maxdepth: 2
:caption: About the Project

why_manidreams
faq
changelog
:::

:::{toctree}
:maxdepth: 2
:caption: Core Concepts

dris
cage_constraints
tsip
solvers
:::

:::{toctree}
:maxdepth: 2
:caption: Tasks

supported_tasks
custom_tasks
zeroshot_real2sim
:::

:::{toctree}
:maxdepth: 2
:caption: API Reference

api/index
:::
