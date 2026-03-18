# Changelog

## v0.1.0 (Current)

Initial release.

- Three-layer modular architecture (Abstract Interfaces → Concrete Implementations → Task Integration)
- DRIS as universal state representation with domain randomization support
- Cage-constrained action selection via `ManiDreamsEnv.dream()`
- Simulation-based and learning-based TSIP backends
- Solvers: PolicySampler (RL), MPPIOptimizer (planning), GeometricOptimizer (discrete)
- Cage implementations: CircularCage, DRISCage, PlateCage, Geometric3DCage, CircularPixelCage
- Five task categories: object pushing, catching, picking, pixel-based pushing, ManiSkill defaults
- Documentation site with Sphinx + sphinx-book-theme, deployed via GitHub Pages
