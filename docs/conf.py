"""Sphinx configuration for ManiDreams documentation."""

import os
import sys

# Add src directory to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------

project = "ManiDreams"
copyright = "2024, ManiDreams Team"
author = "ManiDreams Team"
release = "0.1.0"

# -- General configuration -----------------------------------------------------

extensions = [
    "myst_parser",                    # Markdown support
    "sphinx.ext.autodoc",             # API auto-generation from docstrings
    "sphinx.ext.napoleon",            # Google-style docstring parsing
    "sphinx.ext.viewcode",            # [source] links to highlighted source
    "sphinx.ext.intersphinx",         # Cross-project links (numpy, python)
    "sphinx_autodoc_typehints",       # Type hints in docs
    "sphinx_copybutton",              # Copy button on code blocks
    "sphinx_design",                  # Cards, grids, badges, tabs
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document (renamed from index.md to avoid conflict with landing page)
master_doc = "documentation"

# Templates path (contains custom landing page)
templates_path = ["_templates"]

# Custom standalone pages (Nerfies-style landing page)
html_additional_pages = {"index": "landing.html"}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress known harmless warnings (dataclass duplicate field descriptions)
suppress_warnings = ["ref.duplicate_object"]

# Mock heavy dependencies that are unavailable in CI / docs build environment
autodoc_mock_imports = [
    "mani_skill",
    "torch",
    "gymnasium",
    "cv2",
    "transforms3d",
    "tensordict",
    "torchrl",
    "hydra",
    "omegaconf",
    "wandb",
    "tensorboard",
]

# -- MyST (Markdown) configuration ---------------------------------------------

myst_enable_extensions = [
    "colon_fence",      # ::: directive syntax
    "deflist",          # Definition lists
    "fieldlist",        # Field lists
]

# -- Napoleon (Google-style docstrings) -----------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# -- Autodoc -------------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_class_content = "class"  # Avoid duplicate field docs for dataclasses
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output ---------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "ManiDreams"
html_show_copyright = True
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/Rice-RobotPI-Lab/ManiDreams",
    "repository_provider": "github",
    "use_repository_button": True,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "path_to_docs": "docs/",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "logo": {
        "text": "ManiDreams",
    },
}

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ]
}
