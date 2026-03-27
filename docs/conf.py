"""Sphinx configuration for pytrist documentation."""

import os
import sys

# Make pytrist importable during the build
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "pytrist"
author = "pytrist contributors"
release = "0.1.0"
copyright = "2024, pytrist contributors"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",      # pull docstrings from the package
    "sphinx.ext.napoleon",     # parse NumPy-style docstrings
    "sphinx.ext.viewcode",     # add [source] links to each class/method
    "sphinx.ext.intersphinx",  # link to numpy/python docs
]

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
autodoc_member_order = "bysource"   # preserve definition order in source
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,         # skip members with no docstring
    "exclude-members": "__weakref__, __dict__, __module__",
}
# Show type hints in the signature rather than duplicating in Parameters
autodoc_typehints = "signature"

# ---------------------------------------------------------------------------
# napoleon (NumPy docstring parser)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# intersphinx — cross-links to external package docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/colbych/pytrist",
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_title = "pytrist"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
