# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CoffeaMate'
copyright = '2024, Joy Zhou'
author = 'Joy Zhou'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',          # Automatic documentation from docstrings
    'sphinx.ext.napoleon',         # Support for Google style and NumPy style docstrings
    'sphinx.ext.viewcode',         # Add links to source code in the documentation
    'sphinx.ext.autosummary',
    'sphinx.ext.todo'             # Support for TODOs in docstrings
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'  # Order of members in the documentation
napoleon_google_docstring = True     # Enable Google style docstrings
napoleon_numpy_docstring = True 

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
import os
import sys
sys.path.insert(0, os.path.abspath('..'))