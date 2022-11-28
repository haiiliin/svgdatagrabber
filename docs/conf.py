# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# The full version, including alpha/beta/rc tags
import os
import sys

project = 'SvgDataGrabber'
copyright = '2022, WANG Hailin'
author = 'WANG Hailin'

try:
    import svgdatagrabber

    release = version = svgdatagrabber.__version__.split('+')[0]
except (ImportError, AttributeError):
    import warnings

    warnings.warn('svgdatagrabber is not installed, using 0.0.1')
    release = version = '0.0.1'
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
]

# AutoAPI configuration
autoapi_dirs = ['../svgdatagrabber']

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
