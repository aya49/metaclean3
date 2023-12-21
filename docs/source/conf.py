# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'MEtaClean3.0'
copyright = '2023, METAFORA Biosystems'
author = 'METAFORA Biosystems'

# The full version, including alpha/beta/rc tags.
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'autoapi.extension', # pip install sphinx-autoapi
    'sphinx_immaterial',
    'sphinx_immaterial.apidoc.python.apigen',
]

autosummary_generate = True

# auto generate api
autoapi_dirs = ['../../src']
# autoapi_root = 'autoapi'
autoapi_add_toctree_entry = False

# markdown
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Sets the default role of `content` to :python:`content`, which uses the custom Python syntax highlighting inline literal
default_role = "python"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_immaterial"
html_theme_options = {
    "site_url": "toc",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "content.code.annotate",
        "content.code.copy",
        "content.tabs.link",
        "content.tooltips",
        "header.autohide",
        "navigation.expand",
        "navigation.footer",
        "navigation.indexes",
        "navigation.instant",
        "navigation.top",
        "navigation.tracking",
        "search.highlight",
        "search.share",
        "search.suggest",
        "toc.follow",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "#1a2784",
            "accent": "#5f68a9",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "lime",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    "social": [
        {
            "icon": "fontawesome/brands/internet-explorer",
            "link": "https://www.metafora-biosystems.com/metaflow/",
            "name": "MetaFlow website",
        },
        {
            "icon": "fontawesome/brands/linkedin",
            "link": "https://www.linkedin.com/company/metafora-biosystems/",
        },
        {
            "icon": "fontawesome/brands/youtube",
            "link": "https://www.youtube.com/@metaforabiosystems6122/",
        },
    ],}
html_static_path = ["_static"]
html_title = "MetaClean3.0 documentation"
html_favicon = "_static/images/favicon.png"  # colored version of material/bookshelf.svg
html_logo = "_static/images/favicon.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

