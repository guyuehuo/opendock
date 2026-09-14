# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'OpenDock'
copyright = '2021-2026, OpenDock contributors'
author = 'Qiuyue Hu, Zechen Wang, Yanjie Wei and Liangzhen Zheng'

release = '1.1.2'
version = '1.1.2'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = []
html_static_path = ['_static']
html_logo = '_static/logo.png'

# 引用自定义 CSS 文件
html_css_files = [
    'custom.css',
]
# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
