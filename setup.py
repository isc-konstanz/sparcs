#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    th-e-yield
    ~~~~~~~~~~
    
    TH-E Yield provides a set of functions to calculate the energy yield of photovoltaic systems.
    It utilizes the independent pvlib toolbox, originally developed in MATLAB at Sandia National Laboratories,
    and can be found on GitHub "https://github.com/pvlib/pvlib-python".
    
"""
from os import path
from setuptools import setup, find_namespace_packages

here = path.abspath(path.dirname(__file__))
info = {}
with open(path.join("th_e_yield", "_version.py")) as f : exec(f.read(), info)

VERSION = info['__version__']

DESCRIPTION = 'TH-E Yield provides a set of functions to calculate the energy yield of photovoltaic systems.'

# Get the long description from the README file
# with open(path.join(here, 'README.md')) as f:
#     README = f.read()

NAME = 'th-e-yield'
LICENSE = 'GPLv3'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'adrian.minde@isc-konstanz.de'
URL = 'https://github.com/isc-konstanz/th-e-yield'

INSTALL_REQUIRES = [
    'numpy >= 1.16',
    'pandas >= 0.23',
    'pvlib >= 0.9',
    'NREL-PySAM >= 4.0',
    'th-e-core @ git+https://github.com/isc-konstanz/th-e-core.git@master'
]

EXTRAS_REQUIRE = {
    'eval': ['th-e-data[excel,plot] @ git+https://github.com/isc-konstanz/th-e-data.git@master']
}

SCRIPTS = ['bin/th-e-yield']

PACKAGES = find_namespace_packages(include=['th_e_yield*'])

SETUPTOOLS_KWARGS = {
    'zip_safe': False,
    'include_package_data': True
}

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    # long_description=README,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    scripts=SCRIPTS,
)
