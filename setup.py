#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    pvsys
    ~~~~~
    
    This repository provides a set of python functions and scripts to calculate the
    energy yield of photovoltaic systems.
    
"""
from os import path
from setuptools import setup, find_namespace_packages

here = path.abspath(path.dirname(__file__))
info = {}
with open(path.join("pvsys", "_version.py")) as f:
    exec(f.read(), info)

VERSION = info['__version__']

DESCRIPTION = 'This repository provides a set of python functions and scripts to calculate the ' \
              'energy yield of photovoltaic systems.'

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    README = f.read()

NAME = 'pvsys'
LICENSE = 'GPLv3'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'adrian.minde@isc-konstanz.de'
URL = 'https://github.com/isc-konstanz/pvsys'

INSTALL_REQUIRES = [
    'pvlib >= 0.10.1',
    'NREL-PySAM >= 4.0',
    'corsys @ git+https://github.com/isc-konstanz/corsys.git@v0.8.4'
]

EXTRAS_REQUIRE = {
    'eval': ['scisys @ git+https://github.com/isc-konstanz/scisys.git@v0.2.10']
}

SCRIPTS = ['bin/pvsys']

PACKAGES = find_namespace_packages(include=['pvsys*'])

SETUPTOOLS_KWARGS = {
    'zip_safe': False,
    'include_package_data': True
}

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=README,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    scripts=SCRIPTS,
    **SETUPTOOLS_KWARGS
)
