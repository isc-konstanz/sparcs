#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
penguin
~~~~~~~

Legacy compatibility setup script for the penguin package.

"""

import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
