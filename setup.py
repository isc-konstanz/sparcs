#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sparcs
~~~~~~

Legacy compatibility setup script for the sparcs package.

"""

import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
