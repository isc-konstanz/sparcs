# -*- coding: utf-8 -*-
"""
    pvsys.pv.db
    ~~~~~~~~~~~
    
    
"""
from abc import ABC, abstractmethod

import os
import json
import shutil
import time

from corsys.io import DatabaseException
from corsys import Configurations


class ModelDatabase(ABC):

    # noinspection PyShadowingBuiltins
    def __init__(self, configs: Configurations, type):
        self._lib_dir = os.path.join(configs.dirs.lib, type)

    def _parse_file(self, key: str, sub_dir: str = ''):
        file = key + '.json'
        file_dir = os.path.join(self._lib_dir, sub_dir)
        return file_dir, file

    def exists(self, key: str, sub_dir: str = ''):
        return os.path.isfile(os.path.join(*self._parse_file(key, sub_dir)))

    def read(self, key: str, sub_dir: str = ''):
        return self._read(*self._parse_file(key, sub_dir))

    @staticmethod
    def _read(path, file):
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            raise DatabaseException("Unable to locate module file %s".format(file_path))

        with open(file_path, encoding='utf-8') as file:
            return json.load(file)

    def write(self, key, data, sub_dir=''):
        file_dir, file = self._parse_file(key, sub_dir)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        self._write(file_dir, file, data)

    @staticmethod
    def _write(path, file, data):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            return

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(data, separators=(',', ':'), indent=4))

    @abstractmethod
    def build(self):
        pass

    def clean(self):
        if os.path.exists(self._lib_dir):
            shutil.rmtree(self._lib_dir)

            while os.path.exists(self._lib_dir):
                time.sleep(.1)

        os.makedirs(self._lib_dir)
