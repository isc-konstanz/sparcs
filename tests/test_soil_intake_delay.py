# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_intake_delay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config-parse tests for ``[field_simulation] intake_delay``. The span behavior
it feeds (each tick reads inputs up to ``now - intake_delay``) is covered in
``test_field_simulation_tick``.

Importing ``base`` pulls the full lories + FiPy/Gmsh stack; ``importorskip``
keeps this out of environments that lack it (the full check runs on the box).
"""

import pytest

import pandas as pd

_base = pytest.importorskip("sparcs.components.agriculture.simulation.base")
FieldSimulation = _base.FieldSimulation

from lories import Configurations  # noqa: E402


def _configs(tmp_path, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=str(tmp_path), require=False, **values)


def test_intake_delay_absent_parses_to_zero(tmp_path):
    configs = _configs(tmp_path)
    assert FieldSimulation._parse_intake_delay(configs) == pd.Timedelta(0)


def test_intake_delay_parses_duration_string(tmp_path):
    configs = _configs(tmp_path, intake_delay="30min")
    assert FieldSimulation._parse_intake_delay(configs) == pd.Timedelta(minutes=30)


def test_intake_delay_default_attr_is_zero_on_field():
    """The class default reads up to now, so a config omitting the key holds nothing back."""
    assert FieldSimulation._intake_delay == pd.Timedelta(0)
