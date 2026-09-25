# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_chain_flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Irrigation flow normalisation and the once-only strip-flux guard of ``WeatherChain``.
"""

import logging

import pytest

from sparcs.components.agriculture.fieldsim.core.chain import WeatherChain, flow_m3s_per_m
from sparcs.components.agriculture.fieldsim.core.config import FieldConfig, FieldSetup, SoilConfig

CHAIN_LOGGER = "sparcs.components.agriculture.fieldsim.core.chain"


def _chain(drip_line_length_m: float) -> WeatherChain:
    soil = SoilConfig.from_dict(
        {"mesh": {"dl": 0.05, "watering_width": 0.05}, "total_drip_line_length_m": drip_line_length_m}
    )
    setup = FieldSetup(field=FieldConfig.from_dict(), soil=soil, shading=None)
    return WeatherChain(setup, shading=None, et=None)


def test_default_length_keeps_per_metre_reading():
    assert flow_m3s_per_m(60.0, 1.0) == pytest.approx(60.0 / 60_000.0)


def test_whole_field_flow_divided_by_drip_line_length():
    assert flow_m3s_per_m(60.0, 1000.0) == pytest.approx(60.0 / 60_000.0 / 1000.0)


def test_absurd_strip_flux_warns_once(caplog):
    chain = _chain(1.0)
    with caplog.at_level(logging.WARNING, logger=CHAIN_LOGGER):
        chain._warn_absurd_strip_flux(flow_m3s_per_m(60.0, 1.0))
        chain._warn_absurd_strip_flux(flow_m3s_per_m(60.0, 1.0))
    warnings = [r for r in caplog.records if "strip flux" in r.getMessage()]
    assert len(warnings) == 1
    assert "total_drip_line_length_m" in warnings[0].getMessage()


def test_sane_strip_flux_stays_silent(caplog):
    chain = _chain(1000.0)
    with caplog.at_level(logging.WARNING, logger=CHAIN_LOGGER):
        chain._warn_absurd_strip_flux(flow_m3s_per_m(16.7, 1000.0))
    assert not [r for r in caplog.records if "strip flux" in r.getMessage()]
    assert not chain._strip_flux_warned
