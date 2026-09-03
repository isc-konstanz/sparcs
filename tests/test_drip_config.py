# -*- coding: utf-8 -*-
"""sparcs.tests.test_drip_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit 1.3: DripConfig -- the single parse of [soil_simulation.drip] (nozzle_count
x nozzle_flow_lph -> design_flow_lpm), and the explicit flag
FieldSimulation._validate_irrigation_input gates the state-driven fallback feed
on. SoilPredictor's [soil_predictor.drip] per-key override against an
already-resolved DripConfig is tested in test_soil_predictor_flow_schedule.py
(SoilPredictor._resolve_drip_layout).
"""

from typing import Optional

from lories import Configurations
from sparcs.components.agriculture.simulation._soil import (
    _DEFAULT_NOZZLE_COUNT,
    _DEFAULT_NOZZLE_FLOW_LPH,
    DripConfig,
    design_flow_lpm,
)


def _soil_block(tmp_dir: str, drip: Optional[dict] = None) -> Configurations:
    """A SoilSimulation-level config: no top-level values, at most one [drip] member."""
    kwargs = {} if drip is None else {"drip": drip}
    return Configurations.load("test.conf", conf_dir=tmp_dir, require=False, **kwargs)


def test_explicit_block_parses_keys_and_derives_flow(tmp_path):
    soil_block = _soil_block(str(tmp_path), drip={"nozzle_count": 32, "nozzle_flow_lph": 2.0})
    drip = DripConfig(soil_block)
    assert drip.explicit is True
    assert drip.nozzle_count == 32
    assert drip.nozzle_flow_lph == 2.0
    assert drip.design_flow_lpm == design_flow_lpm(32, 2.0)


def test_absent_block_defaults_and_not_explicit(tmp_path):
    """No [soil_simulation.drip] at all: not explicit, falls back to the shared
    nozzle defaults -- FieldSimulation._validate_irrigation_input then refuses
    the on/off state fallback feed for this field."""
    soil_block = _soil_block(str(tmp_path))
    drip = DripConfig(soil_block)
    assert drip.explicit is False
    assert drip.nozzle_count == _DEFAULT_NOZZLE_COUNT
    assert drip.nozzle_flow_lph == _DEFAULT_NOZZLE_FLOW_LPH
    assert drip.design_flow_lpm == design_flow_lpm(_DEFAULT_NOZZLE_COUNT, _DEFAULT_NOZZLE_FLOW_LPH)


def test_present_but_empty_block_is_explicit_with_defaults(tmp_path):
    """[soil_simulation.drip] present but empty: explicit=True (the block WAS
    declared), individual keys still fall back to the shared nozzle defaults."""
    soil_block = _soil_block(str(tmp_path), drip={})
    drip = DripConfig(soil_block)
    assert drip.explicit is True
    assert drip.nozzle_count == _DEFAULT_NOZZLE_COUNT
    assert drip.nozzle_flow_lph == _DEFAULT_NOZZLE_FLOW_LPH
