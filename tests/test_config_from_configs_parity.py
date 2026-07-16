# -*- coding: utf-8 -*-
"""Optional pin (issue 16 / unit 1.5): FeddesConfig/PondingConfig.from_configs
produces the same resolved fields as the __init__ shim, for both classes, with
and without base=. Guards the from_configs collapse -- fails pre-change with
AttributeError, since from_configs does not exist before this unit.
"""

from lories import Configurations
from sparcs.components.agriculture.simulation._soil import FeddesConfig, PondingConfig


def _configs(tmp_path, name="t.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


def test_feddes_from_configs_matches_init_no_base(tmp_path):
    cfg = _configs(tmp_path, enabled=True, root_distribution="  Linear  ", p2_pf=2.5)

    via_init = FeddesConfig(cfg)
    via_classmethod = FeddesConfig.from_configs(cfg)

    assert via_classmethod == via_init
    assert via_classmethod.root_distribution == "linear"


def test_feddes_from_configs_matches_init_with_base(tmp_path):
    base = FeddesConfig(_configs(tmp_path, name="base.conf", enabled=True, p2_pf=2.8))
    cfg = _configs(tmp_path, name="override.conf", p0_pf=0.5)

    via_init = FeddesConfig(cfg, base=base)
    via_classmethod = FeddesConfig.from_configs(cfg, base=base)

    assert via_classmethod == via_init


def test_ponding_from_configs_matches_init_no_base(tmp_path):
    cfg = _configs(tmp_path, h_max_mm=7.0)

    via_init = PondingConfig(cfg)
    via_classmethod = PondingConfig.from_configs(cfg)

    assert via_classmethod == via_init
    assert via_classmethod.watering_h_max_mm == 7.0  # follows h_max_mm, no base


def test_ponding_from_configs_matches_init_with_base(tmp_path):
    base = PondingConfig(_configs(tmp_path, name="base.conf", h_max_mm=8.0, watering_h_max_mm=50.0))
    cfg = _configs(tmp_path, name="override.conf", h_max_mm=20.0)

    via_init = PondingConfig(cfg, base=base)
    via_classmethod = PondingConfig.from_configs(cfg, base=base)

    assert via_classmethod == via_init
    assert via_classmethod.watering_h_max_mm == 50.0  # follows base, not new h_max_mm
