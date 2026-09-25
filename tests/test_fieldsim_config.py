# -*- coding: utf-8 -*-
"""sparcs.tests.test_fieldsim_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config sections of the ``fieldsim`` skeleton: lories ``Parameter`` resolution,
bounds and choices, the strict unknown-key rule (recursive through declared
groups), typed sub-sections (``[mesh]`` / ``[drip]``) becoming objects, the
schema with its ``passthrough`` flag, and a real-shaped ``soil_simulation.conf``
loaded from disk.
"""

import pytest

from lories.core import ConfigurationError
from lories.core.configs.configurations import Configurations
from sparcs.components.agriculture.fieldsim.core.config import (
    DripConfig,
    FieldConfig,
    FieldSetup,
    MeshConfig,
    PlannerConfig,
    SoilConfig,
)
from sparcs.components.agriculture.fieldsim.core.shading import ShadingConfig

SOIL_CONF = """
type = "soil_simulation"
enabled = true
plot_structure = true
total_drip_line_length_m = 12.6

[drip]
nozzle_count = 32
nozzle_flow_lph = 1.0

[testing]
enabled = true
history_window = "30d"

[mesh]
filename = "./soil.msh"
dl = 0.05
height = 1.0
plant_width = 0.5
plant_height = 0.5
watering_width = 0.1
d_x = 0.5

[pde]
dt_max = 600

[anchor]
enabled = false

[data.channels.soil_30cm]
type = "float"

[probes.points.soil_30cm]
x = 0.0
depth = 0.3
"""


# --------------------------------------------------------------------------- resolution


def test_field_defaults_and_coercion():
    f = FieldConfig.from_dict({"lai_type": "apple", "interval": 30, "offset": 20, "intake_delay": "30min"})
    assert f.lai_type == "apple"
    assert f.interval == 30 and f.offset == 20
    assert f.intake_delay.total_seconds() == 1800
    assert f.interval_td.total_seconds() == 1800
    assert f.ndvi == 0.25  # default applied


@pytest.mark.parametrize(
    "values, fragment",
    [
        ({"lai_type": "corn"}, "must be one of"),
        ({"ndvi": 1.5}, "exceeds the maximum"),
        ({"roughness": -1.0}, "minimum"),
        ({"interval": 30, "offset": 45}, "offset must satisfy"),
    ],
)
def test_field_rejects_bad_values(values, fragment):
    with pytest.raises(ConfigurationError, match=fragment):
        FieldConfig.from_dict(values)


# --------------------------------------------------------------------------- strict keys


def test_unknown_top_level_key_is_hard_error():
    with pytest.raises(ConfigurationError, match=r"unknown configuration keys \['SoilConfig.typo'\]"):
        SoilConfig.from_dict({"mesh": {}, "typo": 1})


def test_unknown_key_inside_declared_group_is_hard_error():
    # ShadingConfig's geometry keys are flat (mirroring the live
    # GroundShading parser, which never nests them under a [tracker] table);
    # PlannerConfig.state is the nearest declared-children group left to
    # exercise the same recursive-into-a-group check.
    with pytest.raises(ConfigurationError, match=r"PlannerConfig.state.typo"):
        PlannerConfig.from_dict({"state": {"typo": 1}})


def test_section_given_where_scalar_expected():
    with pytest.raises(ConfigurationError, match="section given, scalar expected"):
        ShadingConfig.from_dict({"albedo": {"x": 1}})


def test_passthrough_group_accepts_anything():
    s = SoilConfig.from_dict({"mesh": {}, "pde": {"whatever": 1, "nested": {"deeper": True}}})
    assert s.pde["whatever"] == 1


def test_bool_scalar_is_not_mistaken_for_a_section():
    # lories has_member() reports bool values as members; the strict check must not.
    s = SoilConfig.from_dict({"mesh": {}, "discover_sensor_probes": True})
    assert s.discover_sensor_probes is True


# --------------------------------------------------------------------------- typed sub-sections


def test_mesh_is_required_and_becomes_an_object():
    with pytest.raises(ConfigurationError, match=r"Missing required configuration section '\[mesh\]'"):
        SoilConfig.from_dict({})
    s = SoilConfig.from_dict({"mesh": {"dl": 0.05, "d_x": 0.5, "watering_width": 0.1}})
    assert isinstance(s.mesh, MeshConfig)
    assert s.mesh.dl == 0.05 and s.mesh.dx == 0.5
    assert s.mesh.width is None  # filled by derive()


def test_mesh_geometry_rules():
    with pytest.raises(ConfigurationError, match="watering_width must be >= dl"):
        MeshConfig.from_dict({"dl": 0.2, "watering_width": 0.1})
    with pytest.raises(ConfigurationError, match="d_x must be larger than dl"):
        MeshConfig.from_dict({"dl": 0.5, "d_x": 0.5})
    with pytest.raises(ConfigurationError, match="non-negative integer"):
        MeshConfig.from_dict({"width": 3.3, "plant_width": 0.5, "d_x": 0.5})
    m = MeshConfig.from_dict({"plant_width": 0.5, "d_x": 0.5}).derive(bay_width=3.5)
    assert m.width == 3.5 and m.top_segments == 3


def test_drip_absent_means_placeholder_not_explicit():
    s = SoilConfig.from_dict({"mesh": {}})
    assert isinstance(s.drip, DripConfig)
    assert s.drip.explicit is False
    assert s.drip.design_flow_lpm == pytest.approx(1.0 / 60.0)
    s2 = SoilConfig.from_dict({"mesh": {}, "drip": {"nozzle_count": 32, "nozzle_flow_lph": 1.0}})
    assert s2.drip.explicit is True
    assert s2.drip.design_flow_lpm == pytest.approx(32.0 / 60.0)


def test_planner_drip_is_a_per_key_override_over_the_sims():
    soil = SoilConfig.from_dict({"mesh": {}, "drip": {"nozzle_count": 32, "nozzle_flow_lph": 2.0}})
    planner = PlannerConfig.from_dict({"drip": {"nozzle_count": 8}})
    setup = FieldSetup(field=FieldConfig.from_dict(), soil=soil, shading=None, planner=planner)
    assert setup.planner_drip.nozzle_count == 8
    assert setup.planner_drip.nozzle_flow_lph == 2.0  # not restated -> falls back to the sim's
    assert setup.planner_drip.explicit is True
    assert FieldSetup(field=setup.field, soil=soil, shading=None).planner_drip is soil.drip


# --------------------------------------------------------------------------- schema


def test_schema_marks_passthrough_groups_and_carries_bounds():
    sch = SoilConfig.schema()
    assert sch["pde"]["passthrough"] is True
    assert "passthrough" not in sch["mesh"]
    assert "dl" in sch["mesh"]["children"]
    assert sch["total_drip_line_length_m"]["type"] == "float"
    assert FieldConfig.schema()["lai_type"]["choices"] == ["fao", "grass", "apple"]


# --------------------------------------------------------------------------- from disk


def test_real_shaped_soil_conf_from_disk(tmp_path):
    (tmp_path / "soil_simulation.conf").write_text(SOIL_CONF, encoding="utf-8")
    configs = Configurations.load("soil_simulation.conf", data_dir=str(tmp_path), flat=True)
    s = SoilConfig()
    s.configure(configs)
    assert s.plot_structure is True
    assert s.total_drip_line_length_m == 12.6
    assert s.drip.explicit is True and s.drip.nozzle_count == 32
    assert s.mesh.dl == 0.05 and s.mesh.filename == "./soil.msh"
    assert s.pde["dt_max"] == 600
    assert "soil_30cm" in s.probes["points"]
    s.mesh.derive(bay_width=3.5)
    assert s.mesh.top_segments == 3
