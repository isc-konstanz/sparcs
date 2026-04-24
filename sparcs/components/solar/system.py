# -*- coding: utf-8 -*-
"""
sparcs.components.solar.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import itertools
import os
import re
from collections import OrderedDict
from math import floor
from typing import Dict, Mapping, Optional, Sequence

# noinspection PyProtectedMember
from pvlib.pvsystem import PVSystem, _unwrap_single_value, pvwatts_losses, scale_voltage_current_power

import pandas as pd
from lories.components import Component, ComponentError, register_component_type
from lories.core import ConfigurationError, Constant
from lories.core.typing import Configurations, ContextArgument
from sparcs.components.solar.array import SolarArray
from sparcs.components.solar.db import InverterDatabase
from sparcs.components.solar.model import SolarModel, DEFAULTS as MODEL_DEFAULTS


@register_component_type("pv", "solar")
# noinspection SpellCheckingInspection
class SolarSystem(Component, PVSystem):
    INCLUDES = ["model", "inverter", "arrays", *SolarArray.INCLUDES]

    POWER_AC = Constant(float, "ac_power", "PV Power", "W", context="pv")
    POWER_AC_EXP = Constant(float, "ac_exp_power", "Export PV Power", "W", context="pv")

    POWER_DC = Constant(float, "dc_power", "PV (DC) Power", "W", context="pv")
    POWER_DC_FRONT = Constant(float, "dc_power_front", "PV (DC) Back Side Power", "W", context="pv")
    POWER_DC_BACK = Constant(float, "dc_power_back", "PV (DC) Front Side Power", "W", context="pv")

    CURRENT_SC = Constant(float, "dc_current_sc", "Short Circuit Current", "A", context="pv")
    CURRENT_MP = Constant(float, "dc_current_mp", "Maximum Power Point Current", context="pv")

    VOLTAGE_OC = Constant(float, "dc_voltage_oc", "Open Circuit Voltage", "V", context="pv")
    VOLTAGE_MP = Constant(float, "dc_voltage_mp", "Maximum Power Point Voltage", "V", context="pv")

    ENERGY_AC = Constant(float, "ac_energy", "PV Energy", "kWh", context="pv")
    ENERGY_DC = Constant(float, "dc_energy", "PV (DC) Energy", "kWh", context="pv")
    ENERGY_DC_FRONT = Constant(float, "dc_energy_front", "PV (DC) Back Side Energy", "W", context="pv")
    ENERGY_DC_BACK = Constant(float, "dc_energy_back", "PV (DC) Front Side Energy", "W", context="pv")

    TEMPERATURE_CELL = Constant(float, "temp_cell", "Cell Temperature", "°C", context="pv")

    YIELD_SPECIFIC = Constant(float, "yield_specific", "Specific Yield", "kWh/kWp", context="pv")
    YIELD_ENERGY = Constant(float, "yield_energy", "Energy Yield", "kWh", context="pv")
    YIELD_ENERGY_DC = Constant(float, "yield_energy_dc", "Energy Yield (DC)", "kWh", context="pv")
    BIFACIAL_GAIN = Constant(float, "bifacial_gain", "Bifacial Gain", "%", context="pv")

    arrays: Sequence[SolarArray]

    inverter: str = None
    inverter_parameters: dict = {}
    inverters_per_system: int = 1

    modules_per_inverter: int = SolarArray.modules_per_string * SolarArray.strings

    power_max: float = 0

    losses_parameters: dict = {}

    def __init__(self, context: ContextArgument, configs: Configurations, **kwargs) -> None:
        super().__init__(context=context, configs=configs, **kwargs)
        self.get_ac = self.run_ac_model

    def __repr__(self) -> str:
        return Component.__repr__(self)

    def __str__(self) -> str:
        return Component.__str__(self)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        if name is None:
            return
        self._name = name

    def has_rows(self) -> bool:
        return any(array.has_rows() for array in self.arrays)

    def _load_arrays(self, configs: Configurations) -> None:
        arrays = []
        for array in self.components.load_from_type(
            SolarArray,
            configs,
            "arrays",
            key="array",
            name=f"{self.name} Array",
            includes=SolarArray.INCLUDES,
        ):
            model_defaults = {"transposition_model":MODEL_DEFAULTS["transposition_model"]}
            model = configs.get_member("model", defaults=model_defaults)
            transposition_model = re.sub("[^A-Za-z0-9]+", "", model.get("transposition_model")).lower()
            if array.has_rows() and transposition_model in ["viewfactor", "solarfactor", "pvfactor", "raytracing"]:
                rows = array.configs.get_member("rows")
                row_model = array.transposition_model_parameters
                row_modules = round(array.rows.modules / array.rows.count)
                row_arrays = []
                for i in range(min(row_model["rows"], rows["count"])):
                    _array_configs = array.configs.copy()
                    _array_configs["key"] = f"{array.key}_{i}"
                    _array_configs["name"] = f"{array.name} Row {i+1}"

                    _array_rows = _array_configs.get_member("rows")
                    _array_rows.update(rows.get_member(str(i), defaults={"modules": row_modules}))
                    _array_rows["count"] = 1

                    _array_row_model = _array_configs.get_member("transposition")
                    _array_row_model["row_index"] = i + 1
                    row_arrays.append(_array_configs)

                center_array_row = row_arrays[floor(row_model["rows"] / 2)].get_member("rows")
                center_array_row["count"] += array.rows.count - row_model["rows"]
                center_array_row["modules"] += array.rows.modules - sum(a["rows"]["modules"] for a in row_arrays)
                arrays.extend(array.duplicate(configs=c) for c in row_arrays)
            else:
                arrays.append(array)
        self.arrays = tuple(arrays)

    # noinspection PyProtectedMembers
    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self._load_arrays(configs)

        def add_channel(constant: Constant, **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = constant.name.replace("PV", self.name, 1)
            channel["column"] = constant.id.replace("pv", self.key, 1)
            channel["aggregate"] = "mean"
            channel.update(custom)
            self.data.add(**channel)

        add_channel(SolarSystem.POWER_AC)
        add_channel(SolarSystem.POWER_DC)
        if any(a.is_bifacial() for a in self.arrays):
            add_channel(SolarSystem.POWER_DC_FRONT)
            add_channel(SolarSystem.POWER_DC_BACK)
        add_channel(SolarSystem.CURRENT_SC)
        add_channel(SolarSystem.CURRENT_MP)
        add_channel(SolarSystem.VOLTAGE_OC)
        add_channel(SolarSystem.VOLTAGE_MP)
        add_channel(SolarSystem.TEMPERATURE_CELL)

    def _on_configure(self, configs: Configurations) -> None:
        super()._on_configure(configs)
        self.losses_parameters = configs.get("losses", default=SolarSystem.losses_parameters)
        try:
            # The converter needs to be configured, after all solar arrays were configured
            inverter = configs.get_member("inverter", defaults={})
            self.inverter = inverter.get("model", default=SolarSystem.inverter)
            self.inverter_parameters = self._infer_inverter_params()
            self.inverter_parameters = self._fit_inverter_params()
            self.inverters_per_system = inverter.get_int("count", default=SolarSystem.modules_per_inverter)

            self.modules_per_inverter = sum([array.modules_per_string * array.strings for array in self.arrays])

            if all(["pdc0" in a.module_parameters for a in self.arrays]):
                self.power_max = round(
                    sum(
                        array.modules_per_string * array.strings * array.module_parameters["pdc0"]
                        for array in self.arrays
                    )
                )
        except ConfigurationError as e:
            self._logger.warning(f"Unable to configure inverter for system '{self.key}': ", e)

    def _infer_inverter_params(self) -> dict:
        params = {}
        self._inverter_parameters_override = False
        if not self._read_inverter_params(params):
            self._read_inverter_database(params)

        inverter_params_exist = len(params) > 0
        if self._read_inverter_configs(params) and inverter_params_exist:
            self._inverter_parameters_override = True

        return params

    def _fit_inverter_params(self) -> dict:
        params = self.inverter_parameters

        if "pdc0" not in params and all(["pdc0" in a.module_parameters for a in self.arrays]):
            params["pdc0"] = round(
                sum(
                    [
                        array.modules_per_string * array.strings * array.module_parameters["pdc0"]
                        for array in self.arrays
                    ]
                )
            )

        if "eta_inv_nom" not in params and "Efficiency" in params:
            if params["Efficiency"] > 1:
                params["Efficiency"] /= 100.0
                self._logger.debug(
                    "Inverter efficiency configured in percent and will be adjusted: ", params["Efficiency"] * 100.0
                )
            params["eta_inv_nom"] = params["Efficiency"]

        return params

    def _read_inverter_params(self, params: dict) -> bool:
        if self.configs.has_member("Inverter"):
            module_params = dict({k: v for k, v in self.configs["Inverter"].items() if k not in ["count", "model"]})
            if len(module_params) > 0:
                _update_parameters(params, module_params)
                self._logger.debug("Extract inverter from config file")
                return True
        return False

    def _read_inverter_database(self, params: dict) -> bool:
        if self.inverter is not None:
            try:
                inverters = InverterDatabase(self.configs)
                inverter_params = inverters.read(self.inverter)
                _update_parameters(params, inverter_params)
            except IOError as e:
                self._logger.warning(f"Error reading inverter '{self.inverter}' from database: ", str(e))
                return False
            self._logger.debug(f"Read inverter '{self.inverter}' from database")
            return True
        return False

    # noinspection PyUnresolvedReferences
    def _read_inverter_configs(self, params: dict) -> bool:
        inverter_file = os.path.join(self.configs.dirs.conf, f"{self.key}.d", "inverter.conf")
        if os.path.exists(inverter_file):
            with open(inverter_file) as f:
                inverter_str = "[Inverter]\n" + f.read()

            from configparser import ConfigParser

            inverter_configs = ConfigParser()
            inverter_configs.optionxform = str
            inverter_configs.read_string(inverter_str)
            inverter_params = dict(inverter_configs["Inverter"])
            _update_parameters(params, inverter_params)
            self._logger.debug("Read inverter file: %s", inverter_file)
            return True
        return False

    # noinspection SpellCheckingInspection, PyMethodOverriding
    @_unwrap_single_value
    def pvwatts_losses(self, solar_position: pd.DataFrame):
        # noinspection SpellCheckingInspection
        def _pvwatts_losses(array: SolarArray):
            return pvwatts_losses(**array.pvwatts_losses(solar_position))

        if self.num_arrays > 1:
            return tuple(_pvwatts_losses(array) for array in self.arrays)
        else:
            return _pvwatts_losses(self.arrays[0])

    # TODO: Remove solar_elevation as soon as
    @_unwrap_single_value
    def run_irradiance_model(
        self,
        solar_zenith,
        solar_azimuth,
        solar_elevation,
        dni,
        ghi,
        dhi,
        dni_extra=None,
        airmass=None,
        albedo=None,
        model="viewfactor",
        **kwargs,
    ):
        """
        Uses :py:func:`pvlib.irradiance.get_total_irradiance` to
        calculate the plane of array irradiance components on the tilted
        surfaces defined by each array's ``surface_tilt`` and
        ``surface_azimuth``.

        Parameters
        ----------
        solar_zenith : float or Series
            Solar zenith angle.
        solar_azimuth : float or Series
            Solar azimuth angle.
        solar_elevation : float or Series.
            Solar elevation angle.
        dni : float, Series, or tuple of float or Series
            Direct Normal Irradiance. [W/m2]
        ghi : float, Series, or tuple of float or Series
            Global horizontal irradiance. [W/m2]
        dhi : float, Series, or tuple of float or Series
            Diffuse horizontal irradiance. [W/m2]
        dni_extra : float, Series or tuple of float or Series, optional
            Extraterrestrial direct normal irradiance. [W/m2]
        airmass : float or Series, optional
            Airmass. [unitless]
        albedo : float or Series, optional
            Ground surface albedo. [unitless]
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to
            :py:func:`pvlib.irradiance.get_total_irradiance`.

        Notes
        -----
        Each of ``dni``, ``ghi``, and ``dni`` may be passed as a float, Series,
        or tuple of float or Series. If passed as a float or Series, these
        values are used for all Arrays. If passed as a tuple, the tuple length
        must be the same as the number of Arrays. The first tuple element is
        used for the first Array, the second tuple element for the second
        Array, and so forth.

        Some sky irradiance models require ``dni_extra``. For these models,
        if ``dni_extra`` is not provided and ``solar_zenith`` has a
        ``DatetimeIndex``, then ``dni_extra`` is calculated.
        Otherwise, ``dni_extra=1367`` is assumed.

        Returns
        -------
        poa_irradiance : DataFrame or tuple of DataFrame
            Column names are: ``'poa_global', 'poa_direct', 'poa_diffuse',
            'poa_sky_diffuse', 'poa_ground_diffuse'``.

        See also
        --------
        pvlib.irradiance.get_total_irradiance
        """
        dni = self._validate_per_array(dni, system_wide=True)
        ghi = self._validate_per_array(ghi, system_wide=True)
        dhi = self._validate_per_array(dhi, system_wide=True)

        albedo = self._validate_per_array(albedo, system_wide=True)

        return tuple(
            array.run_irradiance_model(
                solar_zenith,
                solar_azimuth,
                solar_elevation,
                dni,
                ghi,
                dhi,
                dni_extra=dni_extra,
                airmass=airmass,
                albedo=albedo,
                model=model,
                **kwargs,
            )
            for array, dni, ghi, dhi, albedo in zip(self.arrays, dni, ghi, dhi, albedo)
        )

    @_unwrap_single_value
    def run_cell_temperature_model(
        self,
        poa_front,
        poa_back,
        temp_air,
        wind_speed,
        model,
        effective_irradiance=None,
    ):
        """
        Determine cell temperature using the method specified by ``model``.

        Parameters
        ----------
        poa_front : numeric or tuple of numeric
            Total front side irradiance in W/m^2.

        poa_back : numeric or tuple of numeric
            Total back side irradiance in W/m^2.

        temp_air : numeric or tuple of numeric
            Ambient dry bulb temperature in degrees C.

        wind_speed : numeric or tuple of numeric
            Wind speed in m/s.

        model : str
            Supported models include ``'sapm'``, ``'pvsyst'``,
            ``'faiman'``, ``'fuentes'``, and ``'noct_sam'``

        effective_irradiance : numeric or tuple of numeric, optional
            The irradiance that is converted to photocurrent in W/m^2.
            Only used for some models.

        Returns
        -------
        numeric or tuple of numeric
            Values in degrees C.

        See Also
        --------
        Array.get_cell_temperature

        Notes
        -----
        The `temp_air` and `wind_speed` parameters may be passed as tuples
        to provide different values for each Array in the system. If passed as
        a tuple the length must be the same as the number of Arrays. If not
        passed as a tuple then the same value is used for each Array.
        """
        poa_front = self._validate_per_array(poa_front)
        poa_back = self._validate_per_array(poa_back)
        temp_air = self._validate_per_array(temp_air, system_wide=True)
        wind_speed = self._validate_per_array(wind_speed, system_wide=True)
        # Not used for all models, but Array.get_cell_temperature handles it
        effective_irradiance = self._validate_per_array(effective_irradiance, system_wide=True)

        return tuple(
            array.run_cell_temperature_model(
                poa_front,
                poa_back,
                temp_air,
                wind_speed,
                model,
                effective_irradiance,
            )
            for array, poa_front, poa_back, temp_air, wind_speed, effective_irradiance in zip(
                self.arrays,
                poa_front,
                poa_back,
                temp_air,
                wind_speed,
                effective_irradiance,
            )
        )

    # noinspection PyArgumentList
    @_unwrap_single_value
    def run_singlediode_model(self, irradiance, cell_temperature, params_function):
        def _make_diode_params(
            photocurrent,
            saturation_current,
            resistance_series,
            resistance_shunt,
            nNsVth,
        ):
            return pd.DataFrame(
                {
                    "I_L": photocurrent,
                    "I_o": saturation_current,
                    "R_s": resistance_series,
                    "R_sh": resistance_shunt,
                    "nNsVth": nNsVth,
                }
            )

        params = params_function(irradiance, cell_temperature, unwrap=False)
        params_diode = tuple(itertools.starmap(_make_diode_params, params))

        dc = tuple(itertools.starmap(self.singlediode, params))
        dc = tuple(d.where(d > 1e-3, other=0) for d in dc)
        dc = self.scale_voltage_current_power(dc, unwrap=False)
        return dc, params_diode

    def run_ac_model(self, model, p_dc, v_dc=None):
        r"""Calculates AC power from p_dc using the inverter model indicated
        by model and self.inverter_parameters.

        Parameters
        ----------
        model : str
            Must be one of 'sandia', 'adr', or 'pvwatts'.
        p_dc : numeric, or tuple, list or array of numeric
            DC power on each MPPT input of the inverter. Use tuple, list or
            array for inverters with multiple MPPT inputs. If type is array,
            p_dc must be 2d with axis 0 being the MPPT inputs. [W]
        v_dc : numeric, or tuple, list or array of numeric
            DC voltage on each MPPT input of the inverter. Required when
            model='sandia' or model='adr'. Use tuple, list or
            array for inverters with multiple MPPT inputs. If type is array,
            v_dc must be 2d with axis 0 being the MPPT inputs. [V]

        Returns
        -------
        p_ac : numeric
            AC power output for the inverter. [W]

        Raises
        ------
        ValueError
            If model is not one of 'sandia', 'adr' or 'pvwatts'.
        ValueError
            If model='adr' and the PVSystem has more than one array.

        See also
        --------
        pvlib.inverter.sandia
        pvlib.inverter.sandia_multi
        pvlib.inverter.adr
        pvlib.inverter.pvwatts
        pvlib.inverter.pvwatts_multi
        """
        return scale_voltage_current_power(super().get_ac(model, p_dc, v_dc=v_dc), current=self.inverters_per_system)

    def run(self, weather: pd.DataFrame) -> pd.DataFrame:
        if len(self.arrays) < 1:
            raise ComponentError(self, "PV system must have at least one Array.")
        if not all(a.is_parametrized() for a in self.arrays):
            raise ComponentError(
                self,
                "PV array configurations of this system are not valid: ",
                ", ".join(a.name for a in self.arrays if not a.is_configured()),
            )
        columns = OrderedDict()
        columns[SolarArray.POWER_AC] = self.data[SolarSystem.POWER_AC].id
        columns[SolarArray.POWER_DC] = self.data[SolarSystem.POWER_DC].id
        if any(a.is_bifacial() for a in self.arrays):
            columns[SolarArray.POWER_DC_FRONT] = self.data[SolarSystem.POWER_DC_FRONT].id
            columns[SolarArray.POWER_DC_BACK] = self.data[SolarSystem.POWER_DC_BACK].id
        columns[SolarArray.CURRENT_SC] = self.data[SolarSystem.CURRENT_SC].id
        columns[SolarArray.VOLTAGE_OC] = self.data[SolarSystem.VOLTAGE_OC].id
        columns[SolarArray.CURRENT_MP] = self.data[SolarSystem.CURRENT_MP].id
        columns[SolarArray.VOLTAGE_MP] = self.data[SolarSystem.VOLTAGE_MP].id
        columns[SolarArray.TEMPERATURE_CELL] = self.data[SolarSystem.TEMPERATURE_CELL].id

        model = SolarModel.load(self)
        results = model(weather).loc[:, list(columns.keys())]
        return results.rename(columns=columns)


def _update_parameters(parameters: Dict, update: Mapping):
    for key, value in update.items():
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value

    return parameters
