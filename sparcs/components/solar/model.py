# -*- coding: utf-8 -*-
"""
sparcs.components.solar.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

# noinspection PyProtectedMember
from pvlib.modelchain import ModelChain, _to_tuple, _tuple_from_dfs

import pandas as pd
from lories import Configurations, Configurator
from lories.typing import Location

# noinspection SpellCheckingInspection
DEFAULTS = dict(
    # ac_model='pvwatts',
    # dc_model="desoto",
    # temperature_model="pvsyst",
    aoi_model="physical",
    spectral_model="no_loss",
    dc_ohmic_model="no_loss",
    losses_model="pvwatts",
)


# noinspection SpellCheckingInspection, PyAbstractClass
class SolarModel(Configurator, ModelChain):
    TYPE: str = "model"

    # noinspection PyUnresolvedReferences
    @classmethod
    def load(cls, pvsystem, include_file: str = "model.conf") -> SolarModel:
        include_dir = pvsystem.configs.path.replace(".conf", ".d")
        configs_dirs = pvsystem.configs.dirs.to_dict()
        configs_dirs["conf_dir"] = include_dir

        configs = Configurations.load(
            include_file,
            **configs_dirs,
            **pvsystem.configs,
            require=False,
        )
        params = DEFAULTS
        if cls.TYPE in configs:
            params.update(configs.get_member(cls.TYPE))

        return cls(configs, pvsystem, pvsystem.context.location, **params)

    def __init__(self, configs: Configurations, pvsystem, location: Location, **kwargs):
        super().__init__(configs=configs, system=pvsystem, location=location, **kwargs)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name

    def __call__(self, weather, **_):
        self.run_model(weather)

        def _agg_dc_power(data, name):
            if isinstance(data, tuple):
                data = pd.concat([dc[["p_mp"]] for dc in data], axis="columns").sum(axis="columns")
            else:
                data = data["p_mp"]
            data.name = name
            return data

        results = []
        results_ac = self.results.ac.to_frame() if isinstance(self.results.ac, pd.Series) else self.results.ac
        results_ac = results_ac["p_mp"]
        results_ac.name = "p_ac"
        results.append(results_ac)

        results_dc = _agg_dc_dfs(*self.results.dc) if isinstance(self.results.dc, tuple) else self.results.dc
        results_dc.rename(columns={"p_mp": "p_dc"}, inplace=True)
        results.append(results_dc)

        if any(a.is_bifacial() for a in self.system.arrays):
            results.append(_agg_dc_power(self.results.dc_front, "p_dc_f"))
            results.append(_agg_dc_power(self.results.dc_back, "p_dc_b"))

        cell_temperature = self.results.cell_temperature
        if isinstance(cell_temperature, tuple):
            cell_temperature = pd.concat(cell_temperature, axis="columns").mean(axis="columns")
        cell_temperature.name = "t_cell"
        results.append(cell_temperature)

        losses = self.results.losses
        if isinstance(losses, pd.Series) or (
            isinstance(losses, tuple) and all([isinstance(loss, pd.Series) for loss in losses])
        ):
            if isinstance(losses, tuple):
                losses = pd.concat(losses, axis="columns").mean(axis="columns")
            losses.name = "losses"
            results.append(losses)

        return pd.concat(results, axis="columns")

    def _set_celltemp(self, model):
        """
        Set self.results.cell_temperature using the given cell temperature model.

        Parameters
        ----------
        model : str
            A cell temperature model name to pass to
            :py:meth:`pvlib.pvsystem.PVSystem.get_cell_temperature`.
            Valid names are 'sapm', 'pvsyst', 'faiman', 'fuentes', 'noct_sam'

        Returns
        -------
        self
        """
        if self.system.has_rows():
            poa_front = _tuple_from_dfs(self.results.total_irrad, "poa_front")
            poa_back = _tuple_from_dfs(self.results.total_irrad, "poa_back")
            temp_air = _tuple_from_dfs(self.results.weather, "temp_air")
            wind_speed = _tuple_from_dfs(self.results.weather, "wind_speed")
            kwargs = {}
            if model == "noct_sam":
                kwargs["effective_irradiance"] = self.results.effective_irradiance

            self.results.cell_temperature = self.system.run_cell_temperature_model(
                poa_front, poa_back, temp_air, wind_speed, model=model, **kwargs
            )
            return self
        return super()._set_celltemp(model)

    def _singlediode(self, params_function):
        if self.system.has_rows():
            cell_temperature = self.results.cell_temperature

            def _run_singlediode_model(irrad):
                return self.system.run_singlediode_model(irrad, cell_temperature, params_function)

            is_bifacial = any(a.is_bifacial() for a in self.system.arrays)
            if is_bifacial:

                def _agg_bifacial_power(dc_front, dc_back, losses_parameters):
                    dc_back["p_mp"] *= (1 - losses_parameters.get("mismatch_bifaciality", 7) / 100)
                    return _agg_dc_dfs(dc_front,  dc_back)

                def _prepare_bifacial_poa_back(irrad, module_parameters):
                    return irrad * module_parameters["module_bifaciality"]

                poa_front = _tuple_from_dfs(self.results.total_irrad, "poa_front")
                poa_back = _tuple_from_dfs(self.results.total_irrad, "poa_back")
                if len(self.system.arrays) > 1:
                    poa_back = tuple(
                        _prepare_bifacial_poa_back(_poa_back, array.module_parameters)
                        for _poa_back, array in zip(poa_back, self.system.arrays)
                    )
                else:
                    poa_back = _prepare_bifacial_poa_back(poa_back, self.system.arrays[0].module_parameters)

                self.results.dc_front, self.results.diode_params_front = _run_singlediode_model(poa_front)
                self.results.dc_back, self.results.diode_params_back = _run_singlediode_model(poa_back)
                if len(self.system.arrays) > 1:
                    self.results.dc = tuple(
                        _agg_bifacial_power(_dc_front, _dc_back, array.array_losses_parameters)
                        for _dc_front, _dc_back, array in zip(
                            self.results.dc_front,
                            self.results.dc_back,
                            self.system.arrays,
                        )
                    )
                else:
                    self.results.dc = _agg_bifacial_power(
                        self.results.dc_front[0],
                        self.results.dc_back[0],
                        self.system.arrays[0].array_losses_parameters,
                    )
            else:
                self.results.dc, self.results.diode_params = _run_singlediode_model(self.results.effective_irradiance)

            # If the system has one Array, unwrap the single return value
            # to preserve the original behavior of ModelChain
            if self.system.num_arrays == 1:
                self.results.dc = self.results.dc[0]
                if is_bifacial:
                    self.results.dc_front = self.results.dc_front[0]
                    self.results.dc_back = self.results.dc_back[0]
                    self.results.diode_params_front = self.results.diode_params_front[0]
                    self.results.diode_params_back = self.results.diode_params_back[0]
                else:
                    self.results.diode_params = self.results.diode_params[0]
            return self
        return super()._singlediode(params_function)

    def effective_irradiance_model(self):
        if self.system.has_rows():

            def _eff_irradiance(module_parameters, total_irrad):
                b = module_parameters.get("module_bifaciality", 0.0)
                return total_irrad["poa_front"] + b * total_irrad["poa_back"]

            if isinstance(self.results.total_irrad, tuple):
                self.results.effective_irradiance = tuple(
                    _eff_irradiance(array.module_parameters, irrad)
                    for array, irrad in zip(
                        self.system.arrays,
                        self.results.total_irrad,
                    )
                )
            else:
                self.results.effective_irradiance = _eff_irradiance(
                    self.system.arrays[0].module_parameters,
                    self.results.total_irrad,
                )
            return self
        return super().effective_irradiance_model()

    def prepare_inputs(self, weather):
        """
        Prepare the solar position, irradiance, and weather inputs to
        the model, starting with GHI, DNI and DHI.

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrames
            Required column names include ``'dni'``, ``'ghi'``, ``'dhi'``.
            Optional column names are ``'wind_speed'``, ``'temp_air'``,
            ``'albedo'``.

            If optional columns ``'wind_speed'``, ``'temp_air'`` are not
            provided, air temperature of 20 C and wind speed
            of 0 m/s will be added to the ``weather`` DataFrame.

            If optional column ``'albedo'`` is provided, albedo values in the
            ModelChain's PVSystem.arrays are ignored.

            If `weather` is a tuple or list, it must be of the same length and
            order as the Arrays of the ModelChain's PVSystem.

        Raises
        ------
        ValueError
            If any `weather` DataFrame(s) is missing an irradiance component.
        ValueError
            If `weather` is a tuple or list and the DataFrames it contains have
            different indices.
        ValueError
            If `weather` is a tuple or list with a different length than the
            number of Arrays in the system.

        Notes
        -----
        Assigns attributes to ``results``: ``times``, ``weather``,
        ``solar_position``, ``airmass``, ``total_irrad``, ``aoi``, ``albedo``.

        See also
        --------
        ModelChain.complete_irradiance
        """
        if self.system.has_rows():
            weather = _to_tuple(weather)
            self._check_multiple_input(weather, strict=False)
            self._verify_df(weather, required=["ghi", "dni", "dhi"])
            self._assign_weather(weather)

            self._prep_inputs_solar_pos(weather)
            self._prep_inputs_airmass()
            self._prep_inputs_albedo(weather)
            self._prep_inputs_fixed()

            self.results.total_irrad = self.system.run_irradiance_model(
                self.results.solar_position["apparent_zenith"],
                self.results.solar_position["azimuth"],
                self.results.solar_position["apparent_elevation"],
                _tuple_from_dfs(self.results.weather, "dni"),
                _tuple_from_dfs(self.results.weather, "ghi"),
                _tuple_from_dfs(self.results.weather, "dhi"),
                albedo=self.results.albedo,
                airmass=self.results.airmass["airmass_relative"],
                model=self.transposition_model,
            )
            return self
        return super().prepare_inputs(weather)

    def pvwatts_losses(self):
        losses = self.system.pvwatts_losses(self.results.solar_position)

        if isinstance(self.results.dc, tuple):
            self.results.losses = tuple((100 - l) / 100.0 for l in losses)
            for dc, losses in zip(self.results.dc, losses):
                dc[:] = dc.mul(losses, axis="index")
            if any(a.is_bifacial() for a in self.system.arrays):
                for dc_front, dc_back, losses in zip(self.results.dc_front, self.results.dc_back, losses):
                    dc_front[:] = dc_front.mul(losses, axis="index")
                    dc_back[:] = dc_back.mul(losses, axis="index")
        else:
            self.results.losses = (100 - self.system.pvwatts_losses()) / 100.0
            self.results.dc *= self.results.losses
            if any(a.is_bifacial() for a in self.system.arrays):
                self.results.dc_front *= self.results.losses
                self.results.dc_back *= self.results.losses
        return self


# noinspection PyUnresolvedReferences
def _agg_dc_dfs(*dfs, voltage="mean", current="sum"):
    voltage_keys = ["v_mp", "v_oc"]
    current_keys = ["i_mp", "i_x", "i_xx", "i_sc"]
    power_keys = ["p_mp"]

    def _filter_dfs(keys):
        return pd.concat([_df.filter(keys, axis="columns") for _df in dfs], axis="columns")

    # TODO: This is where mismatch is happening. Look into this further.
    voltage_df = _filter_dfs(voltage_keys).T.groupby(lambda d: d).agg(voltage).T
    current_df = _filter_dfs(current_keys).T.groupby(lambda d: d).agg(current).T
    power_df = _filter_dfs(power_keys).T.groupby(lambda d: d).sum().T
    df = pd.concat([power_df, current_df, voltage_df], axis="columns")
    return df
