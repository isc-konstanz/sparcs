# -*- coding: utf-8 -*-
"""
sparcs.devices.charge_big
~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import numpy as np
import pandas as pd
from lories.components import register_component_type
from lories.core import ConfigurationError, Constant
from lories.typing import Configurations, ContextArgument
from sparcs.components import Meter
from sparcs.components.vehicle import EVSE


@register_component_type("charge_big")
class ChargeBig(Meter):
    SETPOINT = Constant(float, "setpoint", "Setpoint Current", "A")
    SETPOINT_MAX = Constant(float, "setpoint_max", "Setpoint Current Maximum", "A")
    SETPOINT_POWER = Constant(float, "setpoint_power", "Setpoint Power", "W")

    CURRENT = Constant(float, "current", "Charging Current", "A")

    L1_COS_PHI = Constant(float, "l1_cos_phi", "L1 Cos Phi", "")
    L2_COS_PHI = Constant(float, "l2_cos_phi", "L2 Cos Phi", "")
    L3_COS_PHI = Constant(float, "l3_cos_phi", "L3 Cos Phi", "")

    _stations: int

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        connector = configs.get("connector")
        if connector is None:
            raise ConfigurationError(f"Missing 'connector' for {type(self).__name__} '{self.id}'")

        def add_channel(constant: Constant, address: str, aggregate: str = "mean", **custom) -> None:
            channel = constant.to_dict()
            channel["connector"] = connector
            channel["address"] = address
            channel["aggregate"] = aggregate
            channel.update(custom)
            self.data.add(**channel)

        def add_virtual_channel(constant: Constant, aggregate: str = "mean", **custom) -> None:
            self.data.add(key=constant, aggregate=aggregate, connector=None, **custom)

        add_channel(ChargeBig.SETPOINT, "Sollwert_aktiv")

        add_channel(Meter.POWER_L1_ACTIVE, "Zähler_Leistung_Phase1")
        add_channel(Meter.POWER_L2_ACTIVE, "Zähler_Leistung_Phase2")
        add_channel(Meter.POWER_L3_ACTIVE, "Zähler_Leistung_Phase3")

        add_channel(Meter.ENERGY_L1_ACTIVE, "Zähler_Energiebezug_Phase1", aggregate="last")
        add_channel(Meter.ENERGY_L2_ACTIVE, "Zähler_Energiebezug_Phase2", aggregate="last")
        add_channel(Meter.ENERGY_L3_ACTIVE, "Zähler_Energiebezug_Phase3", aggregate="last")

        add_channel(Meter.CURRENT_L1, "Zähler_Strom_Phase1")
        add_channel(Meter.CURRENT_L2, "Zähler_Strom_Phase2")
        add_channel(Meter.CURRENT_L3, "Zähler_Strom_Phase3")

        add_channel(ChargeBig.L1_COS_PHI, "Zähler_CosPhi_Phase1")
        add_channel(ChargeBig.L2_COS_PHI, "Zähler_CosPhi_Phase2")
        add_channel(ChargeBig.L3_COS_PHI, "Zähler_CosPhi_Phase3")

        add_virtual_channel(ChargeBig.CURRENT)
        add_virtual_channel(Meter.POWER_ACTIVE)
        add_virtual_channel(Meter.POWER_REACTIVE)
        add_virtual_channel(ChargeBig.SETPOINT_POWER)
        add_virtual_channel(ChargeBig.SETPOINT_MAX)

        defaults = ChargeBigStation._build_defaults(configs, strict=True)
        stations = configs.get_member("stations", defaults=defaults)
        mapping = stations.get("mapping", default={})
        for station_id in range(stations.get_int("count")):
            mapped_id = mapping[str(station_id + 1)] - 1 if str(station_id + 1) in mapping.keys() else station_id

            station_defaults = ChargeBigStation._build_defaults(stations, strict=True)
            station_configs = stations.get_member(f"station_{station_id}", defaults=station_defaults)
            station = ChargeBigStation(
                key=f"station_{station_id}",
                name=f"{self.name} Station {station_id}",
                context=self,
                configs=station_configs,
                station_index=station_id,
                station_id=mapped_id,
                connector=connector,
            )

            self.components.add(station)

    def activate(self) -> None:
        super().activate()
        self.data.register(
            self._on_power_received,
            [self.data[Meter.POWER_L1_ACTIVE], self.data[Meter.POWER_L2_ACTIVE], self.data[Meter.POWER_L3_ACTIVE]],
            how="any",
            unique=False,
        )
        self.data.register(
            self._on_current_received,
            [self.data[Meter.CURRENT_L1], self.data[Meter.CURRENT_L2], self.data[Meter.CURRENT_L3]],
            how="any",
            unique=False,
        )
        self.data.register(
            self._on_reactive_power_received,
            [
                self.data[Meter.POWER_L1_ACTIVE],
                self.data[Meter.POWER_L2_ACTIVE],
                self.data[Meter.POWER_L3_ACTIVE],
                self.data[ChargeBig.L1_COS_PHI],
                self.data[ChargeBig.L2_COS_PHI],
                self.data[ChargeBig.L3_COS_PHI],
            ],
            how="any",
            unique=False,
        )

    def _on_power_received(self, data: pd.DataFrame) -> None:
        power = data.loc[:, Meter.POWER_L1_ACTIVE].dropna()
        power = power + data.loc[power.index, Meter.POWER_L2_ACTIVE].fillna(0)
        power = power + data.loc[power.index, Meter.POWER_L3_ACTIVE].fillna(0)
        if not power.empty:
            self.data[Meter.POWER_ACTIVE].set(power.index[0], power)

    def _on_current_received(self, data: pd.DataFrame) -> None:
        current = data.loc[:, Meter.CURRENT_L1].dropna()
        current = current + data.loc[current.index, Meter.CURRENT_L2].fillna(0)
        current = current + data.loc[current.index, Meter.CURRENT_L3].fillna(0)
        if not current.empty:
            self.data[ChargeBig.CURRENT].set(current.index[0], current)

    def _on_reactive_power_received(self, data: pd.DataFrame) -> None:
        def phase_reactive(power_col, cos_phi_col) -> pd.Series:
            power = data.loc[:, power_col]
            cos_phi = data.loc[:, cos_phi_col]
            return power * np.sqrt(1 - cos_phi**2) / cos_phi

        reactive = phase_reactive(Meter.POWER_L1_ACTIVE, ChargeBig.L1_COS_PHI).dropna()
        reactive = reactive + phase_reactive(Meter.POWER_L2_ACTIVE, ChargeBig.L2_COS_PHI).reindex(
            reactive.index
        ).fillna(0)
        reactive = reactive + phase_reactive(Meter.POWER_L3_ACTIVE, ChargeBig.L3_COS_PHI).reindex(
            reactive.index
        ).fillna(0)
        if not reactive.empty:
            self.data[Meter.POWER_REACTIVE].set(reactive.index[0], reactive)


class ChargeBigStation(EVSE):
    STATE = Constant(float, "state", "State", alias="chargebig_state")
    LIMIT = Constant(float, "limit", "Current Limit", "A", alias="chargebig_limit")

    station_id: int
    connector: str

    def __init__(
        self,
        context: ContextArgument,
        configs: Configurations,
        station_index: int,  # table column
        station_id: int,  # physical id
        connector: str,
        **kwargs,
    ) -> None:
        super().__init__(context=context, configs=configs, **kwargs)
        self.station_index = station_index
        self.station_id = station_id
        self.connector = connector

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        def add_channel(constant: Constant, address: str, aggregate: str = "mean", **custom) -> None:
            channel = constant.to_dict()
            channel["connector"] = self.connector
            channel["station_id"] = self.station_index
            channel["address"] = address
            channel["aggregate"] = aggregate
            channel.update(custom)
            self.data.add(**channel)

        add_channel(ChargeBigStation.STATE, f"Ladepunkt_{self.station_id}_Status")
        add_channel(ChargeBigStation.LIMIT, f"Ladepunkt_{self.station_id}_Grenzwert")
