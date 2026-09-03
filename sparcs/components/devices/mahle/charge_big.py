# -*- coding: utf-8 -*-
"""
sparcs.components.devices.mahle.charge_big
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from lories.components import register_component_type
from lories.core import Constant
from lories.typing import Configurations, ContextArgument
from sparcs.components.meter import EnergyMeter
from sparcs.components.vehicle import EVSE


@register_component_type("charge_big")
class ChargeBig(EnergyMeter):
    """
    A Mahle chargeBIG charging park read over its OPC UA server: the park-level meter points
    plus one `ChargeBigStation` child per charge point. The node names are fixed by Mahle, so
    the channel set is declared here instead of through the `phases`/`quality` flags; `power`,
    `current` and `reactive_power` are virtual and summed from the phases on arrival.

    The connector is referenced by the id `opcua`: declare `[connectors.opcua]` with the
    server's host on this component or on a parent. `settings = "ns=1"` is the family default
    for a locally declared connector.
    """

    CONNECTOR = "opcua"

    SETPOINT = Constant(float, "setpoint", "Setpoint Current", "A", aggregate="mean")
    SETPOINT_MAX = Constant(float, "setpoint_max", "Setpoint Current Maximum", "A", aggregate="mean")
    SETPOINT_POWER = Constant(float, "setpoint_power", "Setpoint Power", "W", aggregate="mean")

    L1_COS_PHI = Constant(float, "l1_cos_phi", "L1 Cos Phi", "", aggregate="mean")
    L2_COS_PHI = Constant(float, "l2_cos_phi", "L2 Cos Phi", "", aggregate="mean")
    L3_COS_PHI = Constant(float, "l3_cos_phi", "L3 Cos Phi", "", aggregate="mean")

    ENERGY_L1 = Constant(
        float, "l1_active_energy", "Phase 1 Active Energy", "kWh", context="chargebig", aggregate="last"
    )
    ENERGY_L2 = Constant(
        float, "l2_active_energy", "Phase 2 Active Energy", "kWh", context="chargebig", aggregate="last"
    )
    ENERGY_L3 = Constant(
        float, "l3_active_energy", "Phase 3 Active Energy", "kWh", context="chargebig", aggregate="last"
    )

    ADDRESSES: Dict[Constant, str] = {
        SETPOINT: "Sollwert_aktiv",
        EnergyMeter.POWER_L1: "Zähler_Leistung_Phase1",
        EnergyMeter.POWER_L2: "Zähler_Leistung_Phase2",
        EnergyMeter.POWER_L3: "Zähler_Leistung_Phase3",
        ENERGY_L1: "Zähler_Energiebezug_Phase1",
        ENERGY_L2: "Zähler_Energiebezug_Phase2",
        ENERGY_L3: "Zähler_Energiebezug_Phase3",
        EnergyMeter.CURRENT_L1: "Zähler_Strom_Phase1",
        EnergyMeter.CURRENT_L2: "Zähler_Strom_Phase2",
        EnergyMeter.CURRENT_L3: "Zähler_Strom_Phase3",
        L1_COS_PHI: "Zähler_CosPhi_Phase1",
        L2_COS_PHI: "Zähler_CosPhi_Phase2",
        L3_COS_PHI: "Zähler_CosPhi_Phase3",
    }

    VIRTUAL = (
        EnergyMeter.CURRENT,
        EnergyMeter.POWER,
        EnergyMeter.POWER_REACTIVE,
        SETPOINT_POWER,
        SETPOINT_MAX,
    )

    def _connector_defaults(self) -> Dict[str, Any]:
        return {"settings": "ns=1"}

    def _bind(self, constant: Constant) -> Dict[str, Any]:
        address = ChargeBig.ADDRESSES.get(constant)
        if address is None:
            return {}
        return {"address": address, "connector": self._connector_id}

    def _add_channels(self, configs: Configurations) -> None:
        for constant in ChargeBig.ADDRESSES:
            self._add_channel(constant)
        for constant in ChargeBig.VIRTUAL:
            self._add_channel(constant, connector=None)

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        defaults = ChargeBigStation._build_defaults(configs, strict=True)
        stations = configs.get_member("stations", defaults=defaults)
        mapping = stations.get("mapping", default={})
        for station_index in range(stations.get_int("count")):
            mapped = str(station_index + 1)
            station_id = mapping[mapped] - 1 if mapped in mapping.keys() else station_index

            station_defaults = ChargeBigStation._build_defaults(stations, strict=True)
            station_configs = stations.get_member(f"station_{station_index}", defaults=station_defaults)
            station_configs.update({"connector": self._connector_id}, replace=False)
            station = ChargeBigStation(
                key=f"station_{station_index}",
                name=f"{self.name} Station {station_index}",
                context=self,
                configs=station_configs,
                station_index=station_index,
                station_id=station_id,
            )
            self.components.add(station)

    def activate(self) -> None:
        super().activate()
        self.data.register(
            self._on_power_received,
            [self.data[EnergyMeter.POWER_L1], self.data[EnergyMeter.POWER_L2], self.data[EnergyMeter.POWER_L3]],
            how="any",
            unique=False,
        )
        self.data.register(
            self._on_current_received,
            [self.data[EnergyMeter.CURRENT_L1], self.data[EnergyMeter.CURRENT_L2], self.data[EnergyMeter.CURRENT_L3]],
            how="any",
            unique=False,
        )
        self.data.register(
            self._on_reactive_power_received,
            [
                self.data[EnergyMeter.POWER_L1],
                self.data[EnergyMeter.POWER_L2],
                self.data[EnergyMeter.POWER_L3],
                self.data[ChargeBig.L1_COS_PHI],
                self.data[ChargeBig.L2_COS_PHI],
                self.data[ChargeBig.L3_COS_PHI],
            ],
            how="any",
            unique=False,
        )

    def _on_power_received(self, data: pd.DataFrame) -> None:
        power = data.loc[:, EnergyMeter.POWER_L1].dropna()
        power = power + data.loc[power.index, EnergyMeter.POWER_L2].fillna(0)
        power = power + data.loc[power.index, EnergyMeter.POWER_L3].fillna(0)
        if not power.empty:
            self.data[EnergyMeter.POWER].set(power.index[0], power)

    def _on_current_received(self, data: pd.DataFrame) -> None:
        current = data.loc[:, EnergyMeter.CURRENT_L1].dropna()
        current = current + data.loc[current.index, EnergyMeter.CURRENT_L2].fillna(0)
        current = current + data.loc[current.index, EnergyMeter.CURRENT_L3].fillna(0)
        if not current.empty:
            self.data[EnergyMeter.CURRENT].set(current.index[0], current)

    def _on_reactive_power_received(self, data: pd.DataFrame) -> None:
        def phase_reactive(power_col, cos_phi_col) -> pd.Series:
            power = data.loc[:, power_col]
            cos_phi = data.loc[:, cos_phi_col]
            return power * np.sqrt(1 - cos_phi**2) / cos_phi

        reactive = phase_reactive(EnergyMeter.POWER_L1, ChargeBig.L1_COS_PHI).dropna()
        for power_col, cos_phi_col in (
            (EnergyMeter.POWER_L2, ChargeBig.L2_COS_PHI),
            (EnergyMeter.POWER_L3, ChargeBig.L3_COS_PHI),
        ):
            reactive = reactive + phase_reactive(power_col, cos_phi_col).reindex(reactive.index).fillna(0)
        if not reactive.empty:
            self.data[EnergyMeter.POWER_REACTIVE].set(reactive.index[0], reactive)


class ChargeBigStation(EVSE):
    """
    One chargeBIG charge point: its status and current limit, addressed by the physical
    `station_id` on the server while `station_index` names the logged row. The park hands its
    connector id down as the station's default, so a station needs no connector config of its
    own; the id still resolves through the upward lookup.
    """

    CONNECTOR = "opcua"

    STATE = Constant(float, "state", "State", context="chargebig", aggregate="mean")
    LIMIT = Constant(float, "limit", "Current Limit", "A", context="chargebig", aggregate="mean")

    POINTS: Dict[Constant, str] = {
        STATE: "Status",
        LIMIT: "Grenzwert",
    }

    station_index: int
    station_id: int

    def __init__(
        self,
        context: ContextArgument,
        configs: Configurations,
        station_index: int,  # table column
        station_id: int,  # physical id
        **kwargs,
    ) -> None:
        super().__init__(context=context, configs=configs, **kwargs)
        self.station_index = station_index
        self.station_id = station_id

    def _bind(self, constant: Constant) -> Dict[str, Any]:
        point = ChargeBigStation.POINTS.get(constant)
        if point is None:
            return {}
        return {"address": f"Ladepunkt_{self.station_id}_{point}", "connector": self._connector_id}

    def _add_channels(self, configs: Configurations) -> None:
        for constant in ChargeBigStation.POINTS:
            self._add_channel(constant, station_id=self.station_index)
