# -*- coding: utf-8 -*-
"""
sparcs.devices.charge_big
~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lories.components import Component, register_component_type
from lories.connectors.opcua import OpcUaConnector
from lories.core import Constant
from lories.typing import Configurations


@register_component_type("charge_big")
class ChargeBigDevice(Component):
    SETPOINT_ENABLED = Constant(bool, "charge_big_setpoint_enabled", "Charge Big Setpoint Enabled", "")
    SETPOINT = Constant(float, "charge_big_setpoint", "Charge Big Setpoint", "kW")

    L1_POWER = Constant(float, "charge_big_l1_power", "Charge Big L1 Power", "kW")
    L2_POWER = Constant(float, "charge_big_l2_power", "Charge Big L2 Power", "kW")
    L3_POWER = Constant(float, "charge_big_l3_power", "Charge Big L3 Power", "kW")

    L1_ENERGY = Constant(float, "charge_big_l1_energy", "Charge Big L2 Energy", "kWh")
    L2_ENERGY = Constant(float, "charge_big_l2_energy", "Charge Big L2 Energy", "kWh")
    L3_ENERGY = Constant(float, "charge_big_l3_energy", "Charge Big L3 Energy", "kWh")

    L1_CURRENT = Constant(float, "charge_big_l1_current", "Charge Big L1 Current", "A")
    L2_CURRENT = Constant(float, "charge_big_l2_current", "Charge Big L2 Current", "A")
    L3_CURRENT = Constant(float, "charge_big_l3_current", "Charge Big L3 Current", "A")

    L1_COS_PHI = Constant(float, "charge_big_l1_cos_phi", "Charge Big L1 Cos Phi", "")
    L2_COS_PHI = Constant(float, "charge_big_l2_cos_phi", "Charge Big L2 Cos Phi", "")
    L3_COS_PHI = Constant(float, "charge_big_l3_cos_phi", "Charge Big L3 Cos Phi", "")

    POWER = Constant(float, "charge_big_power", "Charge Big Power", "kW")
    ENERGY = Constant(float, "charge_big_energy", "Charge Big Energy", "kWh")
    CURRENT = Constant(float, "charge_big_current", "Charge Big Current", "A")

    STATION_STATE = Constant(int, "charge_big_id_state", "Charge Big Station ID State", "")
    STATION_LIMIT = Constant(int, "charge_big_id_limit", "Charge Big Station ID Limit", "")

    _stations: int

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        connector = OpcUaConnector(
            key=f"{self.key}_connector",
            name=f"{self.name} Connector",
            context=self.context,
            configs=configs.get_member("connector", defaults={})
        )
        self.connectors.add(connector)
        freq = configs.get("freq", default="60s")

        def add_channel(constant: Constant, address: str, station_index: int = None, aggregate: str = "mean", **custom) -> None:
            channel = constant.to_dict()
            channel["name"] = channel["name"].replace("charge_big", self.name, 1)
            channel["key"] = channel["key"].replace("Charge Big", self.key, 1)
            channel["connector"] = connector.id
            channel["address"] = address
            channel["aggregate"] = aggregate
            channel["freq"] = freq  # Todo: why does data.channels.freq doesnt work???
            if station_index is not None:
                channel["name"] = channel["name"].replace("ID", str(station_index + 1), 1)
                channel["key"] = channel["key"].replace("id", str(station_index + 1), 1)
                channel["address"] = channel["address"].replace("id", str(station_index), 1)
            channel.update(custom)
            self.data.add(**channel)

        add_channel(ChargeBigDevice.SETPOINT_ENABLED, "Summen_Grenzwert_aktiv", aggregate="last")
        add_channel(ChargeBigDevice.SETPOINT, "Sollwert_aktiv")

        add_channel(ChargeBigDevice.L1_POWER, "Zähler_Leistung_Phase1")
        add_channel(ChargeBigDevice.L2_POWER, "Zähler_Leistung_Phase2")
        add_channel(ChargeBigDevice.L3_POWER, "Zähler_Leistung_Phase3")

        add_channel(ChargeBigDevice.L1_ENERGY, "Zähler_Energiebezug_Phase1", aggregate="last")
        add_channel(ChargeBigDevice.L2_ENERGY, "Zähler_Energiebezug_Phase2", aggregate="last")
        add_channel(ChargeBigDevice.L3_ENERGY, "Zähler_Energiebezug_Phase3", aggregate="last")

        add_channel(ChargeBigDevice.L1_CURRENT, "Zähler_Strom_Phase1")
        add_channel(ChargeBigDevice.L2_CURRENT, "Zähler_Strom_Phase2")
        add_channel(ChargeBigDevice.L3_CURRENT, "Zähler_Strom_Phase3")

        add_channel(ChargeBigDevice.L1_COS_PHI, "Zähler_CosPhi_Phase1")
        add_channel(ChargeBigDevice.L2_COS_PHI, "Zähler_CosPhi_Phase2")
        add_channel(ChargeBigDevice.L3_COS_PHI, "Zähler_CosPhi_Phase3")

        self._stations = configs.get_int("stations", default=1)

        for station in range(self._stations):
            add_channel(ChargeBigDevice.STATION_STATE, "Ladepunkt_id_Status", station_index=station, aggregate="last")
            add_channel(ChargeBigDevice.STATION_LIMIT, "Ladepunkt_id_Grenzwert", station_index=station, aggregate="last")