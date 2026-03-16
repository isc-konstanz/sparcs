# -*- coding: utf-8 -*-
"""
sparcs.devices.charge_big
~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from lories.components import register_component_type
from lories.connectors.opcua import OpcUaConnector
from lories.core import Constant
from lories.data import Channels
from lories.typing import Configurations, ContextArgument
from sparcs.components import Meter
from sparcs.components.vehicle import EVSE


@register_component_type("charge_big")
class ChargeBig(Meter):
    SETPOINT_ENABLED = Constant(bool, "setpoint_enabled", "Setpoint Enabled", "")
    SETPOINT = Constant(float, "setpoint", "Setpoint", "kW")

    L1_COS_PHI = Constant(float, "l1_cos_phi", "L1 Cos Phi", "")
    L2_COS_PHI = Constant(float, "l2_cos_phi", "L2 Cos Phi", "")
    L3_COS_PHI = Constant(float, "l3_cos_phi", "L3 Cos Phi", "")

    _stations: int

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        connector_configs = configs.get_member(
            "connector",
            defaults={
                "settings": "ns=1",
            }
        )
        connector = OpcUaConnector(
            key="opcua",
            name=f"{self.name} OPC UA",
            context=self,
            configs=connector_configs
        )
        connector.configure(connector_configs)
        self.connectors.add(connector)

        def add_channel(constant: Constant, address: str, aggregate: str = "mean", **custom) -> None:
            channel = constant.to_dict()
            #channel["name"] = f"{self.name}_{channel['name']}"
            #channel["key"] = f"{self.key}_{channel['key']}"
            channel["connector"] = connector.id
            channel["address"] = address
            channel["aggregate"] = aggregate
            channel.update(custom)
            self.data.add(**channel)

        add_channel(ChargeBig.SETPOINT_ENABLED, "Summen_Grenzwert_aktiv", aggregate="last")
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

        defaults = ChargeBigStation._build_defaults(configs, strict=True)
        defaults[self.data.TYPE][Channels.TYPE]["connector"] = connector.id
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
            )
            station.configure(station_configs)

            self.components.add(station)


class ChargeBigStation(EVSE):
    STATE = Constant(float, "state", "State", alias="chargebig_state")
    LIMIT = Constant(float, "limit", "Current Limit", "A", alias="chargebig_limit")

    station_id: int

    def __init__(
        self,
        context: ContextArgument,
        configs: Configurations,
        station_index: int,     # table column
        station_id: int,        # physical id
        **kwargs,
    ) -> None:
        super().__init__(context=context, configs=configs, **kwargs)
        self.station_index = station_index
        self.station_id = station_id

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        def add_channel(constant: Constant, address: str, aggregate: str = "mean", **custom) -> None:
            channel = constant.to_dict()
            channel["station_id"] = self.station_index
            channel["address"] = address
            channel["aggregate"] = aggregate
            channel.update(custom)
            self.data.add(**channel)

        add_channel(ChargeBigStation.STATE, f"Ladepunkt_{self.station_id}_Status")
        add_channel(ChargeBigStation.LIMIT, f"Ladepunkt_{self.station_id}_Grenzwert")


