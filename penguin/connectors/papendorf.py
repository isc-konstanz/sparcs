# -*- coding: utf-8 -*-
"""
    penguin.connectors.papendorf
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
from loris import Channels, ChannelState, Configurations, ConnectionException, Connector, ConnectorException
from loris.connectors.mysql import MySqlConnector


# noinspection SpellCheckingInspection
class PapendorfParser(Connector):
    TYPE: str = "papendorf"

    _database: MySqlConnector = None

    def __init__(self, context, configs: Configurations, channels: Channels = None, *args, **kwargs) -> None:
        super().__init__(configs, *args, **kwargs)
        # TODO: Implement papendorf initialisation
        # self._database = MySqlConnector(context, configs)

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        # TODO: Implement papendorf configurations

    def __connect__(self, channels: Channels) -> None:
        super().__connect__(channels)
        self._database.connect(channels)
        # TODO: Implement papendorf database connections

    @property
    def database(self):
        if self._database is None or not self._database.is_connected():
            raise ConnectionException("MySQL Connection not open", connector=self)
        return self._database

    def read(
        self,
        channels: Channels,
        start: Optional[pd.Timestamp, dt.datetime] = None,
        end: Optional[pd.Timestamp, dt.datetime] = None,
    ) -> None:
        # TODO: Implement papendorf parser reading
        for table_name, table_channels in channels.groupby("table"):
            if table_name not in self._database:
                raise ConnectorException(f"Table '{table_name}' not available", connector=self)

            table_columns = [c.id if "column" not in c else c.column for c in table_channels]
            table = self._database.get(table_name)

            if start is None and end is None:
                table_data = table.select_last(table_columns)
            else:
                table_data = table.select(table_columns, start, end)

            for table_channel in table_channels:
                table_channel_column = table_channel.id if "column" not in table_channel else table_channel.column
                if len(table_data.index) > 1:
                    table_channel_data = table_data.loc[:, table_channel_column]
                    table_channel.set(table_data.index[0], table_channel_data)

                elif len(table_data.index) > 0:
                    timestamp = table_data.index[-1]
                    table_channel_data = table_data.loc[timestamp, table_channel_column]
                    table_channel.set(timestamp, table_channel_data)

                else:
                    table_channel.state = ChannelState.NOT_AVAILABLE
                    self._logger.warning(
                        f"Unable to read nonexisting column of table '{table_name}': {table_channel_column}"
                    )

    def write(self, channels: Channels) -> None:
        raise NotImplementedError("Unable to write to papendorf MySQL tables")

    def is_connected(self) -> bool:
        return self._database is not None and self._database.is_connected()
