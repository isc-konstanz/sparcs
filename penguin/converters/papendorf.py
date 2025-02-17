# -*- coding: utf-8 -*-
"""
penguin.converters.papendorf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Iterable, Optional, Tuple, Type

import pandas as pd
from pandas.arrays import FloatingArray

from lori import Channels, Configurations
from lori.converters import Converter, ConversionException, register_converter_type

# FIXME: Remove this once Python >= 3.9 is a requirement
try:
    from typing import Literal

except ImportError:
    from typing_extensions import Literal


# noinspection SpellCheckingInspection
@register_converter_type("papendorf")
class PapendorfConverter(Converter):
    dtype: Type[FloatingArray] = FloatingArray

    curve_length_min: int
    current_min: float

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.curve_length_min = configs.get("curve_length_min", default=6)
        self.current_min = configs.get_float("current_min", default=0.1)

    def is_dtype(self, value: str | List) -> bool:
        if isinstance(value, (str, self.dtype)):
            return True
        return False

    # noinspection PyTypeChecker
    def to_dtype(self, iv_data: pd.DataFrame, axis: Literal["voltage", "current"] = None) -> Optional[FloatingArray]:
        if axis not in ["voltage", "current"]:
            raise ValueError(f"Invalid IV curve axis '{axis}'")
        if axis not in iv_data.columns:
            return None
        return pd.array(iv_data.loc[:, axis].values, dtype=pd.Float32Dtype())

    def convert(self, data: pd.DataFrame, channels: Channels) -> pd.DataFrame:
        converted_data = []
        for group_data, group_channels in self._groupby_iv_hex(data, channels):

            # noinspection PyProtectedMember
            def _convert(iv_row: pd.Series) -> Optional[pd.Series]:
                _converted_rows = []
                _converted_index = []

                iv_data = self._convert_iv_hex(iv_row["iv_hex"])
                for channel in channels:
                    _converted_index.append(channel.id)
                    if iv_data is None:
                        _converted_rows.append(None)
                        continue
                    try:
                        _converted_rows.append(self.to_dtype(iv_data, axis=channel.converter.get("axis")))

                    except TypeError:
                        raise ConversionException(f"Expected str or {self.dtype}, not: {type(data)}")
                return pd.Series(_converted_rows, index=_converted_index)

            converted_data.append(group_data.to_frame().apply(_convert, axis="columns"))

        if len(converted_data) == 0:
            return pd.DataFrame(columns=[c.id for c in channels])
        return pd.concat(converted_data, axis="columns").dropna(axis="index", how="all")

    def _convert_iv_hex(self, iv_hex: str) -> Optional[pd.DataFrame]:
        try:
            iv_hex = [iv_hex[y - 8:y] for y in range(8, len(iv_hex) + 8, 8)]
            iv_data = OrderedDict()
            for n in range(len(iv_hex)):
                v_hex = iv_hex[n][2] + iv_hex[n][3] + iv_hex[n][0] + iv_hex[n][1]
                i_hex = iv_hex[n][6] + iv_hex[n][7] + iv_hex[n][4] + iv_hex[n][5]
                iv_data[n] = {
                    "voltage": int(v_hex, 16) / 100.,
                    "current": int(i_hex, 16) / 1000.,
                }
            iv_curve = pd.DataFrame.from_records(
                data=list(iv_data.values()),
                index=list(iv_data.keys()),
                columns=["voltage", "current"]
            )
            iv_curve["voltage"] = iv_curve["voltage"].round(1)
            iv_curve = iv_curve.groupby("voltage").mean().round(2)
            iv_curve = iv_curve.reset_index()

            if iv_curve["current"].max() < self.current_min or len(iv_curve) <= self.curve_length_min:
                return None
            return iv_curve

        except TypeError:
            pass
        raise ConversionException(f"Expected str or {self.dtype}, not: {type(iv_hex)}")

    # noinspection PyTypeChecker
    def _groupby_iv_hex(self, data: pd.DataFrame, channels: Channels) -> Iterable[Tuple[pd.Series, Channels]]:
        groups = []
        for channel in channels:
            channel_iv = data.loc[:, channel.id].dropna() if channel.id in data.columns else None
            channel_iv.name = "iv_hex"
            if channel_iv is None or channel_iv.empty or not all(channel_iv.apply(self.is_dtype)):
                continue
            group_match = False
            for group_iv, group_channels in groups:
                if all(group_iv == channel_iv):
                    group_channels.append(channel)
                    group_match = True
                    break
            if not group_match:
                groups.append((channel_iv, channel.to_list()))
        return groups
