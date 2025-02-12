# -*- coding: utf-8 -*-
"""
penguin.converters.papendorf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

from typing import Type

import array as arr
import pandas as pd
import json

from lori import Configurations
from lori.converters import Converter, ConversionException, register_converter_type

# FIXME: Remove this once Python >= 3.9 is a requirement
try:
    from typing import Literal

except ImportError:
    from typing_extensions import Literal


# noinspection SpellCheckingInspection
@register_converter_type("papendorf")
class PapendorfConverter(Converter):
    dtype: Type[pd.Series] = pd.Series

    curve: Literal["current", "voltage"]

    def is_dtype(self, value: str | pd.Series) -> bool:
        if isinstance(value, (str, self.dtype)):
            return True
        return False

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)
        self.curve = configs.get("curve")
        self.min_curve_length = configs.get_int("min_curve_length", default=6)

    def convert(self, iv_hexstring: str) -> pd.Series:
        try:
            iv_hex = [iv_hexstring[y - 8:y] for y in range(8, len(iv_hexstring) + 8, 8)]

            u_min = arr.array('d', )
            i_min = arr.array('d', )
            u_rnd_min = arr.array('d', )
            for n in range(len(iv_hex)):
                u_hex = iv_hex[n][2] + iv_hex[n][3] + iv_hex[n][0] + iv_hex[n][1]
                i_hex = iv_hex[n][6] + iv_hex[n][7] + iv_hex[n][4] + iv_hex[n][5]

                i = int(i_hex, 16) / 1000
                i_min.append(i)

                u = int(u_hex, 16) / 100
                u_min.append(u)

                u_rnd = int(round(int(u_hex, 16) / 100, 0))
                u_rnd_min.append(u_rnd)

            iv_min = {'v_rnd': u_rnd_min, 'v': u_min, 'i': i_min}
            iv_curve = pd.DataFrame(iv_min)
            iv_curve = iv_curve.drop_duplicates(subset=['v_rnd'])
            iv_curve = iv_curve.drop(columns=['v_rnd']).T

            i = pd.Series(list(iv_curve.iloc[1].values))
            v = pd.Series(list(iv_curve.iloc[0].values))

            if len(i) >= self.min_curve_length:
                if self.curve == "current":
                    iv_value = i
                elif self.curve == "voltage":
                    iv_value = v
            else:
                iv_value = None

            return iv_value

        except TypeError:
            pass
        raise ConversionException(f"Expected str or {self.dtype}, not: {type(iv_hexstring)}")

    def to_str(self, value: pd.Series) -> str | pd.Series:
        if issubclass(type(value), pd.Series):
            return value.apply(lambda v: json.dumps(v)).astype(str)
        return json.dumps(value)
