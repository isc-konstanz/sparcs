# -*- coding: utf-8 -*-
"""
penguin.converters.papendorf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import json
from typing import Literal, Type

import pandas as pd
from lori import Configurations
from lori.converters import ConversionException, Converter, register_converter_type


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

    def convert(self, value: str) -> pd.Series:
        try:
            # TODO: Implement conversion
            print(value)
            return None

        except TypeError:
            pass
        raise ConversionException(f"Expected str or {self.dtype}, not: {type(value)}")

    def revert(self, value: pd.Series) -> str | pd.Series:
        if issubclass(type(value), pd.Series):
            return value.apply(lambda v: json.dumps(v)).astype(str)
        return json.dumps(value)
