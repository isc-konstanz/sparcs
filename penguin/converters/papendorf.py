# -*- coding: utf-8 -*-
"""
penguin.converters.papendorf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

from typing import Type

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

    def convert(self, value: str) -> pd.Series:
        try:
            # TODO: Implement conversion
            print(value)
            return value

        except TypeError:
            pass
        raise ConversionException(f"Expected str or {self.dtype}, not: {type(value)}")

    def to_str(self, value: pd.Series) -> str | pd.Series:
        if issubclass(type(value), pd.Series):
            return value.apply(lambda v: json.dumps(v)).astype(str)
        return json.dumps(value)
