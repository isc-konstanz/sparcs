# -*- coding: utf-8 -*-
"""
penguin.system
~~~~~~~~~~~~~~


"""
from __future__ import annotations

import colorsys



class Color:
    hex_color: str

    def __init__(self, hex_color: str) -> None:
        self.hex_color = hex_color

    def __str__(self):
        return f"{self.hex_color}"

    def __repr__(self):
        return f"{self.hex_color}\trgb:{self.rgb}\thsv:{self.hsv}"

    @property
    def hex(self) -> str:
        return self.hex_color

    @property
    def rgb(self) -> tuple[float, float, float]:
        r = int(self.hex_color[1:3], 16) / 255.0
        g = int(self.hex_color[3:5], 16) / 255.0
        b = int(self.hex_color[5:7], 16) / 255.0
        return r, g, b

    @property
    def hsv(self) -> tuple[float, float, float]:
        return colorsys.rgb_to_hsv(*self.rgb)

    def set_hex(self, hex_color: str) -> None:
        if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
            raise ValueError("Hex color must be a string in the format '#RRGGBB'.")
        self.hex_color = hex_color

    def set_rgb(self, r: float, g: float, b: float) -> None:
        self.hex_color = "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

    def set_hsv(self, h: float, s: float, v: float) -> None:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        self.set_rgb(r, g, b)

    def copy(self) -> Color:
        return Color(self.hex_color)

    def range(self, n: int) -> list[str]:
        if n <= 0:
            return []
        colors = []
        for i in range(n):
            color = self.copy()
            h, s, v = color.hsv

            h = h - 0.03 + 0.06 * i / (n-1)
            h = h % 1.0  # Ensure hue wraps around
            v = v + 0.1 - 0.2 * i / (n-1)
            v = max(0, min(1, v))  # Ensure value is between 0 and 1

            color.set_hsv(h, s, v)

            colors.append(color.hex)
        return colors


