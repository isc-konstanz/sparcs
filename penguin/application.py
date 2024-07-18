# -*- coding: utf-8 -*-
"""
    loris.application
    ~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

from typing import Type

import loris
from loris import Settings
from penguin import System


def load(name: str = "Penguin", factory: Type[System] = System, **kwargs) -> Application:
    return Application(Settings(name, **kwargs)).setup(factory)


class Application(loris.Application):
    # noinspection PyUnresolvedReferences
    def _run(self, *args, **kwargs) -> None:
        for system in self.components.get_all(System):
            try:
                # TODO: Implement check if data was updated
                self._logger.debug(f"Running {type(system).__name__}: {system.name}")
                system.run(*args, **kwargs)

            except Exception as e:
                self._logger.warning(f"Error running system '{system.id}': {str(e)}")
                self._logger.exception(e)
