# -*- coding: utf-8 -*-
"""
    loris.application
    ~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import loris
from loris import Settings
from penguin import System


# noinspection PyUnresolvedReferences
def load(name: str = "Penguin", **kwargs) -> Application:
    return Application.load(Settings(name, **kwargs), factory=System)


class Application(loris.Application):
    def _run(self, *args, **kwargs) -> None:
        for system in self.components.get_all(System):
            try:
                # TODO: Implement check if data was updated
                self._logger.debug(f"Running {type(system).__name__}: {system.name}")
                system.run(*args, **kwargs)

            except Exception as e:
                self._logger.warning(f"Error running system '{system.id}': ", e)
