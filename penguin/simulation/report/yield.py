# -*- coding: utf-8 -*-
"""
penguin.simulation.report.system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import logging
from lori.simulation import Results
from lori.simulation.report import register_report_type
from lori.simulation.report.pdf import PdfReport

logger = logging.getLogger(__name__)


@register_report_type("yield", "pv_yield", "solar_yield")
class YieldReport(PdfReport):
    # noinspection PyShadowingBuiltins
    def add_results(self, results: Results) -> None:
        self.add_header('Page 1 Header Test', level=1)
        self.add_paragraph('Page 1 Content Test')

        self.add_header('Page 1 Subheader Test', level=2)
        self.add_paragraph('Page 1 Subcontent Test')

        self.add_page_break()

        self.add_header('Page 2 Header Test')
        self.add_paragraph('Page 2 Content Test')
