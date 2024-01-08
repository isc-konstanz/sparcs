# -*- coding: utf-8 -*-
"""
    pvsys.report
    ~~~~~~~~~~~~


"""
from __future__ import annotations

import logging
import scisys as core
from scisys import Results

logger = logging.getLogger(__name__)


class Report(core.Report):

    def build(self, pdf, results: Results) -> None:
        super().build(pdf, results)

        pdf.add_header('Page 1 Header Test', level=1)
        pdf.add_paragraph('Page 1 Content Test')

        pdf.add_header('Page 1 Subheader Test', level=2)
        pdf.add_paragraph('Page 1 Subcontent Test')

        pdf.add_page_break()

        pdf.add_header('Page 2 Header Test')
        pdf.add_paragraph('Page 2 Content Test')
