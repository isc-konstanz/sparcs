#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
penguin
~~~~~~~

To learn how to use local resource integration systems, see "penguin --help"

"""

import os
from argparse import ArgumentParser, RawTextHelpFormatter

os.environ["NUMEXPR_MAX_THREADS"] = str(os.cpu_count())


def main() -> None:
    import penguin

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {penguin.__version__}")

    with penguin.load(parser=parser) as application:
        application.main()


if __name__ == "__main__":
    main()
