#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    th-e-simulation
    ~~~~~~~~~~~~~~~
    
    To learn how to configure the photovoltaic yield simulation, see "th-e-simulation.py --help"

"""
import os
import sys
import copy
import inspect
import logging
import pytz as tz
import pandas as pd
import datetime as dt
import concurrent.futures as futures

from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))


def main(args):
    from th_e_yield import System

    settings_file = os.path.join(args.config_dir, 'settings.cfg')
    if not os.path.isfile(settings_file):
        raise ValueError('Unable to open simulation settings: {}'.format(settings_file))

    settings = ConfigParser()
    settings.read(settings_file)

    kwargs = vars(args)
    kwargs.update(dict(settings.items('General')))

    start = tz.utc.localize(dt.datetime.strptime(settings['General']['start'], '%d.%m.%Y'))
    end = tz.utc.localize(dt.datetime.strptime(settings['General']['end'], '%d.%m.%Y'))

    systems = System.read(**kwargs)
    for system in systems:
        system_dir = system._configs['General']['data_dir']
        database = copy.deepcopy(system._database)
        database.dir = os.path.join(system_dir, 'results')
        database.format = '%Y%m%d'
        database.disabled = False

        with futures.ThreadPoolExecutor() as executor:
            future = executor.submit(system._database.get, start, end)

            results = system.run(**dict(settings.items('General')))
            results['p_ref'] = future.result()['pv_power']
            results['p_err'] = results['p_mp'] - results['p_ref']
            for _, result in results.groupby([results.index.date]):
                database.persist(result)

            # FIXME: Optional outlier cleaning
            #results = results[(results['p_err'] < results['p_err'].quantile(.95)) & (results['p_err'] > results['p_err'].quantile(.05))]
            hours = results.loc[:,'p_err'].groupby([results.index.hour])
            median = hours.median()
            median.name = 'median'
            desc = pd.concat([median, hours.describe()], axis=1).loc[:, ['count', 'median', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            desc.to_csv(os.path.join(system_dir, 'results.csv'), sep=database.separator, decimal=database.decimal, encoding='utf-8')

            try:
                import seaborn as sns

                plot = sns.boxplot(x=results.index.hour, y='p_err', data=results, palette='Blues_d')
                plot.set(xlabel='hours', ylabel='Error [W]')
                plot.figure.savefig(os.path.join(system_dir, 'results.png'))

            except ImportError:
                pass


def _get_parser(root_dir):
    from th_e_yield import __version__

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument('-r', '--root-directory',
                        dest='root_dir',
                        help="directory where the package and related libraries are located",
                        default=root_dir,
                        metavar='DIR')

    parser.add_argument('-c', '--config-directory',
                        dest='config_dir',
                        help="directory to expect configuration files",
                        default='conf',
                        metavar='DIR')

    parser.add_argument('-d', '--data-directory',
                        dest='data_dir',
                        help="directory to expect and write result files to",
                        default='data',
                        metavar='DIR')
    
    return parser


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(root_dir) == 'bin':
        root_dir = os.path.dirname(root_dir)

    os.chdir(root_dir)

    if not os.path.exists('log'):
        os.makedirs('log')

    # Load the logging configuration
    import logging.config
    logging_file = os.path.join(os.path.join(root_dir, 'conf'), 'logging.cfg')
    logging.config.fileConfig(logging_file)
    logger = logging.getLogger('th-e-simulation')

    main(_get_parser(root_dir).parse_args())

