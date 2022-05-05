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
    from th_e_core import configs
    from th_e_yield import System

    settings = configs.read('settings.cfg', **vars(args))

    kwargs = vars(args)
    kwargs.update(settings.items('General'))

    start = tz.utc.localize(dt.datetime.strptime(settings['General']['start'], '%d.%m.%Y'))
    end = tz.utc.localize(dt.datetime.strptime(settings['General']['end'], '%d.%m.%Y'))

    systems = System.read(**kwargs)
    for system in systems:
        system_dir = system.configs['General']['data_dir']
        database = copy.deepcopy(system._database)
        database.dir = os.path.join(system_dir, 'results')
        database.format = '%Y%m%d'
        database.disabled = False

        with futures.ThreadPoolExecutor() as executor:
            future = executor.submit(system._database.read, start, end)

            results = system.run(**dict(settings.items('General')))
            results['p_ref'] = future.result()['pv_power']
            results['p_err'] = results['p_mp'] - results['p_ref']
            for _, result in results.groupby([results.index.date]):
                database.persist(result)

            # FIXME: Optional outlier cleaning
            # results = results[(results['p_err'] < results['p_err'].quantile(.95)) & (results['p_err'] > results['p_err'].quantile(.05))]
            hours = results.loc[:, 'p_err'].groupby([results.index.hour])
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
    run_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(run_dir) == 'bin':
        run_dir = os.path.dirname(run_dir)

    os.chdir(run_dir)

    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

    if not os.path.exists('log'):
        os.makedirs('log')

    logging_file = os.path.join(os.path.join(run_dir, 'conf'), 'logging.cfg')
    if not os.path.isfile(logging_file):
        logging_default = logging_file.replace('logging.cfg', 'logging.default.cfg')
        if os.path.isfile(logging_default):
            shutil.copy(logging_default, logging_file)
        else:
            raise FileNotFoundError("Unable to open logging.cfg in: " +
                                    os.path.join(os.path.join(run_dir, 'conf')))

    # Load the logging configuration
    import logging
    import logging.config
    logging.config.fileConfig(logging_file)
    logger = logging.getLogger('th-e-simulation')

    main(_get_parser(run_dir).parse_args())
