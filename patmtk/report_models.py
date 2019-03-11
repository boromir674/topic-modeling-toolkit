#!/usr/bin/python3.5

import argparse
from reporting import model_reporter
from reporting.reporter import InvalidMetricException


def get_cli_arguments():
    parser = argparse.ArgumentParser(prog='report_models.py', description='Reports on the trained models for the specified collection (dataset)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='collection_name', help='the collection to report models trained on')
    # parser.add_argument('--details', '-d', default=False, action='store_true', help='Switch to show details about the models')
    parser.add_argument('--sort', '-s', default='perplexity', help='Whether to sort the found experiments by checking the desired metric against the corresponding models')
    return parser.parse_args()


if __name__ == '__main__':

    COLUMNS = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity', 'kernel-size',
               'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
               'sparsity-theta',
               'background-tokens-ratio',
               # 'regularizers'
               ]

    # COLUMNS = ['nb-topics', 'collection-passes', 'perplexity']
               # 'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
               # 'sparsity-theta',
               # 'background-tokens-ratio',
               # 'regularizers']

    cli_args = get_cli_arguments()

    sort_metric = cli_args.sort

    while 1:
        try:
            s = model_reporter.get_formatted_string(cli_args.dataset, columns=COLUMNS, metric=sort_metric, verbose=True)
            print('\n{}'.format(s))
            break
        except InvalidMetricException as e:
            print(e)
            sort_metric = input("Please input another metric to sort (blank for 'perplexity'): ")
