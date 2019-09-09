#!/usr/bin/env python

import os
import re
import sys
import click
from .reporting import GraphMaker
# from .patm.definitions import COLLECTIONS_DIR_PATH

c = ['perplexity', 'kernel-size', 'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
                        'sparsity-theta', 'background-tokens-ratio']

# class PythonLiteralOption(click.Option):
#     def type_cast_value(self, ctx, value):
#         try:
#             return ast.literal_eval(value)
#         except:
#             raise click.BadParameter(value)
class ModelLabelsParser(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return re.compile(r'(\w+)').findall(value)
        except:
            raise click.BadParameter(value)

class ScoreMetricsDefinitionParser(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return re.compile(r'([\w\-]+)').findall(value)
        except:
            raise click.BadParameter(value)

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('dataset')  #  help="the name of the dataset/collection in which to look for trained models"
@click.option('--models-labels', '-ml', cls=ModelLabelsParser, default='', help='Selects model(s) solely based on the names provided and ignores all other selection and sorting options below')
@click.option('--sort', '-s', default='perplexity', show_default=True, help="How to order the results' list, from which to select; supports 'alphabetical' and rest of metrics eg: {}".format(', '.join("'{}'".format(x) for x in c)))
@click.option('--model-indices', '-mi', cls=ModelLabelsParser, default='', type=int, help='Selects model(s) based on linear indices corresponding to the list of results per model (see --sort option), residing in collection directory and ignores all othe selection options below')
@click.option('--top', '-t', type=int, help='Selects the first n model results from the list and ignores all othe selection options below')
@click.option('--range-tuple', '-r', cls=ModelLabelsParser, default='', help='Selects model results defined by the range')
@click.option('--allmetrics/--no-allmetrics', default=False, show_default=True, help='Whether to create plots for all possible metrics (maximal) discovered in the model results')
@click.option('--metrics', cls=ScoreMetricsDefinitionParser, default='', help='Whether to limit plotting to the selected graph types (roughly score definition) i.e. "perplexity", "sparsity-phi-0.80", "top-tokens-coherence-10", If --allmetrics flag is enabled it has no effect.')
@click.option('--tautrajectories/--no-tautrajectories', default=False, show_default=True, help="Legacy flag: turn this on explicitly in case you have used dynamic regularization coefficient trajectory feature")
@click.option('--iterations', '-i', type=int, help='Whether to limit the datapoints plotted to the specified number. Defaults to plotting information for all training iterations')
@click.option('--legend/--no-legend', default=True, show_default=True, help="Whether to include legend information in the resulting graph images")
def main(dataset, models_labels, sort, model_indices, top, range_tuple, allmetrics, metrics, tautrajectories, iterations, legend):
    if models_labels:
        selection = models_labels
        print(selection)
    elif model_indices:
        selection = model_indices
    elif top:
        selection = top
    elif range_tuple:
        selection = range(*range_tuple)
    else:
        selection = 8
    if allmetrics:
        metrics = 'all'
    if tautrajectories:
        tautrajectories = 'all'
    if sys.version_info[1] == 3:
        print("The graph building process is probably going to result in an Attribute error ('Legend' object has no attribute 'draggable') "
              "due to a bug of the easyplot module when requesting legend information to be inculded in the graph image built."
              " For now please either invoke program with python2 interpreter or use the '--no-legend' flag.")
    collections_dir = os.getenv('COLLECTIONS_DIR')
    if not collections_dir:
        raise RuntimeError(
            "Please set the COLLECTIONS_DIR environment variable with the path to a directory containing collections/datasets")
    graph_maker = GraphMaker(collections_dir)
    graph_maker.build_graphs_from_collection(dataset, selection,
                                             metric=sort,
                                             score_definitions=metrics,
                                             tau_trajectories=tautrajectories,
                                             showlegend=legend)

    print("\nFigures' paths:")
    for _ in graph_maker.saved_figures:
        print(_)

if __name__ == '__main__':
    main()
