
# -*- coding: utf8 -*-
import sys
import argparse


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='make_graphs.py', description='Creates graphs of evaluation scores and evolvable parameters tracked along the training process and saves them to disk', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name of the collection on which experiments were conducted')
    parser.add_argument('--models', metavar='model_labels', nargs='+', help='the models to compare by plotting graphs')
    parser.add_argument('--metrics', type=list, help='enable plotting for specific metrics tracked; i.e. "perplexity", "sparsity-phi-0.80", "top-tokens-coherence-10", etc. If --all-metrics flag is enabled it has no effect.')
    parser.add_argument('--all-metrics', '--a-m', dest='all_metrics', action='store_true', default=False, help='whether to plot all possible score metrics tracked')
    parser.add_argument('--tau-trajectories', '--t-t', action='store_true', dest='plot_tau_trajectories', default=False, help='whether to plot the dynamic tau coefficients\' trajectories')
    parser.add_argument('--iterations', '-i', metavar='nb_points', type=int, help='whether to limit the datapoints plotted, to the specified number')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    from reporting.graph_maker import GraphMaker
    from patm.definitions import COLLECTIONS_DIR_PATH, GRAPHS_DIR_NAME
    args = get_cl_arguments()

    print(args)

    if args.all_metrics:
        metrics = 'all'
    else:
        metrics = args.metrics

    # from reporting.model_selection import ResultsHandler

    # ResultsHandler.
    # graph_maker = GraphMaker(os.path.join(COLLECTIONS_DIR_PATH, args.collection, GRAPHS_DIR_NAME))
    # graph_maker.build_graphs()