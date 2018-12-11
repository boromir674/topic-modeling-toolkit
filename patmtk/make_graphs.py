#!/usr/bin/python3

import sys
import argparse

from reporting import graph_maker


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='make_graphs.py', description='Creates graphs of evaluation scores and evolvable parameters tracked along the training process and saves them to disk', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name of the collection on which experiments were conducted')
    parser.add_argument('--models', dest='model_labels', type=str, nargs='*', help='the models to compare by plotting graphs; accepts list indices or model labels')
    parser.add_argument('--model-indices', dest='model_indices', type=int, nargs='*', help='whether to pick models specified by the given indices; it applies only if --models argument is not defined')
    parser.add_argument('--sort', '-s', metavar='metric', help='whether to request models after sorting them on the given metric')
    parser.add_argument('--top', '-t', type=int, help='request the first n models; it applies only if --models and -model-indices arguments are not defined')
    parser.add_argument('--range', '-r', nargs='*', type=int, help='request models defined by the range; applies only if --models, --model-indices and --top arguments are not defined')
    parser.add_argument('--metrics', metavar='metric_definitions', nargs='*', help='enable plotting for specific metrics tracked; i.e. "perplexity", "sparsity-phi-0.80", "top-tokens-coherence-10", etc. If --all-metrics flag is enabled it has no effect.')
    parser.add_argument('--all-metrics', '--a-m', action='store_true', default=False, help='whether to plot all possible score metrics tracked')
    parser.add_argument('--tau-trajectories', '--t-t', action='store_true', dest='plot_trajectories_flag', default=False, help='whether to plot the dynamic tau coefficients\' trajectories')
    parser.add_argument('--iterations', '-i', metavar='nb_points', type=int, help='whether to limit the datapoints plotted, to the specified number')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()

    print(args)

    if args.all_metrics:
        metrics = 'all'
    else:
        metrics = args.metrics

    if args.model_labels:
        selection = args.model_labels
    elif args.model_indices:
        selection = args.model_indices
        print([type(_) for _ in selection])
    elif args.top:
        selection = args.top
    elif args.range:
        selection = range(*args.range)
    else:
        selection = 8
    print(metrics)
    print(selection)

    graph_maker.build_graphs_from_collection(args.collection, selection,
                                             metric=args.sort,
                                             score_definitions=metrics,
                                             tau_trajectories=(lambda x: 'all' if x else '')(args.plot_trajectories_flag))