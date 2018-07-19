#!usr/bin/python3
# -*- coding: utf8 -*-

import os
import sys
import json
import argparse
import pickle
import matplotlib as mlp
import matplotlib.pyplot as plt
from collections import Counter

plt.ion()
from easyplot import EasyPlot


collections_dir = '/data/thesis/data/collections'

def _dictify_results(results):
    tr = results['trackables']
    for k, v in tr.items():
        if type(v) == str:
            tr[k] = eval(v)
        elif type(v) == dict:
            for in_k, in_v in v.items():
                if type(in_v) == list:
                    try:
                        # print(type(constructor[in_k]))
                        v[in_k] = [eval(_) for _ in in_v]
                        # print(type(constructor[in_k]))
                    except RuntimeError as e:
                        print(e)
    return results

def load_results(path_file):
    with open(path_file, 'r') as results_file:
        results = json.load(results_file, encoding='utf-8')
    assert 'collection_passes' in results and 'trackables' in results, 'root_dir' in results
    return _dictify_results(results)


class GraphMaker(object):
    # tau_type2
    def __init__(self, plot_dir):
        self.line_designs = ['b',
                             'y',
                             'g',
                             'k',
                             'c',
                             'm',
                             'p',
                             'r'
                             ]
        self._tau_traj_extractor = {
            'phi': lambda x: _infer_trajectory_values_list([(span, params['sparse-phi']['tau']) for span, params in x['reg_parameters']]),
            'theta': lambda x: _infer_trajectory_values_list([(span, params['sparse-theta']['tau']) for span, params in x['reg_parameters']]),
        }
        self._cross_tau_titler = lambda x: 'Coefficient tau for {}-matrix-sparsing regularizer'.format(x)
        # self._all_taus_titler = lambda x, y: 'Coefficients tau for {} sparsing regularizer for model {}'.format(x, y)
        # linestyle / ls: Plot linestyle['-', '--', '-.', ':', 'None', ' ', '']
        # marker: '+', 'o', '*', 's', 'D', ',', '.', '<', '>', '^', '1', '2'

        self._max_digits_prepend = 2
        self._plot_counter = Counter()
        self.eplot = None
        self._max_len = 0 # this variable holds the longest value list of y's length
        self.gr_dir = plot_dir
        if os.path.exists(self.gr_dir):
            if not os.path.isdir(self.gr_dir):
                target_dir = '\\tmp\graphs'
                print('Found file \'{}\'. Output graphs will be stored in', target_dir)
                os.makedirs(target_dir)
                print('Created \'{}\' directory'.format(target_dir))
            else:
                print('Using found \'{}\' directory to store graphs'.format(self.gr_dir))
        else:
            os.makedirs(self.gr_dir)
            print('Created \'{}\' directory'.format(self.gr_dir))

    def save_tau_trajectories(self, results, nb_points=None):
        """Plots tau coefficient value trajectory for sparsing phi and theta matrices and saves to disk"""
        assert len(results) <= len(self.line_designs)
        graph_plots = []
        labels_list = list(map(lambda x: x['model_label'], results))
        for tau_type, extractor in self._tau_traj_extractor.items():
            ys, xs = self._build_ys(results, extractor, nb_points=nb_points)
            graph_plots.append(('{}-tau-{}'.format('-'.join(map(lambda x: x['model_label'], results)), tau_type),
                                _build_graph(xs, ys, self.line_designs[:len(results)], labels_list, self._cross_tau_titler(tau_type), 'iteration', 'τ')))
        for graph_type, plot in graph_plots:
            self._save_plot(graph_type, plot)

    def build_metric_graphs(self, results, scores='all', nb_points=None):
        """
        Call this method to create and save comparison plots between the tracked metrics of the given experimental results. Currently supported maximum 8 plots on the same figure\n.
        :param list of dicts results: experimental results, gathered after model training
        :param list or str scores: if 'all' then builds all admissible scores, else builds the custom selected scores
        :param int nb_points: number of points to plot. Defaults to plotting all measurements found
        """
        if scores == 'all':
            scores = sorted(results[0]['trackables'].keys())
        for score in scores:
            try:
                graph_types_n_eplots = self._build_metric_graphs(results, score, sub_scores='all', limit_iteration=nb_points)
                for graph_type, eplot_obj in graph_types_n_eplots:
                    self._save_plot(graph_type, eplot_obj)
            except KeyError:
                print("Score '{}' is not found in tracked experimental result metrics".format(score))

    def _build_ys(self, results, extractor, nb_points=None):
        ys = list(map(lambda x: _limit_points(extractor(x), nb_points), results))
        return ys, [range(len(_)) for _ in ys]

    def _build_metric_graphs(self, results, score, sub_scores='all', limit_iteration=None):
        assert len(results) <= len(self.line_designs)
        if score not in results[0]['trackables']:
            raise KeyError
        labels_list = list(map(lambda x: x['model_label'], results))
        graph_plots = []
        if sub_scores == 'all':
            sub_scores = sorted(results[0]['trackables'][score].keys())
        for sub_score in sub_scores:
            measure_name = score + '-' + sub_score
            try:
                ys, xs = self._build_ys(results, lambda x: x['trackables'][score][sub_score], limit_iteration)
                graph_plots.append(('{}-{}'.format('-'.join(labels_list), measure_name),
                                    _build_graph(xs, ys, self.line_designs[:len(results)], labels_list, measure_name.replace('-', '.'), 'iteration', 'y')))
            except TypeError:
                print("Did not create sub-score '{}' plot of '{}'".format(sub_score, score))
        return graph_plots

    def _save_plot(self, graph_type, eplot):
        target_name = self._get_target_name(graph_type)
        while os.path.exists(target_name): # while old graph files are found with the same name, increment the plot 'version'
            target_name = self._get_target_name(graph_type)
        eplot.kwargs['fig'].savefig(target_name)
        print('Saved figure as', target_name)

    def _get_target_name(self, graph_type):
        self._plot_counter[graph_type] += 1
        return os.path.join(self.gr_dir, '{}_{}.png'.format(graph_type, self._iter_prepend(self._plot_counter[graph_type])))

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits >= self._max_digits_prepend:
            return str(int_num)
        return '{}{}'.format((self._max_digits_prepend - nb_digits) * '0', int_num)


def _infer_trajectory_values_list(iterspan_value_tuples):
    _ = []
    for span, val in list(iterspan_value_tuples):
        _.extend([val] * span)
    return _

def _limit_points(value_list, limit):
    if limit is None:
        return value_list
    else:
        return value_list[:limit]

def _build_graph(xs, ys, line_designs, labels, title, xlabel, ylabel, grid='on'):
    assert len(ys) == len(xs) == len(line_designs) == len(labels)
    pl = EasyPlot(xs[0], ys[0], line_designs[0], label=labels[0], showlegend=True, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
    for i, y_vals in enumerate(ys[1:]):
        pl.add_plot(xs[i+1], y_vals, line_designs[i+1], label=labels[i+1])
    return pl

            # Advanced grid modification
    # eplot.new_plot(x, 1 / (1 + x), '-s', label=r"$y = \frac{1}{1+x}$", c='#fdb462')
    # eplot.grid(which='major', axis='x', linewidth=2, linestyle='--', color='b', alpha=0.5)
    # eplot.grid(which='major', axis='y', linewidth=2, linestyle='-', color='0.85', alpha=0.5)

#     eplot = EasyPlot(x, res_dict['trackables'], 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')
#     eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict, label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')


class NoDataGivenPlotCreationException(Exception):
    def __init__(self, msg):
        super(NoDataGivenPlotCreationException, self).__init__(msg)


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='make_graphs.py', description='Creates graphs of evaluation scores tracked along the training process and saves them to disk', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name of the collection on which experiments were conducted')
    parser.add_argument('--models', metavar='model_labels', nargs='+', help='the models to compare by plotting graphs')
    parser.add_argument('--metrics', help='Enable pltotting for specific metrics tracked. Supports comma separated values i.e. "perplexity,sparsity-phi"')
    parser.add_argument('--all-metrics', '--a-m', dest='all_metrics', action='store_true', default=False, help='Enable pltotting of all possible metrics tracked')
    parser.add_argument('--tau-trajectories', '--t-t', action='store_true', dest='plot_tau_trajectories', default=False, help='Enable ploting of the dynamic tau coefficients\' trajectories')
    parser.add_argument('--iterations', '-i', metavar='nb_points', type=int, help='limit or not the dapoints plotted, to the specified number')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cl_arguments()
    if args.all_metrics:
        metrics = 'all'
    elif args.metrics:
        metrics = args.metrics.split(',')
    else:
        print("Either enable '--all-metrics' or use '--metrics'")
        sys.exit(1)
    col_root = os.path.join(collections_dir, args.collection)
    if os.path.isdir(col_root):
        plotter = GraphMaker(os.path.join(col_root, 'graphs'))
    else:
        print("Collection '{}' not found in {}".format(args.collection, col_root))
        sys.exit(1)
    exp_results = []
    for model_label in args.models:
        exp_results.append(load_results(os.path.join(col_root, 'results', model_label + '-train.json')))

    plotter.build_metric_graphs(exp_results, scores=metrics, nb_points=args.iterations)
    if args.plot_tau_trajectories:
        # plotter.save_tau_trajectories(exp_results, nb_points=args.iterations, cross=True)
        plotter.save_tau_trajectories(exp_results, nb_points=args.iterations)

# eplot = EasyPlot(xlabel=r'$x$', ylabel='$y$', fontsize=16,
#                  colorcycle=["#66c2a5", "#fc8d62", "#8da0cb"], figsize=(8, 5))
# eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict,
#                 label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')
