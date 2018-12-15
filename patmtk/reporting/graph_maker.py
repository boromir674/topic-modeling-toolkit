# -*- coding: utf8 -*-

import os
from functools import reduce
from collections import Counter
# import matplotlib as mlp

import matplotlib.pyplot as plt
plt.ion()
from easyplot import EasyPlot

from . import results_handler


class GraphMaker(object):
    SUPPORTED_GRAPHS = ['perplexity', 'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
                        'sparsity-theta', 'background-tokens-ratio']
    LINES = ['b', 'y', 'g', 'k', 'c', 'm', 'p', 'r']
    # linestyle / ls: Plot linestyle['-', '--', '-.', ':', 'None', ' ', '']
    # marker: '+', 'o', '*', 's', 'D', ',', '.', '<', '>', '^', '1', '2'
    def __init__(self, collections_root_path, plot_dir_name='graphs'):
        self._collections_dir_path = collections_root_path
        self._plot_dir_name = plot_dir_name
        self._verbose_save = True
        self._save_lambdas = {True: lambda z: self._save_plot_n_return(z[0], z[1], verbose=self._verbose_save), False: lambda y: (x[0], x[1])}
        self._model_label_joiner = '+'
        self._results_indices = []
        self._exp_results_list = []
        self._tau_traj_extractor = {'phi': lambda x: getattr(x.tracked.tau_trajectories, 'phi').all,
                                    'theta': lambda x: getattr(x.tracked.tau_trajectories, 'theta').all}
        self._max_digits_prepend = 2
        self._plot_counter = Counter()
        self._graph_names_n_eplots = []  # list of tuples

    def _prepare_output_folder(self, collection_name):
        self._output_dir_path = os.path.join(self._collections_dir_path, collection_name, self._plot_dir_name)
        if os.path.exists(self._output_dir_path):
            if not os.path.isdir(self._output_dir_path):
                target_dir = '\\tmp\graphs'
                print('Found file \'{}\'. Output graphs will be stored in', target_dir)
                os.makedirs(target_dir)
                print('Created \'{}\' directory'.format(target_dir))
            else:
                print('Using found \'{}\' directory to store graphs'.format(self._output_dir_path))
        else:
            os.makedirs(self._output_dir_path)
            print('Created \'{}\' directory'.format(self._output_dir_path))

    def build_graphs_from_collection(self, collection_name, selection, metric='perplexity',
                                     score_definitions='all', tau_trajectories='all', save=True, nb_points=None, verbose=True):
        """
        Call this method to create plots for the given collection.\n
        :param str collection_name:
        :param str or range or int or list selection: whether to select a subset of the experimental results fron the given collection\n
            - if selection == 'all', returns every experimental results object "extracted" from the jsons
            - if type(selection) == range, returns a "slice" of the experimental results based on the range
            - if type(selection) == int, returns the first n experimental results
            - if type(selection) == list, then it represents specific indices to sample the list of experimental results from
        :param str metric:
        :param list or str score_definitions:
        :param list or str tau_trajectories:
        :param bool save:
        :param int nb_points:
        :param bool verbose:
        """
        self._prepare_output_folder(collection_name)
        self._exp_results_list = results_handler.get_experimental_results(collection_name, sort=metric, selection=selection)
        print("Retrieved {} models from collection '{}', sorted {}.\nModels: [{}]".format(
            len(self._exp_results_list), collection_name, (lambda x: x if x else 'alphabetically')(metric),
            ', '.join((_.scalars.model_label for _ in self._exp_results_list))))
        self.build_graphs(self._exp_results_list, score_definitions=score_definitions, tau_trajectories=tau_trajectories,
                          save=save, nb_points=nb_points, verbose=verbose)

    def build_graphs(self, results, score_definitions='all', tau_trajectories='all', save=True, nb_points=None, verbose=True):
        self._verbose_save = verbose
        if score_definitions == 'all':
            score_definitions = self.determine_metrics_usable_for_comparison(results)
        if tau_trajectories == 'all':
            tau_trajectories = ['phi', 'theta']
        self._graph_names_n_eplots.extend(self.build_metrics_graphs(results, scores=score_definitions, save=save, nb_points=nb_points, verbose=verbose))
        self._graph_names_n_eplots.extend(self.build_tau_trajectories_graphs(results, matrices=tau_trajectories, save=save, nb_points=nb_points, verbose=verbose))

    def build_tau_trajectories_graphs(self, results, matrices='all', save=True, nb_points=None, verbose=True):
        return [self._save_lambdas[save](self._build_metric_graph(results, x, limit_iteration=nb_points, verbose=verbose)) for x in matrices]

    def build_metrics_graphs(self, results, scores='all', save=True, nb_points=None, verbose=True):
        """
        Call this method to create and potentially save comparison plots between the tracked metrics (scores) of the given experimental results. Currently supported maximum 8 plots on the same figure\n.
        :param list of results.experimental_results.ExperimentalResults results:
        :param list or str scores: if 'all' then builds all admissible scores, else builds the custom selected scores
        :param bool save: whether to save figure on disk as .png files
        :param int nb_points: number of points to plot. Defaults to plotting all measurements found
        :param bool verbose:
        """
        return [self._save_lambdas[save](self._build_metric_graph(results, x, limit_iteration=nb_points, verbose=verbose)) for x in scores]

    def _build_metric_graph(self, exp_results_list, metric, limit_iteration=None, verbose=True):
        """Call this method to get a graph name (as a string) and a graph (as an Eplot object) tuple.\n
        :param list of results.experimental_results.ExperimentalResults exp_results_list:
        :param str metric: Supports unique metric definitions such as {'perplexity', 'sparsity-theta', 'sprasity-phi-d',
            'sparsity-phi-i', 'kernel-coherence-0.80', 'top-tokens-coherence-100', ..} as well as the 'phi-tau-trajectory' and 'theta-tau-trajectory' tracked values
        :param None or int limit_iteration: whether to limit the length at which it will plot along the x axis: if None, plots all available datapoint; if int value given limits plotting at a maximum of 'value' along the x axis
        """
        assert len(exp_results_list) <= len(GraphMaker.LINES)
        labels = [_.scalars.model_label for _ in exp_results_list]
        print('_BUILD METRIC', metric)
        if metric in ('phi', 'theta'):
            measure_name = 'τ_{}'.format(metric)
        else:
            measure_name = results_handler.get_abbreviation(metric).replace('-', '.')
        extractor = self._tau_traj_extractor.get(metric, lambda x: results_handler.extract(x, metric, 'all'))
        ys, xs = self._get_ys_n_xs(exp_results_list, extractor, nb_points=limit_iteration)
        return '{}-{}'.format(self._model_label_joiner.join(labels), measure_name), \
               GraphMaker.build_graph(xs, ys,
                                      GraphMaker.LINES[:len(exp_results_list)],
                                      labels,
                                      title=measure_name,
                                      xlabel='iteration',
                                      ylabel='y')

    def _save_plot_n_return(self, graph_type, eplot, verbose=True):
        self._save_plot(graph_type, eplot, verbose=verbose)
        return graph_type, eplot

    def _save_plot(self, graph_type, eplot, verbose=True):
        """Call this method to save a graph on the disk as a .png file. The input graph type contributes to naming the file
        and appending any numbers for versioning and the plot object has a 'save' method."""
        target_name = self._get_target_file_path(graph_type)
        while os.path.exists(target_name): # while old graph files are found with the same name and version number, increment the plot 'version'
            target_name = self._get_target_file_path(graph_type)
        eplot.kwargs['fig'].savefig(target_name)
        if verbose:
            print('Saved figure as', target_name)

    def _get_target_file_path(self, graph_type):
        self._plot_counter[graph_type] += 1
        return os.path.join(self._output_dir_path, '{}_{}.png'.format(graph_type, self._iter_prepend(self._plot_counter[graph_type])))

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits >= self._max_digits_prepend:
            return str(int_num)
        return '{}{}'.format((self._max_digits_prepend - nb_digits) * '0', int_num)

    def _get_ys_n_xs(self, exp_results_list, extractor, nb_points=None):
        if 0:
            ys = filter(None, map(extractor, exp_results_list))
            self._results_indices = [ind for ind, el in enumerate(ys) if el is not None]
            # assert len(ys) == len(self._results_indices)
            # print 'results_indices len: {}. MUST BE LENGTH number of plots on the same graph.'
            ys = [GraphMaker._limit_points(_, nb_points) for _ in ys]
            # ys = list([GraphMaker._limit_points(x, nb_points) for x in [_f for _f in ys if _f]])
        # elif 0:
        #     ys = GraphMaker._limit_points(list(map(lambda x: ResultsHandler.extract(x, metric_definition), exp_results_list)), nb_points)
        #     self._results_indices = range(len(ys))
        #     return ys, [range(len(_)) for _ in ys]
        else:
            ys = [GraphMaker._limit_points(extractor(_), nb_points) for _ in exp_results_list]
            # ys = [_f for _f in [GraphMaker._limit_points(extractor(x), nb_points) for x in exp_results_list] if _f]
            # ys = list(filter(None, map(lambda x: GraphMaker._limit_points(ResultsHandler.extract(x, metric_definition), nb_points), exp_results_list)))
            self._results_indices = list(range(len(ys)))
        return ys, [list(range(len(_))) for _ in ys]

    @classmethod
    def determine_metrics_usable_for_comparison(cls, exp_results_list):
        c = Counter()
        _ = reduce(lambda i, j: i + j, [results_handler.get_all_columns(x, cls.SUPPORTED_GRAPHS) for x in exp_results_list])
        c.update(_)
        return [k for k, v in c.items() if v > 1]

    @staticmethod
    def build_graph(xs, ys, line_designs, labels, title, xlabel, ylabel, grid='on'):
        assert len(ys) == len(xs) == len(line_designs) == len(labels)
        pl = EasyPlot(xs[0], ys[0], line_designs[0], label=labels[0], showlegend=True, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
        for i, y_vals in enumerate(ys[1:]):
            pl.add_plot(xs[i + 1], y_vals, line_designs[i + 1], label=labels[i + 1])
        return pl

        # Advanced grid modification
        # eplot.new_plot(x, 1 / (1 + x), '-s', label=r"$y = \frac{1}{1+x}$", c='#fdb462')
        # eplot.grid(which='major', axis='x', linewidth=2, linestyle='--', color='b', alpha=0.5)
        # eplot.grid(which='major', axis='y', linewidth=2, linestyle='-', color='0.85', alpha=0.5)

        #     eplot = EasyPlot(x, tracked_metrics_dict['trackables'], 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')
        #     eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict, label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')

    @staticmethod
    def _limit_points(values_list, limit):
        if limit is None: return values_list
        return values_list[:limit]
