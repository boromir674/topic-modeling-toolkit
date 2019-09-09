import os
from functools import reduce
from collections import Counter

import matplotlib.pyplot as plt
# plt.ion()

from easyplot import EasyPlot
from .model_selection import ResultsHandler


class GraphMaker(object):
    SUPPORTED_GRAPHS = ['perplexity', 'kernel-size', 'kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi',
                        'sparsity-theta', 'background-tokens-ratio']
    LINES = ['m', 'y', 'b', 'g', 'k', 'c', 'r', '#4D79D1']
    # linestyle / ls: Plot linestyle['-', '--', '-.', ':', 'None', ' ', '']
    # marker: '+', 'o', '*', 's', 'D', ',', '.', '<', '>', '^', '1', '2'
    def __init__(self, collections_root_path, plot_dir_name='graphs', results_dir_name='results'):
        self._collections_dir_path = collections_root_path
        self.results_handler = ResultsHandler(self._collections_dir_path, results_dir_name=results_dir_name)
        self._plot_dir_name = plot_dir_name
        self._verbose_save = True
        self._save_lambdas = {True: lambda z: self._save_plot_n_return(z[0], z[1], verbose=self._verbose_save),
                              False: lambda y: (y[0], y[1])}
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

    @property
    def saved_figures(self):
        return [_[0] for _ in self._graph_names_n_eplots if _[1]]

    def build_graphs_from_collection(self, collection_name, selection, metric='perplexity',
                                     score_definitions='all', tau_trajectories='all', save=True, nb_points=None, showlegend=False, verbose=True):
        """
        Call this method to create plots for the given collection.\n
        :param str collection_name:
        :param str or range or int or list selection: whether to select a subset of the experimental results fron the given collection\n
            - if selection == 'all', returns every experimental results object "extracted" from the jsons
            - if type(selection) == range, returns a "slice" of the experimental results based on the range
            - if type(selection) == int, returns the first n experimental results
            - if type(selection) == list, then it returns the experimental results indexed by the list
        :param str metric: If 'alphabetical' or None picks model after sorting alphabeticall by file name. Else supports any tracked metric definition
        :param list or str score_definitions:
        :param list or str tau_trajectories:
        :param bool save:
        :param int or None nb_points:
        :param bool verbose:
        """
        self._prepare_output_folder(collection_name)
        if metric == 'alphabetical':
            metric = None
        self._exp_results_list = self.results_handler.get_experimental_results(collection_name, sort=metric, selection=selection)
        print("Retrieved {} models from collection '{}'. Will sort with '{}'.\nModels: [{}]".format(
            len(self._exp_results_list), collection_name, (lambda x: x if x else 'their labels alphabetically')(metric),
            ', '.join((_.scalars.model_label for _ in self._exp_results_list))))
        self.build_graphs(self._exp_results_list, score_definitions=score_definitions, tau_trajectories=tau_trajectories,
                          save=save, nb_points=nb_points, showlegend=showlegend, verbose=verbose)

    def build_graphs(self, results, score_definitions='all', tau_trajectories='all', save=True, nb_points=None, showlegend=False, verbose=True):
        self._verbose_save = verbose
        if score_definitions == 'all':
            score_definitions = self.determine_metrics_usable_for_comparison(results)
        print("Graph definitins: {}".format(score_definitions))
        # if not score_definitions:
        #     raise ValueError("Something went wrong determining the metrics to plot for: results [{}]".format(', '.join(type(x).__name__ for x in results)))
        if tau_trajectories == 'all':
            tau_trajectories = ['phi', 'theta']
        self._graph_names_n_eplots.extend(self.build_metrics_graphs(results, scores=score_definitions, save=save, nb_points=nb_points, showlegend=showlegend))
        if tau_trajectories:
            self._graph_names_n_eplots.extend(self.build_tau_trajectories_graphs(results, tau_trajectories, save=save, nb_points=nb_points, showlegend=showlegend))

    def build_tau_trajectories_graphs(self, results, matrices, save=True, nb_points=None, showlegend=False):
        return [self._save_lambdas[save](self._build_graph(results, x, limit_iteration=nb_points, showlegend=showlegend)) for x in matrices]

    def build_metrics_graphs(self, results, scores='all', save=True, nb_points=None, showlegend=False):
        """
        Call this method to create and potentially save comparison plots between the tracked metrics (scores) of the given experimental results. Currently supported maximum 8 plots on the same figure\n.
        :param list of results.experimental_results.ExperimentalResults results:
        :param list or str scores: if 'all' then builds all admissible scores, else builds the custom selected scores
        :param bool save: whether to save figure on disk as .png files
        :param int nb_points: number of points to plot. Defaults to plotting all measurements found
        """
        return [self._save_lambdas[save](self._build_graph(results, x, limit_iteration=nb_points, showlegend=showlegend)) for x in scores]
        # try:
        #
        # except TypeError as e:
        #     raise TypeError("Error: {}. ".format(e))

    def _build_graph(self, exp_results_list, metric_definition, limit_iteration=None, showlegend=False):
        """Call this method to get a graph name (as a string) and a graph (as an Eplot object) tuple.\n
        :param list of results.experimental_results.ExperimentalResults exp_results_list:
        :param str metric_definition: Supports unique metric definitions such as {'perplexity', 'sparsity-theta', 'sprasity-phi-d',
            'sparsity-phi-i', 'kernel-coherence-0.80', 'top-tokens-coherence-100', ..} as well as the 'phi-tau-trajectory' and 'theta-tau-trajectory' tracked values
        :param None or int limit_iteration: whether to limit the length at which it will plot along the x axis: if None, plots all available datapoint; if int value given limits plotting at a maximum of 'value' along the x axis
        :rtype: tuple
        """
        assert len(exp_results_list) <= len(GraphMaker.LINES)
        labels = [_.scalars.model_label for _ in exp_results_list]
        self._metric_definition = metric_definition
        if metric_definition in ('phi', 'theta'):
            measure_name = 'tau_{}'.format(metric_definition)
        else:
            measure_name = self.results_handler.get_abbreviation(metric_definition).replace('-', '.')
        try:
            extractor = self._tau_traj_extractor.get(metric_definition, lambda x: self.results_handler.extract(x, metric_definition, 'all'))

            ys, xs = self._get_ys_n_xs(exp_results_list, extractor, nb_points=limit_iteration)
            return '{}-{}'.format(self._model_label_joiner.join(labels), measure_name), \
                   GraphMaker.build_graph(xs, ys,
                                          GraphMaker.LINES[:len(exp_results_list)],
                                          labels,
                                          title=measure_name,
                                          xlabel='iteration',
                                          ylabel='y',
                                          grid='on',
                                          showlegend=showlegend)
        except TypeError:
            return None, None

    def _get_ys_n_xs(self, exp_results_list, extractor, nb_points=None):
        getter = lambda x: x
        if nb_points is not None:
            getter = lambda x: x[:nb_points]
        ys = [getter(extractor(_)) for _ in exp_results_list]
        self._results_indices = list(range(len(ys)))
        return ys, [list(range(len(_))) for _ in ys]

    def determine_metrics_usable_for_comparison(self, exp_results):
        """
        If 2 or more models are found to contain information about the same metric (ie kernel-contrast-0.80) then the metric is included in the returning list.\n
        :param list exp_results:
        :return: list of strings; the metric definitions
        :rtype: list
        """
        _ = sorted([k for k, v in Counter([item for sublist in [self.results_handler.get_all_columns(model_results, self.SUPPORTED_GRAPHS) for model_results in exp_results] for item in sublist]).items() if v > 1])
        if not _:
            raise ValueError("{}".format([self.results_handler.get_all_columns(model_results, self.SUPPORTED_GRAPHS) for model_results in exp_results]))
        return _
        # return sorted([k for k, v in Counter([item for sublist in [self.results_handler.get_all_columns(model_results, self.SUPPORTED_GRAPHS) for model_results in exp_results] for item in sublist]).items() if v > 1])
        # return sorted([k for k, v in Counter(reduce(lambda i, j: i + j, [self.results_handler.get_all_columns(x, self.SUPPORTED_GRAPHS) for x in exp_results])).items() if v > 1])

    ### Create Figure Warpper object (Easyplot object in this implementation)
    @staticmethod
    def build_graph(xs, ys, line_designs, labels, title, xlabel, ylabel, grid='on', showlegend=False):
        assert len(ys) == len(xs) == len(line_designs) == len(labels)
        try:
            pl = EasyPlot(xs[0], ys[0], line_designs[0], label=labels[0], showlegend=showlegend, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError("Probably one of label: {}, xlabel: {} or ylabel: {} contains non-ascii characters".format(labels[0], xlabel, ylabel))
        for i, y_vals in enumerate(ys[1:]):
            pl.add_plot(xs[i + 1], y_vals, line_designs[i + 1], label=labels[i + 1])
        return pl

    ### Saving on disk
    def _save_plot_n_return(self, graph_label, eplot, verbose=True):
        save_path = self._save_plot(graph_label, eplot, verbose=verbose)
        return save_path, eplot

    def _save_plot(self, graph_label, eplot, verbose=True):
        """Call this method to save a graph on the disk as a .png file. The input graph type contributes to naming the file
            and appending any numbers for versioning and the plot object has a 'save' method."""
        if eplot:
            target_path = self._get_target_file_path(graph_label)
            while os.path.exists(target_path): # while old graph files are found with the same name and version number, increment the plot 'version'
                target_path = self._get_target_file_path(graph_label)
            eplot.kwargs['fig'].savefig(target_path)
            if verbose:
                print("Saved '{}' figure".format(self._metric_definition))
            return target_path
        elif verbose:
            print("Failed to save '{}' figure".format(self._metric_definition))
            return None

    # Figure path building with versioning
    def _get_target_file_path(self, graph_type):
        self._plot_counter[graph_type] += 1
        return os.path.join(self._output_dir_path, '{}_v{}.png'.format(graph_type, self._iter_prepend(self._plot_counter[graph_type])))

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits >= self._max_digits_prepend:
            return str(int_num)
        return '{}{}'.format((self._max_digits_prepend - nb_digits) * '0', int_num)

        # Advanced grid modification
        # eplot.new_plot(x, 1 / (1 + x), '-s', label=r"$y = \frac{1}{1+x}$", c='#fdb462')
        # eplot.grid(which='major', axis='x', linewidth=2, linestyle='--', color='b', alpha=0.5)
        # eplot.grid(which='major', axis='y', linewidth=2, linestyle='-', color='0.85', alpha=0.5)

        #     eplot = EasyPlot(x, tracked_metrics_dict['trackables'], 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')
        #     eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict, label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')
