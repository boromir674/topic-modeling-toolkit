#!usr/bin/python3

import os
import sys
import json
import argparse
import pickle
import matplotlib as mlp
import matplotlib.pyplot as plt

plt.ion()
from easyplot import EasyPlot


collections_dir = '/data/thesis/data/collections'


# def load_results(path_file):
#     with open(path_file, 'rb') as results_file:
#         res_dict = pickle.load(results_file)
#     assert 'collection_passes' in res_dict and 'trackables' in res_dict, 'root_dir' in res_dict
#     return res_dict

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
        # res_dict = pickle.load(results_file)
    assert 'collection_passes' in results and 'trackables' in results, 'root_dir' in results
    return _dictify_results(results)


class GraphMaker(object):
    def __init__(self, plot_dir):
        self.line_designs = ['b',
                             'y',
                             'g',
                             'k',
                             'c',
                             'm',
                             'p ',
                             'r'
                             ]
        self._tau_traj_extractor = {
            'phi': lambda x: self._infer_trajectory([(span, params['smooth-sparse-phi']['tau']) for span, params in x['reg_parameters']]),
            'theta': lambda x: self._infer_trajectory([(span, params['smooth-sparse-theta']['tau']) for span, params in x['reg_parameters']]),
        }

        self._cross_tau_titler = lambda x: 'tau coefficient for {} matrix sparsing regularization'.format(x)
        self._all_taus_titler = lambda x, y: 'tau coefficients for {} sparsing regularization for model {}'.format(x, y)
        # linestyle / ls: Plot linestyle['-', '--', '-.', ':', 'None', ' ', '']
        # marker: '+', 'o', '*', 's', 'D', ',', '.', '<', '>', '^', '1', '2'
        self.plot = {
            'perplexity': lambda x: create_perplexity_graph(x),
            'all': lambda x: create_graphs(x)
        }
        self._max_digits_prepend = 2
        self._inds = {key: 1 for key in self.plot.keys()}
        self.eplot = None

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

    def _infer_trajectory(self, iterspan_value_tuples):
        res = []
        for span, val in list(iterspan_value_tuples):
            res.extend([val] * span)
        return res

    def save_tau_trajectories(self, results, cross=True, nb_points=None):
        """Plots tau coefficient value trajectory for sparsing phi and theta matrices and saves to disk"""
        assert len(results) <= len(self.line_designs)
        # assert len(set([res['root_dir'] for res in results_list])) <= 1
        graph_plots = []
        _vals = []
        if cross:
            for tau_type, extractor in self._tau_traj_extractor.items():
                values = extractor(results[0])[:nb_points]
                _vals.append(values)
                x = range(len(values[:nb_points]))
                graph_plots.append((
                                    '{}-tau-{}'.format('-'.join(map(lambda x: x['model_label'], results)), tau_type),
                                    EasyPlot(x, values[:nb_points], self.line_designs[0], label=results[0]['model_label'],
                                             showlegend=True,
                                             xlabel='x', ylabel='y', title=self._cross_tau_titler(tau_type), grid='on')
                ))

                for i, exp_res in enumerate(results[1:]):
                    values = extractor(exp_res)[:nb_points]
                    _vals.append(values)
                    graph_plots[-1][1].add_plot(x, values, self.line_designs[i + 1], label=exp_res['model_label'][i + 1])
        else:
            for res in results:
                print(type(res['reg_parameters']))
                print(res['reg_parameters'][0])
                values = list(self._tau_traj_extractor['phi'](res))
                _vals.append(values)
                x = range(len(values[:nb_points]))
                el = '.'.join(sorted(self._tau_traj_extractor.keys()))
                graph_plots.append((
                    '{}-{}-taus'.format(res['model_label'], el),
                    EasyPlot(x, values[:nb_points], self.line_designs[0], label='phi',
                             showlegend=True,
                             xlabel='x', ylabel='y', title=self._all_taus_titler(el, res['model_label']), grid='on')
                ))
                values = list(self._tau_traj_extractor['theta'](res))
                _vals.append(values)
                graph_plots[-1][1].add_plot(x, values, self.line_designs[1], label='theta')

        assert all(len(i) == len(_vals[0]) for i in _vals)
        for graph_type, plot in graph_plots:
            self._save_plot(graph_type, plot)

    # def create_easy_plot(self):
    def save_plot(self, graph_type, results):
        """
        Creates plots of various metrics and saves png files
        :param graph_type:
        :param results:
        :return:
        """
        assert graph_type in self.plot or graph_type == 'all'
        if graph_type == 'all':
            graphs = create_graphs(results)
            for gr in graphs:
                self._save_plot(gr[0], gr[1])
        else:
            plot = self.plot[graph_type](results)
            self._save_plot(graph_type, plot)

    def save_cross_models_plots(self, results, nb_points=None):
        """
        Currently supported maximum 8 plots on the same figure\n.
        :param list of dicts results: experimental results, gathered after model training
        :param int nb_points: number of points to plot. Defaults to plotting all measurements found
        """
        assert len(results) <= len(self.line_designs)
        graphs = create_cross_graphs(results, self.line_designs[:len(results)], [res['model_label'] for res in results], limit_iterations=nb_points)
        for graph_type, plot in graphs:
            self._save_plot(graph_type, plot)

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits >= self._max_digits_prepend:
            return str(int_num)
        return '{}{}'.format((self._max_digits_prepend - nb_digits) * '0', int_num)

    def _get_target_name(self, graph_type):
        if graph_type not in self._inds:
            self._inds[graph_type] = 1
        return os.path.join(self.gr_dir, '{}_{}.png'.format(graph_type, self._iter_prepend(self._inds[graph_type])))

    def _save_plot(self, graph_type, eplot):
        target_name = self._get_target_name(graph_type)
        while os.path.exists(target_name):
            self._inds[graph_type] += 1
            target_name = self._get_target_name(graph_type)
        eplot.kwargs['fig'].savefig(target_name)
        print('Saved figure as', target_name)
        self._inds[graph_type] += 1

    # Advanced grid modification
    # eplot.new_plot(x, 1 / (1 + x), '-s', label=r"$y = \frac{1}{1+x}$", c='#fdb462')
    # eplot.grid(which='major', axis='x', linewidth=2, linestyle='--', color='b', alpha=0.5)
    # eplot.grid(which='major', axis='y', linewidth=2, linestyle='-', color='0.85', alpha=0.5)


#     eplot = EasyPlot(x, res_dict['trackables'], 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')
#     eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict, label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')


def create_perplexity_graph(results):
    if 'perplexity' not in results['trackables']:
        raise NoDataGivenPlotCreationException("Key 'perplexity' not found in the result keys: {}".format(', '.join(results['trackables'].keys())))
    struct = results['trackables']['perplexity']
    return EasyPlot(range(len(struct['value'])), struct['value'], 'b-o', label='perplexity', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')


def create_graphs(results):
    grs = []
    for eval_name, sub_score2values in results['trackables'].items():
        for sub_score, values in sub_score2values.items():
            sub_plot_name = eval_name+'-'+sub_score
            try:
                grs.append((
                    sub_plot_name,
                    EasyPlot(range(len(values)), values, 'b-o', label=sub_plot_name, showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')))
            except TypeError as e:
                print('Failed to create {} plot: type({}) = {}'.format(sub_plot_name, sub_score, type(values)))
    return grs


def create_cross_graphs(results_list, line_design_list, labels_list, limit_iterations=None):
    assert len(results_list) == len(line_design_list) == len(labels_list)
    assert len(set([res['root_dir'] for res in results_list])) <= 1
    models = '-'.join(labels_list)
    graph_plots = []
    _vals = []
    for eval_name, sub_score2values in results_list[0]['trackables'].items():
        for sub_score, values in sub_score2values.items():
             # takes the first 'limit_iterations' metric values
            _vals.append(values)
            measure_name = eval_name + '-' + sub_score
            try:
                x = range(len(values[:limit_iterations]))
                graph_plots.append((
                    '{}-{}'.format(models, measure_name),
                    EasyPlot(x, values[:limit_iterations], line_design_list[0], label=labels_list[0], showlegend=True,
                             xlabel='x', ylabel='y', title=measure_name.replace('-', '.'), grid='on')))
                for j, result in enumerate(results_list[1:]):
                    graph_plots[-1][1].add_plot(x, result['trackables'][eval_name][sub_score][:limit_iterations], line_design_list[j+1], label=labels_list[j+1])
            except TypeError as e:
                print('Failed to create {} plot: type({}) = {}'.format(measure_name, sub_score, type(values)))
    assert all(len(i) == len(_vals[0]) for i in _vals)
    return graph_plots


class NoDataGivenPlotCreationException(Exception):
    def __init__(self, msg):
        super(NoDataGivenPlotCreationException, self).__init__(msg)


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='make_graphs.py', description='Creates graphs of evaluation scores tracked along the training process and saves them to disk', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name of the collection on which experiments were conducted')
    parser.add_argument('--models', metavar='model_labels', nargs='+', help='the models to compare by plotting graphs')
    parser.add_argument('--metrics', action='store_true', default=False, help='Enable pltotting of all possible metrics tracked')
    parser.add_argument('--tau-trajectories', '-t', action='store_true', dest='plot_tau_trajectories', default=False, help='Enable ploting of the dynamic tau coefficients\' trajectories')
    parser.add_argument('--iterations', '-i', metavar='nb_points', type=int, help='limit or not the dapoints plotted, to the specified number')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def _get_results_file_path(collection_root):
    return os.path.join(collection_root, 'train-res.pkl')


if __name__ == '__main__':
    args = get_cl_arguments()
    col_root = os.path.join(collections_dir, args.collection)
    if os.path.isdir(col_root):
        plotter = GraphMaker(os.path.join(col_root, 'graphs'))
    else:
        print("Collection '{}' not found in {}".format(args.collection, col_root))
        sys.exit(1)

    exp_results = []
    for model_label in args.models:
        exp_results.append(load_results(os.path.join(col_root, 'results', model_label + '-train.json')))

    # plotter.save_plot('all', exp_results)
    # print(len(results_list[0]['trackables']['perplexity']['value']))
    # assert len(results_list[0]['trackables']['perplexity']['value']) == 150
    print(type(args.iterations))
    if args.metrics:
        plotter.save_cross_models_plots(exp_results, nb_points=args.iterations)
    if args.plot_tau_trajectories:
        # plotter.save_tau_trajectories(exp_results, nb_points=args.iterations, cross=True)
        plotter.save_tau_trajectories(exp_results, nb_points=args.iterations, cross=False)

# eplot = EasyPlot(xlabel=r'$x$', ylabel='$y$', fontsize=16,
#                  colorcycle=["#66c2a5", "#fc8d62", "#8da0cb"], figsize=(8, 5))
# eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict,
#                 label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')
