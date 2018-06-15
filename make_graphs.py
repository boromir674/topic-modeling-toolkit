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
#         results = pickle.load(results_file)
#     assert 'collection_passes' in results and 'trackables' in results, 'root_dir' in results
#     return results

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
        # results = pickle.load(results_file)
    assert 'collection_passes' in results and 'trackables' in results, 'root_dir' in results
    return _dictify_results(results)


class GraphMaker(object):

    def __init__(self, plot_dir):
        self.line_designs = ['b-',
                             'y-',
                             'g-',
                             'p-',
                             'y-+',
                             'p-:',
                             'o-.',
                             ]
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

    def save_cross_models_plots(self, results_list):
        """
        Currently supported maximum 6 plots on the same figure\n.
        :param results_list:
        :return:
        """
        assert len(results_list) <= len(self.line_designs)
        graphs = create_cross_graphs(results_list, self.line_designs[:len(results_list)], [res['model_label'] for res in results_list])
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


#     eplot = EasyPlot(x, results['trackables'], 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')
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


def create_cross_graphs(results_list, line_design_list, labels_list):
    assert len(results_list) == len(line_design_list) == len(labels_list)
    assert len(set([res['root_dir'] for res in results_list])) <= 1
    models = '-'.join(labels_list)
    graph_plots = []
    for eval_name, sub_score2values in results_list[0]['trackables'].items():
        for sub_score, values in sub_score2values.items():
            measure_name = eval_name + '-' + sub_score
            try:
                x = range(len(values))
                graph_plots.append((
                    '{}-{}'.format(models, measure_name),
                    EasyPlot(x, values, line_design_list[0], label=labels_list[0], showlegend=True, xlabel='x', ylabel='y', title=results_list[0]['root_dir'], grid='on')))
                for j, result in enumerate(results_list[1:]):
                    graph_plots[-1][1].add_plot(x, result['trackables'][eval_name][sub_score], line_design_list[j+1], label=labels_list[j+1])
            except TypeError as e:
                print('Failed to create {} plot: type({}) = {}'.format(measure_name, sub_score, type(values)))
    return graph_plots


class NoDataGivenPlotCreationException(Exception):
    def __init__(self, msg):
        super(NoDataGivenPlotCreationException, self).__init__(msg)


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='make_graphs.py', description='Creates graphs of evaluation scores tracked along the training process and saves them to disk', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name of the collection on which experiments were conducted')
    # parser.add_argument('results', help='path to a pickle file with experimental data to plot')
    parser.add_argument('--models', metavar='model_labels', nargs='+', help='the models to compare by plotting graphs')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def _get_results_file_path(collection_root):
    return os.path.join(collection_root, 'train-res.pkl')


if __name__ == '__main__':
    args = get_cl_arguments()
    collection_root = os.path.join(collections_dir, args.collection)
    if os.path.isdir(collection_root):
        plotter = GraphMaker(os.path.join(collection_root, 'graphs'))
    else:
        print("Collection '{}' not found in {}".format(args.collection, collection_root))
        sys.exit(1)

    exp_results = []
    for model_label in args.models:
        exp_results.append(load_results(os.path.join(collection_root, 'results', model_label + '-train.pkl')))

    # plotter.save_plot('all', exp_results)

    plotter.save_cross_models_plots(exp_results)

# eplot = EasyPlot(xlabel=r'$x$', ylabel='$y$', fontsize=16,
#                  colorcycle=["#66c2a5", "#fc8d62", "#8da0cb"], figsize=(8, 5))
# eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict,
#                 label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')
