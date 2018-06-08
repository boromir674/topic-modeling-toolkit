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
                        print(type(v[in_k]))
                        v[in_k] = [eval(_) for _ in in_v]
                        print(type(v[in_k]))
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
        assert graph_type in self.plot or graph_type == 'all'
        if graph_type == 'all':
            graphs = create_graphs(results)
            for gr in graphs:
                self._save_plot(gr[1], gr[0])
        else:
            plot = self.plot[graph_type](results)
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
                grs.append((EasyPlot(range(len(values)), values, 'b-o', label=sub_plot_name, showlegend=True, xlabel='x', ylabel='y', title='title', grid='on'),
                            sub_plot_name))
            except TypeError as e:
                print('Failed to create {} plot: type({}) = {}'.format(sub_plot_name, sub_score, type(values)))
    return grs


def make_plots():
    # assert all(map(lambda x: len(x) == sum(self.collection_passes), [score_list for scores_dict in self.trackables.values() for score_list in scores_dict]))

    x = [1, 2, 3, 4, 5]
    y1 = [1, 4, 7, 9, 4]
    y2 = [2, 3, 5, 8, 9]

    eplot = EasyPlot(x, y1, 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')
    eplot.add_plot(x, y2, 'r--s', label='y2')
    eplot.kwargs['fig'].savefig('/data/thesis/data/fig.png')
    return eplot


class NoDataGivenPlotCreationException(Exception):
    def __init__(self, msg):
        super(NoDataGivenPlotCreationException, self).__init__(msg)


def get_cl_arguments():
    parser = argparse.ArgumentParser(prog='make_graphs.py', description='Creates graphs of evaluation scores tracked along the training process and saves them to disk', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('collection', help='the name of the collection on which experiments were conducted')
    # parser.add_argument('results', help='path to a pickle file with experimental data to plot')
    # parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def _get_results_file_path(collection_name):
    return os.path.join(collections_dir, collection_name, 'train-res.pkl')


if __name__ == '__main__':
    args = get_cl_arguments()
    results = load_results(_get_results_file_path(args.collection))

    plotter = GraphMaker(os.path.join(collections_dir, args.collection, 'graphs'))
    plotter.save_plot('all', results)


# eplot = EasyPlot(xlabel=r'$x$', ylabel='$y$', fontsize=16,
#                  colorcycle=["#66c2a5", "#fc8d62", "#8da0cb"], figsize=(8, 5))
# eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict,
#                 label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')
