#!usr/bin/python3

import os
import argparse
import matplotlib as mlp
import matplotlib.pyplot as plt

plt.ion()
from easyplot import EasyPlot


class GraphMaker(object):
    def __init__(self, plot_dir):
        self.plot = {
            'perplexity': lambda x: create_perplexity_graph(x)
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
        self.eplot = self.plot[graph_type](results)
        self._save_plot(graph_type)

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits >= self._max_digits_prepend:
            return str(int_num)
        return '{}{}'.format((self._max_digits_prepend - nb_digits) * '0', int_num)

    def _get_target_name(self, graph_type):
        return os.path.join(self.gr_dir, '{}_{}.png'.format(graph_type, self._iter_prepend(self._inds[graph_type])))

    def _save_plot(self, graph_type):
        target_name = self._get_target_name(graph_type)
        while os.path.exists(target_name):
            self._inds[graph_type] += 1
            target_name = self._get_target_name(graph_type)
        self.eplot.kwargs['fig'].savefig(target_name)
        self._inds[graph_type] += 1

#
#
# def create_coherence_plot(results):
#
#     eplot = EasyPlot(x, results['trackables'], 'b-o', label='y1 != x**2', showlegend=True, xlabel='x', ylabel='y', title='title',
#                      grid='on')

# def create_plot(results):
#
#     eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict, label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')


def create_perplexity_graph(results):
    if perplexity not in results['trackables']:
        raise NoDataGivenPlotCreationException("Key 'perplexity' not found in the result keys: {}".format(', '.join(results['trackables'].keys())))
    struct = results['trackables']['perplexity']
    return EasyPlot(range(len(struct['value'])), struct['value'], 'b-o', label='perplexity', showlegend=True, xlabel='x', ylabel='y', title='title', grid='on')


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
    parser.add_argument('collection', help='the name for the collection to train on')
    parser.add_argument('config', help='the .cfg file to use for constructing and training the topic_model')
    # parser.add_argument('category', help='the category of data to use')
    # parser.add_argument('--sample', metavar='nb_docs', default='all', help='the number of documents to consider. Defaults to all documents')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    gr = make_plots()


# eplot = EasyPlot(xlabel=r'$x$', ylabel='$y$', fontsize=16,
#                  colorcycle=["#66c2a5", "#fc8d62", "#8da0cb"], figsize=(8, 5))
# eplot.iter_plot(x, y_dict, linestyle=linestyle_dict, marker=marker_dict,
#                 label=labels_dict, linewidth=3, ms=10, showlegend=True, grid='on')
