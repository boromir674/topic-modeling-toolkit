

class GraphMaker(object):
    SUPPORTED_GRAPHS = ['perplexity', '']

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
            'phi': lambda x: _infer_trajectory_values_list([(span, params['sparse-phi']['tau']) for span, params in x['reg_parameters']]) if 'sparse-phi' in x['reg_parameters'][-1][1] else None,
            'theta': lambda x: _infer_trajectory_values_list([(span, params['sparse-theta']['tau']) for span, params in x['reg_parameters']]) if 'sparse-theta' in x['reg_parameters'][-1][1] else None,
        }
        self._cross_tau_titler = lambda x: 'Coefficient tau for {}-matrix-sparsing regularizer'.format(x)
        self._results_indices = []
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
            ys, xs = self._build_ys_n_xs(results, extractor, nb_points=nb_points)
            print("Ommiting '{}' model(s) for not having a {}-matrix-sparse regularizer".format(', '.join([labels_list[_] for _ in range(len(results)) if _ not in self._results_indices]), tau_type))
            graph_plots.append(('{}-tau-{}'.format('-'.join(map(lambda x: x['model_label'], results)), tau_type),
                                _build_graph(xs, ys, [self.line_designs[_] for _ in self._results_indices], [labels_list[_] for _ in self._results_indices], self._cross_tau_titler(tau_type), 'iteration', 'τ')))
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
                print("Try a metric in [{}]".format(', '.join(sorted(results[0]['trackables'].keys()))))

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
                ys, xs = self._build_ys_n_xs(results, lambda x: x['trackables'][score][sub_score], limit_iteration)
                graph_plots.append(('{}-{}'.format('-'.join(labels_list), measure_name),
                                    _build_graph(xs, ys, self.line_designs[:len(results)], labels_list, measure_name.replace('-', '.'), 'iteration', 'y')))
            except TypeError:
                print("Not creating sub-score '{}' plot of '{}', because either the format is not supported or the average of the same metric is plotted".format(sub_score, score))
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

    def _build_ys_n_xs(self, results, extractor, nb_points=None):
        _ = list(map(extractor, results))
        self._results_indices = [ind for ind, el in enumerate(_) if el is not None]
        ys = list(map(lambda x: self._limit_points(x, nb_points), filter(None, _)))

        return ys, [range(len(_)) for _ in ys]

    def _limit_points(self, value_list, limit):
        if limit is None:
            return value_list
        else:
            return value_list[:limit]

    def _iter_prepend(self, int_num):
        nb_digits = len(str(int_num))
        if nb_digits >= self._max_digits_prepend:
            return str(int_num)
        return '{}{}'.format((self._max_digits_prepend - nb_digits) * '0', int_num)
