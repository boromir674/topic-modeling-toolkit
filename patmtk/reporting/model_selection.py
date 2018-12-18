import os
import re
from glob import glob
from functools import reduce
from collections import Counter
import numpy as np

from results import ExperimentalResults

from .fitness import FitnessFunction


KERNEL_SUB_ENTITIES = ('coherence', 'contrast', 'purity')

MAX_DECIMALS = 2  # this consant should agree with the patm.modeling.experiment.Experiment.MAX_DECIMALS

def _get_kernel_sub_hash(entity):
    assert entity in KERNEL_SUB_ENTITIES
    return {'kernel-'+entity: {'scalar-extractor': lambda x,y: getattr(getattr(x.tracked, 'kernel'+y[2:]).average, entity).last if hasattr(x.tracked, 'kernel'+y[2:]) else None,
                               'list-extractor': lambda x, y: getattr(getattr(x.tracked, 'kernel' + y[2:]).average, entity).all if hasattr(x.tracked, 'kernel' + y[2:]) else None,
                               'column-title': lambda x: 'k'+entity[:3:2]+'.'+str(x)[2:],
                               'to-string': '{:.4f}',
                               'definitions': lambda x: ['kernel-{}-{}'.format(entity, y) for y in x.tracked.kernel_thresholds]}}

COLUMNS_HASH = {
    'nb-topics': {'scalar-extractor': lambda x: x.scalars.nb_topics,
                  'column-title': lambda: 'tpcs'},
    'collection-passes': {'scalar-extractor': lambda x: x.scalars.dataset_iterations,
                          'column-title': lambda: 'col-i'},
    'document-passes': {'scalar-extractor': lambda x: x.scalars.document_passes,
                        'column-title': lambda: 'doc-i'},
    'total-phi-updates': {'scalar-extractor': lambda x: x.scalars.dataset_iterations * x.scalars.document_passes,
                          'column-title': lambda: 'phi-u'},
    'perplexity': {'scalar-extractor': lambda x: x.tracked.perplexity.last,
                   'list-extractor': lambda x: x.tracked.perplexity.all,
                   'column-title': lambda: 'prpl',
                   'to-string': '{:.1f}'},
    'top-tokens-coherence': {'scalar-extractor': lambda x,y: getattr(x.tracked, 'top'+str(y)).average_coherence.last if hasattr(x.tracked, 'top'+str(y)) else None,
                             'list-extractor': lambda x,y: getattr(x.tracked, 'top'+str(y)).average_coherence.all if hasattr(x.tracked, 'top'+str(y)) else None,
                             'column-title': lambda x: 'top'+str(x)+'ch',
                             'to-string': '{:.4f}',
                             'definitions': lambda x: ['top-tokens-coherence-'+str(y) for y in x.tracked.top_tokens_cardinalities]},
    'sparsity-phi': {'scalar-extractor': lambda x,y: getattr(x.tracked, 'sparsity_phi_'+y).last if hasattr(x.tracked, 'sparsity_phi_'+y) else None,
                     'list-extractor': lambda x,y: getattr(x.tracked, 'sparsity_phi_'+y).all if hasattr(x.tracked, 'sparsity_phi_'+y) else None,
                     'column-title': lambda y: 'spp@'+y,
                     'to-string': '{:.2f}',
                     'definitions': lambda x: ['sparsity-phi-{}'.format(y) for y in x.tracked.modalities_initials]},
    'sparsity-theta': {'scalar-extractor': lambda x: x.tracked.sparsity_theta.last,
                       'list-extractor': lambda x: x.tracked.sparsity_theta.all,
                       'column-title': lambda: 'spt',
                       'to-string': '{:.2f}'},
    'background-tokens-ratio': {'scalar-extractor': lambda x,y: getattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]).last if hasattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]) else None,
                                # 'list-extractor': lambda x,y: getattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]).all if hasattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]) else None,
                                'list-extractor': lambda x,y: getattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]).all if hasattr(x.tracked, 'background_tokens_ratio_'+str(y)[2:]) else None,
                                'column-title': lambda x: 'btr.'+str(x)[2:],
                                'to-string': '{:.2f}',
                                'definitions': lambda x: ['background-tokens-ratio-{}'.format(y[:4]) for y in x.tracked.background_tokens_thresholds]},
    'regularizers': {'scalar-extractor': lambda x: '[{}]'.format(', '.join(x.regularizers)),
                     'column-title': lambda: 'regs'}
}

COLUMNS_HASH = reduce(lambda x, y: dict(y, **x), [COLUMNS_HASH] + [_get_kernel_sub_hash(z) for z in KERNEL_SUB_ENTITIES])


class ResultsHandler(object):
    _list_selector_hash = {str: lambda x: x[0] if x[1] == 'all' else None,
                           range: lambda x: [x[0][_] for _ in x[1]],
                           int: lambda x: x[0][:x[1]],
                           list: lambda x: [x[0][_] for _ in x[1]]}
    _QUANTITY_2_EXTRACTOR_KEY = {'last': 'scalar', 'all': 'list'}
    DYNAMIC_COLUMNS = ['kernel-coherence', 'kernel-contrast', 'kernel-purity', 'top-tokens-coherence', 'sparsity-phi', 'background-tokens-ratio']
    DEFAULT_COLUMNS = ['nb-topics', 'collection-passes', 'document-passes', 'total-phi-updates', 'perplexity'] +\
                      DYNAMIC_COLUMNS[:-1] + ['sparsity-theta'] + [DYNAMIC_COLUMNS[-1]] + ['regularizers']

    def __init__(self, collection_root_path, results_dir_name='results'):
        self._collection_root_path = collection_root_path
        self._results_dir_name = results_dir_name
        self._results_hash = {}
        self._fitness_function_hash = {}
        self._list_selector = None

    def get_experimental_results(self, collection_name, sort='', selection='all'):
        """
        Call this method to get a list of experimental result objects from topic models trained on the given collection.\n
        :param str collection_name:
        :param str sort:
        :param str or range or int or list selection: whether to select a subset of the experimental results fron the given collection\n
            - if selection == 'all', returns every experimental results object "extracted" from the jsons
            - if type(selection) == range, returns a "slice" of the experimental results based on the range
            - if type(selection) == int, returns the first n experimental results
            - if type(selection) == list, then it represents specific indices to sample the list of experimental results from
        :return: the ExperimentalResults objects
        :rtype: list
        """
        result_paths = glob('{}/*.json'.format(os.path.join(self._collection_root_path, collection_name, self._results_dir_name)))
        # print("GET exp results: selection '{}' of type {}".format(selection, type(selection)))
        # print('All models are {}'.format(len(result_paths)))
        if type(selection) == list and all([type(x) == 'str' for x in selection]):
            self._list_selector = lambda y: ResultsHandler._label_selection(selection, y)
        self._list_selector = lambda y: ResultsHandler._list_selector_hash[type(selection)]([y, selection])
        r = self._get_experimental_results(result_paths, callable_metric=self._get_metric(sort))
        assert len(result_paths) == len(r)
        return self._list_selector(self._get_experimental_results(result_paths, callable_metric=self._get_metric(sort)))

    def _get_experimental_results(self, results_paths, callable_metric=None):
        if callable_metric:
            return sorted([self._process_result_path(x) for x in results_paths], key=callable_metric, reverse=True)
        return [self._process_result_path(_) for _ in results_paths]

    def _process_result_path(self, result_path):
        if result_path not in self._results_hash:
            self._results_hash[result_path] = ExperimentalResults.create_from_json_file(result_path)
        return self._results_hash[result_path]

    def _get_metric(self, metric):
        if not metric:
            return None
        if metric not in self._fitness_function_hash:
            self._fitness_function_hash[metric] = FitnessFunction.single_metric(metric)
        return lambda x: self._fitness_function_hash[metric].compute([ResultsHandler.extract(x, metric, 'last')])

    @staticmethod
    def get_titles(column_definitions):
        return [ResultsHandler.get_abbreviation(x) for x in column_definitions]

    @staticmethod
    def get_abbreviation(definition):
        tokens, parameters = ResultsHandler._parse_column_definition(definition)
        return COLUMNS_HASH['-'.join(tokens)]['column-title'](*parameters)

    @staticmethod
    def stringnify(column, value):
        """
        :param str column: key or definition; example values: 'perplexity', 'kernel-coherence', 'kernel-coherence-0.80'
        :param value:
        :return:
        :rtype str
        """
        return COLUMNS_HASH.get(column, COLUMNS_HASH[ResultsHandler._get_hash_key(column)]).get('to-string', '{}').format(value)

    # @staticmethod
    # def get_tau_trajectory(exp_results, matrix_name):
    #     """
    #
    #     :param results.experimental_results.ExperimentalResults exp_results:
    #     :param matrix_name:
    #     :return:
    #     """
    #     return getattr(exp_results.tracked.tau_trajectories, matrix_name).all

    @staticmethod
    def extract(exp_results, column_definition, quantity):
        """
        Call this method to query the given experimental results object about a specific metric. Supports requesting all
        values tracked along the training process.
        :param results.experimental_results.ExperimentalResults exp_results:
        :param str column_definition:
        :param str quantity: must be one of {'last', 'all'}
        :return:
        """
        tokens, parameters = ResultsHandler._parse_column_definition(column_definition)
        return COLUMNS_HASH['-'.join(tokens)][ResultsHandler._QUANTITY_2_EXTRACTOR_KEY[quantity] + '-extractor'](*list([exp_results] + parameters))

    @staticmethod
    def get_all_columns(exp_results, requested_entities):
        return reduce(lambda i, j: i + j,
                      [COLUMNS_HASH[x]['definitions'](exp_results) if x in ResultsHandler.DYNAMIC_COLUMNS else [x] for x
                       in requested_entities])

    ###### UTILITY FUNCTIONS ######
    @staticmethod
    def _parse_column_definition(definition):
        return [list([_f for _f in y if _f]) for y in zip(*[(x, None) if ResultsHandler._is_token(x) else (None, x) for x in definition.split('-')])]

    @staticmethod
    def _get_hash_key(column_definition):
        return '-'.join([_ for _ in column_definition.split('-') if ResultsHandler._is_token(_)])

    @staticmethod
    def _is_token(definition_element):
        try:
            _ = float(definition_element)
            return False
        except ValueError:
            if definition_element[0] == '@' or len(definition_element) == 1:
                return False
            return True

    @staticmethod
    def _label_selection(labels, experimental_results_list):
        _ = [x.scalars.model_label for x in experimental_results_list]
        return [experimental_results_list.index(_.index(i)) for i in labels if i in _]


# class ModelSelector(object):
#
#     def __init__(self, collection_results_dir_path=''):
#         self._working_dir = ''
#         self._results_paths, self._available_model_labels = [], []
#
#         if collection_results_dir_path:
#             self.working_dir = collection_results_dir_path
#
#     @property
#     def working_dir(self):
#         return self._working_dir
#
#     @working_dir.setter
#     def working_dir(self, results_dir_path):
#         self._working_dir = results_dir_path
#         self._results_paths = glob('{}/*.json'.format(self._working_dir))
#         if not self._results_paths:
#             raise NoTrainingResultsFoundException("No saved training results found in '{}' directory.".format(collection_results_dir_path))
#         self._available_model_labels = list(map(lambda x: re.search('/([\w\-.]+)\.json$', x).group(1), self._results_paths))
#
#     def select_n_get(self, expression='all'):
#         if expression == 'all':
#             return self._available_model_labels
#         return []

# class NoTrainingResultsFoundException(Exception):
#     def __init__(self, msg):
#         super(NoTrainingResultsFoundException, self).__init__(msg)


if __name__ == '__main__':
    ms = ModelSelector(collection_results_dir_path='/data/thesis/data/collections/dd/results')
